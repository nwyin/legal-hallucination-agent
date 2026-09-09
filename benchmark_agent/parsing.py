"""Parsing model output: JSON extraction, prediction parsing, and action validation.

Action responses are validated against the Pydantic `ActionChoice` schema; on
failure the agent re-asks the model with the validation error (see
`BayesianOptimalExperimentalDesignAgent._call_llm_for_action_selection`).
"""

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as ValidationError

from .actions import ActionType

logger = logging.getLogger(__name__)


# --- JSON and prediction response helpers ---


def parse_json_from_text(
    text: str, response_name: str = "response"
) -> dict[str, Any] | None:
    """Decode the first object, preferring one inside a ```json fence.

    Returns None if there is no object opener. Malformed or incomplete JSON at
    the selected opener raises ValueError; surrounding text is ignored.
    """
    fenced = re.search(r"```(?:json)?\s*\{", text, flags=re.IGNORECASE)
    start = fenced.end() - 1 if fenced else text.find("{")
    if start == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text, start)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {response_name} response: {e}") from e


def parse_json_response(
    response: str, response_name: str = "response"
) -> dict[str, Any]:
    """Parse a JSON response; raises ValueError if empty or no valid JSON object is present."""
    if not response or not response.strip():
        raise ValueError(f"Empty {response_name} response from model")
    logger.debug(f"Raw {response_name} response: '{response}'")
    data = parse_json_from_text(response, response_name)
    if data is None:
        raise ValueError(f"No JSON object found in {response_name} response")
    return data


def parse_prediction_response(response: str) -> tuple[str, float]:
    """Parse {"action": {"response": ...}, "confidence": ...}; list responses become JSON strings.

    Raises ValueError if the response is empty, invalid JSON, or has an empty prediction.
    """
    data = parse_json_response(response, "prediction")
    raw_prediction = data.get("action", {}).get("response", "")
    if isinstance(raw_prediction, list):
        prediction = json.dumps(
            raw_prediction
        )  # Preserve as JSON string for downstream parsing
    else:
        prediction = str(raw_prediction).strip() if raw_prediction else ""
    confidence = float(data.get("confidence", 0.0))
    if not prediction:
        raise ValueError("Empty prediction in response")
    pred_preview = prediction[:200] + "..." if len(prediction) > 200 else prediction
    logger.info(f"Parsed prediction: {pred_preview} (confidence: {confidence:.3f})")
    return prediction, confidence


def normalize_yes_no_answer(answer: str) -> str:
    """Normalize a Yes/No variant to "Yes" or "No"; raises ValueError otherwise."""
    answer_upper = answer.upper().strip()
    if answer_upper in ["YES", "Y"]:
        return "Yes"
    if answer_upper in ["NO", "N"]:
        return "No"
    raise ValueError(f"Invalid answer format: '{answer}'. Expected 'Yes' or 'No'")


def normalize_yes_no_answers(answers: str) -> str:
    """Normalize semicolon-delimited Yes/No answers (e.g. "Yes; No; Yes")."""
    return "; ".join(normalize_yes_no_answer(a) for a in answers.split(";"))


# --- Action schema (discriminated union on action_type) ---


def _non_empty(field_name: str, example: str):
    @field_validator(field_name, mode="after")
    @classmethod
    def validate(cls, v: str) -> str:
        if not (v and str(v).strip()):
            raise ValueError(
                f"{field_name} must be a non-empty string (e.g. {example})"
            )
        return str(v).strip()

    return validate


class ProvideFinalResponseAction(BaseModel):
    action_type: Literal[ActionType.PROVIDE_FINAL_RESPONSE.value] = Field(
        ..., description="The type of action"
    )
    response: str | list[str] = Field(
        ...,
        description="The final response: a string, or a list of hallucinated citations/sentences",
    )


class ThinkAction(BaseModel):
    action_type: Literal[ActionType.THINK.value] = Field(
        ..., description="The type of action"
    )
    thought: str = Field(
        ...,
        description="Your reasoning, analysis, or thought process about the current situation",
    )


class OpenWebSearchAction(BaseModel):
    action_type: Literal[ActionType.OPEN_WEB_SEARCH.value] = Field(
        ..., description="The type of action"
    )
    query: str = Field(
        ...,
        description="The search query to find relevant web information (must be non-empty, e.g. a citation or search phrase)",
    )
    search_type: str | None = Field(
        None,
        description="The type of search to perform. Options: 'web', 'news', 'google_scholar' (default: 'web')",
    )
    num_results: int | None = Field(
        None, description="Number of results to return (default: 10)"
    )
    news_source: str | None = Field(
        None,
        description="News source to use for news searches. Options: 'serpapi', 'mediastack' (default: 'serpapi')",
    )
    cutoff_date: str | None = Field(
        None,
        description="Cutoff date for search results (YYYY-MM-DD format). Only results published on or before this date will be returned. Optional.",
    )
    exclude_undated: bool | None = Field(
        None,
        description="If true, exclude results without publication dates (default: true)",
    )

    query_non_empty = _non_empty("query", "a citation or search phrase")


class OpenCourtListenerSearchAction(BaseModel):
    action_type: Literal[ActionType.OPEN_COURTLISTENER_SEARCH.value] = Field(
        ..., description="The type of action"
    )
    query: str = Field(
        ...,
        description="The search query to find relevant legal cases and documents (must be non-empty, e.g. a citation like '965 F.2d 962' or a case name)",
    )
    search_type: str | None = Field(
        None,
        description="The type of search to perform. Options: 'opinions', 'cases', 'dockets', 'filings', 'judges', 'oral_arguments' (default: 'opinions')",
    )
    court: str | None = Field(
        None,
        description="The court to search (e.g., 'scotus', 'ca1', 'ca2') (default: 'scotus')",
    )
    date_filter: dict[str, Any] | None = Field(
        None,
        description="Date filter with 'field' and 'before'/'after' keys (e.g., {'field': 'date_filed', 'before': '2022-01-01', 'after': '2021-01-01'}). Dates in YYYY-MM-DD format. Only 'date_filed' field is supported.",
    )

    query_non_empty = _non_empty(
        "query", "a citation like '965 F.2d 962' or a case name"
    )


class CourtListenerCitationLookupAction(BaseModel):
    action_type: Literal[ActionType.COURTLISTENER_CITATION_LOOKUP.value] = Field(
        ..., description="The type of action"
    )
    cite: str = Field(
        ...,
        description="The reporter citation to look up (e.g. '934 F.3d 53', '143 S. Ct. 1196')",
    )

    cite_non_empty = _non_empty("cite", "'965 F.2d 962' or '143 S. Ct. 1196'")


class AccessCourtListenerOpinionAction(BaseModel):
    action_type: Literal[ActionType.ACCESS_COURTLISTENER_OPINION.value] = Field(
        ..., description="The type of action"
    )
    opinion_id: str = Field(
        ...,
        description="The CourtListener opinion ID (obtained from OPEN_COURTLISTENER_SEARCH results)",
    )


class SearchLocalOpinionAction(BaseModel):
    action_type: Literal[ActionType.SEARCH_LOCAL_OPINION.value] = Field(
        ..., description="The type of action"
    )
    opinion_id: str = Field(
        ...,
        description="The CourtListener opinion ID (from a previous ACCESS_COURTLISTENER_OPINION call)",
    )
    search_string: str = Field(
        ..., description="The string to search for in the opinion text"
    )


class ReadDocumentAction(BaseModel):
    action_type: Literal[ActionType.READ_DOCUMENT.value] = Field(
        ..., description="The type of action"
    )
    opinion_id: str = Field(
        ...,
        description="The opinion to read: use opinion_<id> (e.g. opinion_9001448) or just the numeric id (e.g. 9001448) from a previous ACCESS_COURTLISTENER_OPINION.",
    )
    start_line: int = Field(
        ...,
        description="The starting line number to read from (0-indexed). Use 0 for the first line.",
    )
    num_lines: int = Field(
        ...,
        description="The number of lines to read from the document starting from start_line. Must be a positive integer.",
    )


class EditScratchpadAction(BaseModel):
    action_type: Literal[ActionType.EDIT_SCRATCHPAD.value] = Field(
        ..., description="The type of action"
    )
    operation: str = Field(
        ...,
        description="The operation to perform on the scratchpad. Options: 'append', 'insert', 'replace', 'clear'",
    )
    content: str = Field(
        ..., description="The content to add, insert, or replace in the scratchpad"
    )
    position: int | None = Field(
        None,
        description="The position for insert/replace operations (0-indexed). Required for 'insert' and 'replace' operations. Ignored for 'append' and 'clear' operations. Optional.",
    )


ActionUnion = (
    ProvideFinalResponseAction
    | ThinkAction
    | OpenWebSearchAction
    | OpenCourtListenerSearchAction
    | CourtListenerCitationLookupAction
    | AccessCourtListenerOpinionAction
    | SearchLocalOpinionAction
    | ReadDocumentAction
    | EditScratchpadAction
)


class ActionChoice(BaseModel):
    """The model's action-selection response: {"action": {"action_type": ..., <params>}, ...}."""

    action: ActionUnion = Field(..., discriminator="action_type")


def parse_action_response(response: str) -> tuple[str, dict[str, Any]]:
    """Parse and validate an action-selection response into (action_type, parameters).

    Parameters omitted by the model are left out so Action class defaults apply.
    Raises ValueError (no/invalid JSON) or pydantic.ValidationError (schema mismatch).
    """
    data = parse_json_response(response, "action selection")
    action = ActionChoice.model_validate(data).action
    parameters = action.model_dump(exclude_none=True)
    return parameters.pop("action_type"), parameters


def normalize_action_parameters_for_construction(
    action_type: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """
    Normalize parameters so they match Action class constructors.
    E.g. ProvideFinalResponse expects response: str; we accept list from the model and store as JSON array string.
    Strip num_results/k from OPEN_COURTLISTENER_SEARCH - the environment uses search_top_k from config.
    Reject empty query for search actions.
    """
    params = dict(parameters)
    if action_type == ActionType.PROVIDE_FINAL_RESPONSE.value and "response" in params:
        r = params["response"]
        if isinstance(r, list):
            params["response"] = json.dumps(r)
    # OPEN_COURTLISTENER_SEARCH doesn't accept num_results/k; environment uses search_top_k from config
    if action_type == ActionType.OPEN_COURTLISTENER_SEARCH.value:
        params.pop("num_results", None)
        params.pop("k", None)
    # Require non-empty query for search actions
    if action_type in (
        ActionType.OPEN_COURTLISTENER_SEARCH.value,
        ActionType.OPEN_WEB_SEARCH.value,
    ):
        q = params.get("query")
        if not (q is not None and str(q).strip()):
            raise ValueError(
                "query is required and must be non-empty for OPEN_COURTLISTENER_SEARCH and OPEN_WEB_SEARCH "
                "(e.g. a citation like '965 F.2d 962' or a case name)"
            )
        params["query"] = str(q).strip()
    # Require non-empty cite for citation lookup
    if action_type == ActionType.COURTLISTENER_CITATION_LOOKUP.value:
        c = params.get("cite")
        if not (c is not None and str(c).strip()):
            raise ValueError(
                "cite is required and must be non-empty for COURTLISTENER_CITATION_LOOKUP "
                "(e.g. '965 F.2d 962' or '143 S. Ct. 1196')"
            )
        params["cite"] = str(c).strip()
    # READ_DOCUMENT: accept opinion_id; allow legacy document_id for backward compatibility
    if (
        action_type == ActionType.READ_DOCUMENT.value
        and "opinion_id" not in params
        and "document_id" in params
    ):
        params["opinion_id"] = params.pop("document_id")
    return params
