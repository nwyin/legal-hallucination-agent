"""Parsing model output: JSON extraction, prediction parsing, and the action guard."""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from guardrails import Guard
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
)
from pydantic import BaseModel, Field, ValidationError, field_validator

from .actions import ActionType, Action

logger = logging.getLogger(__name__)


# --- JSON and prediction response helpers ---


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON string from text, handling code blocks and standalone JSON.
    Uses proper brace matching to handle nested JSON objects.
    
    Args:
        text: Text that may contain JSON (possibly in code blocks or with extra text)
        
    Returns:
        Extracted JSON string, or None if no valid JSON found
    """
    json_str = None
    
    # Try to find JSON in code block first
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?)\s*```', text, flags=re.IGNORECASE | re.DOTALL)
    if code_block_match:
        # Extract JSON with proper brace matching
        brace_start = code_block_match.start(1)
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[brace_start:i + 1]
                    break
    
    # If no code block, try standalone JSON with brace matching
    if not json_str:
        brace_start = text.find('{')
        if brace_start != -1:
            brace_count = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:i + 1]
                        break
    
    return json_str


def parse_json_from_text(text: str, response_name: str = "response") -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from text that may contain code blocks or extra text.
    Uses proper brace matching to handle nested JSON objects.
    
    Args:
        text: Text that may contain JSON (possibly in code blocks or with extra text)
        response_name: Name of the response type for logging
        
    Returns:
        Parsed JSON data as dictionary, or None if no valid JSON found
        
    Raises:
        ValueError: If JSON is found but invalid
    """
    json_str = extract_json_from_text(text)
    
    if not json_str:
        return None
    
    try:
        data = json.loads(json_str)
        logger.debug(f"Successfully extracted and parsed {response_name} from text")
        return data
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues before giving up
        # Fix trailing commas
        json_str_fixed = json_str.replace(',}', '}').replace(',]', ']')
        
        # Try parsing again with fixed trailing commas
        try:
            data = json.loads(json_str_fixed)
            logger.debug(f"Successfully parsed {response_name} after fixing trailing commas")
            return data
        except json.JSONDecodeError:
            # If still failing, raise the original error with context
            raise ValueError(f"Invalid JSON in {response_name} response: {e}")


def parse_json_response(response: str, response_name: str = "response") -> Dict[str, Any]:
    """
    Parse a JSON response with common cleanup and error handling.
    
    Args:
        response: Raw response string from LLM
        response_name: Name of the response type for logging (e.g., "prediction")
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        ValueError: If response is empty or invalid JSON
        KeyError: If required fields are missing
    """
    if not response or response.strip() == "":
        raise ValueError(f"Empty {response_name} response from model")
    
    logger.debug(f"Raw {response_name} response: '{response}'")
    
    # First try the robust extraction method (handles code blocks, extra text, etc.)
    try:
        data = parse_json_from_text(response, response_name)
        if data is not None:
            return data
    except ValueError:
        # If extraction found JSON but it's malformed, try fallback cleanup
        pass
    
    # Fallback to simple cleanup for cases where extraction didn't find JSON or parsing failed
    # Clean up markdown formatting if present
    response_clean = response.strip()
    if response_clean.startswith("```json"):
        response_clean = response_clean[7:]
    if response_clean.endswith("```"):
        response_clean = response_clean[:-3]
    response_clean = response_clean.strip()
    
    # Fix common JSON issues (trailing commas, etc.)
    response_clean = response_clean.replace(',}', '}').replace(',]', ']')
    
    try:
        data = json.loads(response_clean)
        logger.debug(f"Successfully parsed {response_name}")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {response_name} response: {e}")


def parse_prediction_response(response: str) -> Tuple[str, float]:
    """
    Parse a prediction response and return prediction and confidence.
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Tuple of (prediction, confidence)
        
    Raises:
        ValueError: If response is empty or invalid JSON
        KeyError: If required fields are missing
    """
    data = parse_json_response(response, "prediction")
    
    action = data.get('action', {})
    raw_prediction = action.get('response', '')
    # Handle both str and list (e.g. hallucination checker returns JSON array)
    if isinstance(raw_prediction, list):
        prediction = json.dumps(raw_prediction)  # Preserve as JSON string for downstream parsing
    else:
        prediction = str(raw_prediction).strip() if raw_prediction else ''
    confidence = float(data.get('confidence', 0.0))
    
    if not prediction:
        raise ValueError("Empty prediction in response")

    pred_preview = str(prediction)[:200] + "..." if len(str(prediction)) > 200 else str(prediction)
    logger.info(f"Parsed prediction: {pred_preview} (confidence: {confidence:.3f})")
    return prediction, confidence


def normalize_yes_no_answer(answer: str) -> str:
    """
    Normalize Yes/No answer variations to standard format.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer ("Yes" or "No")
        
    Raises:
        ValueError: If answer is not a recognized Yes/No variation
    """
    answer_upper = answer.upper().strip()
    if answer_upper in ['YES', 'Y']:
        return 'Yes'
    elif answer_upper in ['NO', 'N']:
        return 'No'
    else:
        raise ValueError(f"Invalid answer format: '{answer}'. Expected 'Yes' or 'No'")


def normalize_yes_no_answers(answers: str) -> str:
    """
    Normalize semicolon-delimited Yes/No answers.
    
    Args:
        answers: Semicolon-delimited answers string (e.g., "Yes; No; Yes")
        
    Returns:
        Normalized semicolon-delimited answers
        
    Raises:
        ValueError: If any answer is not a recognized Yes/No variation
    """
    answer_list = [ans.strip() for ans in answers.split(';')]
    normalized_answers = [normalize_yes_no_answer(answer) for answer in answer_list]
    return '; '.join(normalized_answers)


# --- Guardrails action parsing ---


def create_model_callable(model_api, model_id: str, provider: str, max_tokens: int = 1000, temperature: float = 0.7):
    """
    Create a callable function that wraps ModelAPI for Guardrails compatibility.
    
    Following the Guardrails custom LLM wrapper pattern from:
    https://www.guardrailsai.com/docs/how_to_guides/using_llms#custom-llm-wrappers
    
    Args:
        model_api: Your ModelAPI instance
        model_id: The model ID to use
        provider: The provider (openrouter)
        max_tokens: Maximum tokens for responses
        temperature: Temperature for generation
        
    Returns:
        Callable function that Guardrails can use
    """
    def model_callable(*, messages=None, **kwargs):
        """
        Custom LLM API wrapper following Guardrails pattern.
        
        Args:
            messages: List of message dictionaries (format: [{"role": "user", "content": "..."}])
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: The output of the LLM API
        """
        if messages is None:
            raise ValueError("messages parameter is required")
            
        # Extract parameters from kwargs or use defaults
        call_temperature = kwargs.get("temperature", temperature)
        call_max_tokens = kwargs.get("max_tokens", max_tokens)
        
        # Call your model API directly - it expects prompt parameter with messages list!
        try:
            response = model_api(
                model_id=model_id,
                prompt=messages,  # Our ModelAPI expects prompt parameter
                max_attempts=3,   # Guardrails handles retries
                provider=provider,
                temperature=call_temperature,
                max_tokens=call_max_tokens
            )
            
            # Model APIs return strings directly
            if isinstance(response, str):
                return response
            else:
                raise ValueError(f"Expected string response from {provider}, got {type(response)}")
        except Exception as e:
            logger.error(f"Model API call failed: {str(e)}")
            raise
    
    return model_callable




# Individual Pydantic models for each action type using discriminated unions
# Using Literal with enum values for discriminated union compatibility
class ProvideFinalResponseAction(BaseModel):
    action_type: Literal[ActionType.PROVIDE_FINAL_RESPONSE.value] = Field(..., description="The type of action")
    response: Union[str, List[str]] = Field(
        ...,
        description="The final response: a string, or a list of hallucinated citations/sentences"
    )

class ThinkAction(BaseModel):
    action_type: Literal[ActionType.THINK.value] = Field(..., description="The type of action")
    thought: str = Field(..., description="Your reasoning, analysis, or thought process about the current situation")

class OpenWebSearchAction(BaseModel):
    action_type: Literal[ActionType.OPEN_WEB_SEARCH.value] = Field(..., description="The type of action")
    query: str = Field(..., description="The search query to find relevant web information (must be non-empty, e.g. a citation or search phrase)")
    search_type: Optional[str] = Field(None, description="The type of search to perform. Options: 'web', 'news', 'google_scholar' (default: 'web')")
    num_results: Optional[int] = Field(None, description="Number of results to return (default: 10)")
    news_source: Optional[str] = Field(None, description="News source to use for news searches. Options: 'serpapi', 'mediastack' (default: 'serpapi')")
    cutoff_date: Optional[str] = Field(None, description="Cutoff date for search results (YYYY-MM-DD format). Only results published on or before this date will be returned. Optional.")
    exclude_undated: Optional[bool] = Field(None, description="If true, exclude results without publication dates (default: true)")

    @field_validator("query", mode="after")
    @classmethod
    def query_non_empty(cls, v: str) -> str:
        if not (v and str(v).strip()):
            raise ValueError("query must be a non-empty string (e.g. a citation or search phrase)")
        return str(v).strip()

class OpenCourtListenerSearchAction(BaseModel):
    action_type: Literal[ActionType.OPEN_COURTLISTENER_SEARCH.value] = Field(..., description="The type of action")
    query: str = Field(..., description="The search query to find relevant legal cases and documents (must be non-empty, e.g. a citation like '965 F.2d 962' or a case name)")
    search_type: Optional[str] = Field(None, description="The type of search to perform. Options: 'opinions', 'oral_arguments', 'people', 'dockets', 'recap', 'idb' (default: 'opinions')")
    court: Optional[str] = Field(None, description="The court to search (e.g., 'scotus', 'ca1', 'ca2') (default: 'scotus')")
    date_filter: Optional[Dict[str, Any]] = Field(None, description="Date filter with 'field' and 'before'/'after' keys (e.g., {'field': 'date_filed', 'before': '2022-01-01', 'after': '2021-01-01'}). Dates in YYYY-MM-DD format. Only 'date_filed' field is supported.")

    @field_validator("query", mode="after")
    @classmethod
    def query_non_empty(cls, v: str) -> str:
        if not (v and str(v).strip()):
            raise ValueError("query must be a non-empty string (e.g. a citation like '965 F.2d 962' or a case name)")
        return str(v).strip()

class CourtListenerCitationLookupAction(BaseModel):
    action_type: Literal[ActionType.COURTLISTENER_CITATION_LOOKUP.value] = Field(..., description="The type of action")
    cite: str = Field(..., description="The reporter citation to look up (e.g. '934 F.3d 53', '143 S. Ct. 1196')")

    @field_validator("cite", mode="after")
    @classmethod
    def cite_non_empty(cls, v: str) -> str:
        if not (v and str(v).strip()):
            raise ValueError("cite must be a non-empty string (e.g. '965 F.2d 962' or '143 S. Ct. 1196')")
        return str(v).strip()


class AccessCourtListenerOpinionAction(BaseModel):
    action_type: Literal[ActionType.ACCESS_COURTLISTENER_OPINION.value] = Field(..., description="The type of action")
    opinion_id: str = Field(..., description="The CourtListener opinion ID (obtained from OPEN_COURTLISTENER_SEARCH results)")

class SearchLocalOpinionAction(BaseModel):
    action_type: Literal[ActionType.SEARCH_LOCAL_OPINION.value] = Field(..., description="The type of action")
    opinion_id: str = Field(..., description="The CourtListener opinion ID (from a previous ACCESS_COURTLISTENER_OPINION call)")
    search_string: str = Field(..., description="The string to search for in the opinion text")

class ReadDocumentAction(BaseModel):
    action_type: Literal[ActionType.READ_DOCUMENT.value] = Field(..., description="The type of action")
    opinion_id: str = Field(..., description="The opinion to read: use opinion_<id> (e.g. opinion_9001448) or just the numeric id (e.g. 9001448) from a previous ACCESS_COURTLISTENER_OPINION.")
    start_line: int = Field(..., description="The starting line number to read from (0-indexed). Use 0 for the first line.")
    num_lines: int = Field(..., description="The number of lines to read from the document starting from start_line. Must be a positive integer.")

class EditScratchpadAction(BaseModel):
    action_type: Literal[ActionType.EDIT_SCRATCHPAD.value] = Field(..., description="The type of action")
    operation: str = Field(..., description="The operation to perform on the scratchpad. Options: 'append', 'insert', 'replace', 'clear'")
    content: str = Field(..., description="The content to add, insert, or replace in the scratchpad")
    position: Optional[int] = Field(None, description="The position for insert/replace operations (0-indexed). Required for 'insert' and 'replace' operations. Ignored for 'append' and 'clear' operations. Optional.")

# Create the discriminated union
ActionUnion = Union[
    ProvideFinalResponseAction,
    ThinkAction,
    OpenWebSearchAction,
    OpenCourtListenerSearchAction,
    CourtListenerCitationLookupAction,
    AccessCourtListenerOpinionAction,
    SearchLocalOpinionAction,
    ReadDocumentAction,
    EditScratchpadAction
]

# Main ActionChoice model with discriminated union
class ActionChoice(BaseModel):
    """
    Pydantic model that validates action_type and parameters using discriminated union.
    Following the Guardrails 0.2 pattern where the discriminated union is at the top level.
    """
    action: ActionUnion = Field(..., discriminator="action_type")


# Register validators for common LLM output issues
@register_validator(name="json-fence-extractor", data_type="string")
def json_fence_extractor(value, metadata: Dict) -> ValidationResult:
    """
    Extract JSON from code fences and return FailResult with fix_value if found.
    This enables reask with specific error message about code fence usage.
    """
    import re
    
    # Look for JSON in code fences
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, value, re.DOTALL)
    
    if matches:
        # Found JSON in fence, extract it
        json_content = matches[0]
        return FailResult(
            error_message="Please provide the JSON directly without code fences (```json```). I found JSON in a code block.",
            fix_value=json_content
        )
    
    return PassResult()


@register_validator(name="extra-text-cleaner", data_type="string")
def extra_text_cleaner(value, metadata: Dict) -> ValidationResult:
    """
    Clean extra text before/after JSON and return FailResult with fix_value if found.
    This enables reask with specific error message about extra text.
    """
    import re
    import json
    
    # Look for JSON objects in the text - find the first complete JSON object
    import json
    
    # Try to find JSON by looking for opening brace and matching closing brace
    start_idx = value.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(value)):
            if value[i] == '{':
                brace_count += 1
            elif value[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if brace_count == 0:  # Found complete JSON object
            json_content = value[start_idx:end_idx + 1]
            try:
                # Validate that it's actually valid JSON
                json.loads(json_content)
                
                # Check if there's extra text before or after
                if value.strip() != json_content.strip():
                    return FailResult(
                        error_message="Please provide only the JSON object without any extra text before or after it. I found extra text around the JSON.",
                        fix_value=json_content
                    )
            except json.JSONDecodeError:
                pass
    
    return PassResult()


@register_validator(name="malformed-json-fixer", data_type="string")
def malformed_json_fixer(value, metadata: Dict) -> ValidationResult:
    """
    Fix common JSON malformations like missing quotes around keys.
    This enables reask with specific error message about JSON formatting.
    """
    import re
    import json
    
    try:
        # Try to parse as-is first
        json.loads(value)
        return PassResult()
    except json.JSONDecodeError:
        # Try to fix common issues
        fixed_value = value
        
        # Fix missing quotes around keys (simple case)
        # Pattern: {key: "value"} -> {"key": "value"}
        fixed_value = re.sub(r'(\w+):', r'"\1":', fixed_value)
        
        try:
            json.loads(fixed_value)
            return FailResult(
                error_message="Please ensure all JSON keys are properly quoted. I found unquoted keys in your JSON.",
                fix_value=fixed_value
            )
        except json.JSONDecodeError:
            pass
    
    return PassResult()


def normalize_action_parameters_for_construction(
    action_type: str, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize parameters so they match Action class constructors.
    E.g. ProvideFinalResponse expects response: str; we accept list from the model and store as JSON array string.
    Strip num_results/k from OPEN_COURTLISTENER_SEARCH - the environment uses search_top_k from config.
    Reject empty query for search actions (so Guard reask and fallback both enforce non-empty).
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
    # Require non-empty query for search actions (Guard validators + fallback path)
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
    if action_type == ActionType.READ_DOCUMENT.value:
        if "opinion_id" not in params and "document_id" in params:
            params["opinion_id"] = params.pop("document_id")
    return params


def create_unified_action_guard(model_api, model_id: str, provider: str, 
                               max_tokens: int = 1000, temperature: float = 0.7, 
                               num_reasks: int = 1):
    """
    Create a unified Guard that combines formatting cleaning and semantic validation.
    This guard should be used for the original model call to avoid double model calls.
    
    Args:
        model_api: Your ModelAPI instance
        model_id: The model ID to use
        provider: The provider (openrouter)
        max_tokens: Maximum tokens for responses
        temperature: Temperature for generation
        num_reasks: Number of reask attempts
        
    Returns:
        Guard object that can be called with messages
    """
    # Create a callable function for your ModelAPI
    model_callable = create_model_callable(
        model_api=model_api,
        model_id=model_id,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Create Guard from Pydantic model
    unified_guard = Guard.for_pydantic(ActionChoice)
    
    # Create a wrapper that calls the unified guard and transforms the output
    class UnifiedActionGuard:
        def __init__(self, unified_guard, model_callable, num_reasks):
            self.unified_guard = unified_guard
            self.model_callable = model_callable
            self.num_reasks = num_reasks
        
        def __call__(self, messages, **kwargs):
            # Single model call with combined validation pipeline
            
            result = self.unified_guard(
                self.model_callable,
                messages=messages,
                num_reasks=self.num_reasks
            )
            
            # Transform the output from discriminated union format to parameters format
            if result.validated_output and 'action' in result.validated_output:
                action_data = result.validated_output['action']
                action_type = action_data.get('action_type')
                
                # Extract parameters (everything except action_type)
                parameters = {k: v for k, v in action_data.items() if k != 'action_type'}
                
                # Transform to parameters format for easy action initialization
                parameters_output = {
                    'action_type': action_type,
                    'parameters': parameters
                }
                
                # Update the result
                result.validated_output = parameters_output
            
            return result
    
    return UnifiedActionGuard(unified_guard, model_callable, num_reasks)


def parse_action_output_with_fallback(raw_output: str) -> Dict[str, Any]:
    """
    Fallback parser that uses simple JSON extraction when the unified guard fails.
    This provides an additional safety net for edge cases.
    """
    logger.warning("Using fallback parser - simple JSON extraction")
    
    try:
        # Extract JSON from text (handles code blocks, extra text, etc.)
        # First try to match JSON in code fences
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, raw_output, re.DOTALL)
        
        # If code fence found, use the extracted JSON
        if matches:
            pass  # matches already contains the extracted JSON strings
        
        # If no code fence matches, try to find complete JSON objects
        if not matches:
            # Look for complete JSON objects by finding matching braces
            start_idx = raw_output.find('{')
            while start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                
                for i in range(start_idx, len(raw_output)):
                    if raw_output[i] == '{':
                        brace_count += 1
                    elif raw_output[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                if brace_count == 0:  # Found complete JSON object
                    json_text = raw_output[start_idx:end_idx + 1]
                    try:
                        # Try to parse it as JSON
                        json.loads(json_text)
                        matches = [json_text]
                        break
                    except json.JSONDecodeError:
                        pass
                
                # Look for next opening brace
                start_idx = raw_output.find('{', start_idx + 1)
        
        for match in matches:
            json_text = match  # Now match is just the JSON string
            try:
                action_dict = json.loads(json_text)
                
                # Handle the new format: {"action": {"action_type": "...", "param": "..."}}
                if 'action' in action_dict:
                    action_data = action_dict['action']
                    action_type = action_data.get('action_type')
                    parameters = {k: v for k, v in action_data.items() if k != 'action_type'}
                    return {
                        'action_type': action_type,
                        'parameters': parameters
                    }
            except (json.JSONDecodeError, ValidationError):
                continue
        
        raise ValueError("No valid JSON found in output")
        
    except Exception as fallback_error:
        error_msg = f"Fallback parsing failed: {str(fallback_error)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
