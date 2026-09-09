"""Action types, the action registry, and every action the agent can take.

Action definitions stay independent of the environment and of the tool clients
that execute them; the environment dispatches actions.
"""

import warnings
from enum import Enum
from typing import Any, ClassVar


class ActionType(Enum):
    # Internal actions
    PROVIDE_FINAL_RESPONSE = "PROVIDE_FINAL_RESPONSE"
    THINK = "THINK"
    # Tool call actions
    # Search actions
    OPEN_WEB_SEARCH = "OPEN_WEB_SEARCH"
    OPEN_COURTLISTENER_SEARCH = "OPEN_COURTLISTENER_SEARCH"
    ACCESS_COURTLISTENER_OPINION = "ACCESS_COURTLISTENER_OPINION"
    COURTLISTENER_CITATION_LOOKUP = "COURTLISTENER_CITATION_LOOKUP"
    SEARCH_LOCAL_OPINION = "SEARCH_LOCAL_OPINION"
    # Filesystem actions
    READ_DOCUMENT = "READ_DOCUMENT"
    EDIT_SCRATCHPAD = "EDIT_SCRATCHPAD"


class Action:
    """
    A base class for the possible actions available to the agent. Subclass this and implement the
    `forward` method and the following class attributes:

    - **action_type** (`ActionType`) -- The type of the action. This should be the ActionType enum.
    - **description** (`str`) -- A short description of what your action does, the inputs it expects and the output(s) it
      will return. For instance 'This is a action that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
      It has a `type` key, `description` key, `required` key, and an optional `default` key
      applied when an optional input is omitted.
      This can be used in the generated description for your action.

    Inspired by the `Tool` class from the  `smol-agents` library: https://github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py#L106
    """

    action_type: ActionType
    description: str
    inputs: ClassVar[dict[str, dict[str, str | type | bool]]]

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register Action subclasses in the registry when they are defined.
        This enables automatic discovery without manual imports.
        """
        super().__init_subclass__(**kwargs)

        # Only register if the class has an action_type (not the base Action class)
        if hasattr(cls, "action_type") and cls.action_type and cls != Action:
            _action_registry[cls.action_type] = cls

    def __init__(self, **kwargs):
        """
        Initialize the Action with validation of parameters against the inputs specification.

        Args:
            **kwargs: Keyword arguments that should match the inputs specification

        Raises:
            ValueError: If parameters don't match the inputs specification or if required inputs are missing
            TypeError: If parameter types don't match the expected input types
        """
        # Get the inputs specification from the class
        if not hasattr(self, "inputs"):
            raise ValueError(
                f"Action class {self.__class__.__name__} must define an 'inputs' class attribute"
            )

        inputs_spec = self.inputs

        # Validate that all provided parameters are in the inputs specification
        for param_name in kwargs:
            if param_name not in inputs_spec:
                raise ValueError(
                    f"Parameter '{param_name}' is not defined in the inputs specification. "
                    f"Available inputs: {list(inputs_spec.keys())}"
                )

        # Validate that all required inputs are provided
        for input_name, input_spec in inputs_spec.items():
            # Check if the input is optional
            is_required = input_spec.get(
                "required", True
            )  # Default to required if not specified
            if is_required and input_name not in kwargs:
                raise ValueError(
                    f"Required input '{input_name}' is missing. "
                    f"Available inputs: {list(inputs_spec.keys())}"
                )

        # Validate parameter types against input types
        for input_name, input_spec in inputs_spec.items():
            if input_name in kwargs:
                param_value = kwargs[input_name]
                expected_type = input_spec.get("type")

                # Skip validation for None values (optional parameters)
                if param_value is not None and expected_type is not None:
                    self._validate_parameter_type(
                        param_value, expected_type, input_name
                    )

        # Store every input as an attribute; omitted optional inputs take the spec default.
        for input_name, input_spec in inputs_spec.items():
            setattr(self, input_name, kwargs.get(input_name, input_spec.get("default")))

    def _validate_parameter_type(
        self, value: Any, expected_type: str, param_name: str
    ) -> None:
        """
        Validate that a parameter value matches the expected type.

        Args:
            value: The parameter value to validate
            expected_type: The expected type as a string
            param_name: The name of the parameter for error reporting

        Raises:
            TypeError: If the value doesn't match the expected type
        """
        # Type validation mapping
        type_validators = {
            "string": str,
            "boolean": bool,
            "integer": int,
            "float": float,
            "list": list,
            "dict": dict,
            "object": dict,  # object maps to dict in Python
            "any": None,  # Accept any type
            "null": type(None),  # Only None
        }

        if expected_type not in type_validators:
            warnings.warn(
                f"Unknown input type '{expected_type}' for parameter '{param_name}'. Skipping type validation.",
                stacklevel=1,
            )
            return

        expected_python_type = type_validators[expected_type]

        # Skip validation for "any" type
        if expected_python_type is None:
            return

        # Validate the type
        if not isinstance(value, expected_python_type):
            raise TypeError(
                f"Parameter '{param_name}' must be a {expected_type}, got {type(value).__name__}"
            )

    def get_input_parameters(self) -> dict[str, Any]:
        parameters = {}
        for input_name in self.inputs:
            value = getattr(self, input_name)
            parameters[input_name] = value
        return parameters

    def __repr__(self):
        """
        Return a string representation of the Action instance.
        """
        return f"<{self.__class__.__name__} action_type={self.action_type.value}, description={self.description}, inputs={self.inputs!s}>"


# Registry for Action subclasses
_action_registry: dict[ActionType, type["Action"]] = {}


def get_action_class(action_type: ActionType) -> type[Action]:
    """
    Get the Action class for a given action type.

    Args:
        action_type: The ActionType enum value

    Returns:
        The corresponding Action class

    Raises:
        KeyError: If the action_type is not found in the mapping
    """
    if action_type not in _action_registry:
        raise KeyError(
            f"ActionType '{action_type.value}' not found. Available types: {list(_action_registry.keys())}"
        )
    return _action_registry[action_type]


def get_all_action_classes() -> list[type[Action]]:
    return list(_action_registry.values())


# Search type mappings based on CourtListener API documentation. Declared here
# because OpenCourtListenerSearch documents them; courtlistener.py imports this.
SEARCH_TYPES = {
    "opinions": "o",  # Case law opinion clusters with nested Opinion documents
    "cases": "r",  # List of Federal cases (dockets) with up to three nested documents
    "dockets": "d",  # Federal cases (dockets) from PACER
    "filings": "rd",  # Federal filing documents from PACER
    "judges": "p",  # Judges
    "oral_arguments": "oa",  # Oral argument audio files
}


# --- Internal actions: agent reasoning and final response ---


class ProvideFinalResponse(Action):
    """
    Internal action for providing the final response to a given task example.

    This action is used when the agent has completed its exploration and
    wants to provide a final response.
    """

    action_type = ActionType.PROVIDE_FINAL_RESPONSE
    description = "Provide the final response to the given task example. Use this when you have completed your exploration and want to provide a final response."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "response": {
            "type": "string",
            "description": "The final response to the given task example",
            "required": True,
        }
    }


class Think(Action):
    """
    Internal action for agent reasoning and thought process.

    This action allows the agent to think,
    showing its reasoning process before taking other actions.
    """

    action_type = ActionType.THINK
    description = "Thinking. Use this to reason about your history and current state, plan your next steps, or analyze information before taking action. Use only briefly to synthesize or transition between actions. Do not use repeatedly for planning; proceed to search actions instead."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "thought": {
            "type": "string",
            "description": "Your reasoning, analysis, or thought process about the current situation",
            "required": True,
        }
    }


# --- CourtListener search actions ---


class AccessCourtListenerOpinion(Action):
    """
    Access a single CourtListener opinion by ID.

    Use this after OPEN_COURTLISTENER_SEARCH to fetch the full text and metadata
    of a specific opinion (e.g., to verify a citation or quotation).
    The full opinion is stored locally; the agent sees a short snippet. Use
    SEARCH_LOCAL_OPINION to search within a previously fetched opinion.
    """

    action_type = ActionType.ACCESS_COURTLISTENER_OPINION
    description = "Fetch the full opinion from CourtListener by opinion ID. Use after searching to retrieve the complete opinion text for citation verification. The full opinion is stored locally; use SEARCH_LOCAL_OPINION to search within it."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "opinion_id": {
            "type": "string",
            "description": "The CourtListener opinion ID (obtained from OPEN_COURTLISTENER_SEARCH results)",
            "required": True,
        }
    }


class SearchLocalOpinion(Action):
    """
    Search for a string within a previously fetched opinion (by opinion_id).

    Use after ACCESS_COURTLISTENER_OPINION to search within that opinion's full text.
    Returns a snippet around the match, or None if not found.
    """

    action_type = ActionType.SEARCH_LOCAL_OPINION
    description = "Search for a string within an opinion already fetched with ACCESS_COURTLISTENER_OPINION. Returns a snippet around the match, or None if not found."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "opinion_id": {
            "type": "string",
            "description": "The CourtListener opinion ID (from a previous ACCESS_COURTLISTENER_OPINION call)",
            "required": True,
        },
        "search_string": {
            "type": "string",
            "description": "The string to search for in the opinion text (e.g., a quoted phrase or citation)",
            "required": True,
        },
    }


class CourtListenerCitationLookup(Action):
    """
    Look up a reporter citation on CourtListener (e.g. '934 F.3d 53', '143 S. Ct. 1196').
    Returns matching opinion(s) if found.
    """

    action_type = ActionType.COURTLISTENER_CITATION_LOOKUP
    description = "Look up a legal citation (e.g. '934 F.3d 53', '143 S. Ct. 1196') on CourtListener. Returns matching opinion(s) if found."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "cite": {
            "type": "string",
            "description": "The reporter citation to look up (e.g. '934 F.3d 53', '143 S. Ct. 1196')",
            "required": True,
        }
    }


class OpenCourtListenerSearch(Action):
    """
    Search action for CourtListener database.

    This action allows the agent to search the CourtListener database
    for legal cases, opinions, and other legal documents.
    """

    action_type = ActionType.OPEN_COURTLISTENER_SEARCH
    description = "Search the CourtListener database for legal cases, opinions, and documents. Use this to find relevant legal information and precedents."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant legal cases and documents",
            "required": True,
        },
        "search_type": {
            "type": "string",
            "description": "The type of search to perform. Options: "
            + ", ".join([f"'{k}'" for k in SEARCH_TYPES])
            + " (default: 'opinions')",
            "required": False,
            "default": "opinions",
        },
        "court": {
            "type": "string",
            "description": "The court to search (e.g., 'scotus', 'ca1', 'ca2') (default: 'scotus')",
            "required": False,
            "default": "scotus",
        },
        "date_filter": {
            "type": "object",
            "description": "Date filter with 'field' and 'before'/'after' keys (e.g., {'field': 'date_filed', 'before': '2022-01-01', 'after': '2021-01-01'}). Dates in YYYY-MM-DD format. Only 'date_filed' field is supported.",
            "required": False,
        },
    }


# --- Open web search actions ---


class OpenWebSearch(Action):
    """
    Search action for web search operations.

    This action allows the agent to perform web searches to find relevant information
    from the internet using various search engines and news sources.
    """

    action_type = ActionType.OPEN_WEB_SEARCH
    description = "Perform Google search on the open internet (web, news, and Google Scholar). Use this to find current information, news, scholarly articles, and web content that may not be in the closed document index."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant web information",
            "required": True,
        },
        "search_type": {
            "type": "string",
            "description": "The type of search to perform. Options: 'web', 'news', 'google_scholar' (default: 'web')",
            "required": False,
            "default": "web",
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (default: 10)",
            "required": False,
            "default": 10,
        },
        "news_source": {
            "type": "string",
            "description": "News source to use for news searches. Options: 'serpapi', 'mediastack' (default: 'serpapi')",
            "required": False,
            "default": "serpapi",
        },
        "cutoff_date": {
            "type": "string",
            "description": "Cutoff date for search results (YYYY-MM-DD format). Only results published on or before this date will be returned. Optional.",
            "required": False,
        },
        "exclude_undated": {
            "type": "boolean",
            "description": "If true, exclude results without publication dates (default: true)",
            "required": False,
            "default": True,
        },
    }

    def __repr__(self):
        return f"<OpenWebSearch query='{self.query}' search_type='{self.search_type}' num_results={self.num_results}>"


# --- Filesystem actions: document reading and scratchpad editing ---


class ReadDocument(Action):
    """
    Action for reading a portion of a document.

    This action allows the agent to read specific portions of documents
    that have been retrieved from search results or other sources.
    """

    action_type = ActionType.READ_DOCUMENT
    description = "Read a portion of a document from search results or available documents. Use this to examine specific parts of legal cases, opinions, or other documents."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "opinion_id": {
            "type": "string",
            "description": "The opinion to read: use opinion_<id> (e.g. opinion_9001448) or just the numeric id (e.g. 9001448) from a previous ACCESS_COURTLISTENER_OPINION.",
            "required": True,
        },
        "start_line": {
            "type": "integer",
            "description": "The starting line number to read from (0-indexed). Use 0 for the first line.",
            "required": True,
        },
        "num_lines": {
            "type": "integer",
            "description": "The number of lines to read from the document starting from start_line. Must be a positive integer.",
            "required": True,
        },
    }


class EditScratchpad(Action):
    """
    Action for editing the agent's scratchpad.

    This action allows the agent to take notes, organize thoughts,
    and maintain a working memory during reasoning.
    """

    action_type = ActionType.EDIT_SCRATCHPAD
    description = "Edit the agent's scratchpad to take notes, organize thoughts, or maintain working memory. Use this to keep track of important information during reasoning."
    inputs: ClassVar[dict[str, dict[str, Any]]] = {
        "operation": {
            "type": "string",
            "description": "The operation to perform: 'append' (add to end), 'insert' (insert at position), 'replace' (replace at position), or 'clear' (clear all content)",
            "required": True,
        },
        "content": {
            "type": "string",
            "description": "The content to add, insert, or replace in the scratchpad",
            "required": True,
        },
        "position": {
            "type": "integer",
            "description": "The position for insert/replace operations (0-indexed). Required for 'insert' and 'replace' operations. Ignored for 'append' and 'clear' operations. Optional.",
            "required": False,
        },
    }
