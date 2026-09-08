import warnings

from enum import Enum
from typing import Dict, Union, Any, Type, List

class ActionType(Enum):
    # Internal actions
    PROVIDE_FINAL_RESPONSE = "PROVIDE_FINAL_RESPONSE"
    THINK = "THINK"
    # Bandit actions
    PULL_ARM = "PULL_ARM"
    # Tool call actions
    # Search actions
    CLOSED_SEARCH = "CLOSED_SEARCH"
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
      It has a `type` key, `description` key, and `required` key.
      This can be used in the generated description for your action.

    Inspired by the `Tool` class from the  `smol-agents` library: https://github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py#L106
    """

    action_type: ActionType
    description: str
    inputs: Dict[str, Dict[str, Union[str, type, bool]]]

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register Action subclasses in the registry when they are defined.
        This enables automatic discovery without manual imports.
        """
        super().__init_subclass__(**kwargs)
        
        # Only register if the class has an action_type (not the base Action class)
        if hasattr(cls, 'action_type') and cls.action_type and cls != Action:
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
        if not hasattr(self, 'inputs'):
            raise ValueError(f"Action class {self.__class__.__name__} must define an 'inputs' class attribute")
        
        inputs_spec = self.inputs
        
        # Validate that all provided parameters are in the inputs specification
        for param_name in kwargs:
            if param_name not in inputs_spec:
                raise ValueError(f"Parameter '{param_name}' is not defined in the inputs specification. "
                              f"Available inputs: {list(inputs_spec.keys())}")
        
        # Validate that all required inputs are provided
        for input_name, input_spec in inputs_spec.items():
            # Check if the input is optional
            is_required = input_spec.get('required', True)  # Default to required if not specified
            if is_required and input_name not in kwargs:
                raise ValueError(f"Required input '{input_name}' is missing. "
                              f"Available inputs: {list(inputs_spec.keys())}")
        
        # Validate parameter types against input types
        for input_name, input_spec in inputs_spec.items():
            if input_name in kwargs:
                param_value = kwargs[input_name]
                expected_type = input_spec.get('type')
                
                # Skip validation for None values (optional parameters)
                if param_value is not None and expected_type is not None:
                    self._validate_parameter_type(param_value, expected_type, input_name)

        # Store the validated parameters as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _validate_parameter_type(self, value: Any, expected_type: str, param_name: str) -> None:
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
            "null": type(None)  # Only None
        }
        
        if expected_type not in type_validators:
            warnings.warn(f"Unknown input type '{expected_type}' for parameter '{param_name}'. Skipping type validation.")
            return
            
        expected_python_type = type_validators[expected_type]
        
        # Skip validation for "any" type
        if expected_python_type is None:
            return
            
        # Validate the type
        if not isinstance(value, expected_python_type):
            raise TypeError(f"Parameter '{param_name}' must be a {expected_type}, got {type(value).__name__}")


    def get_input_parameters(self) -> Dict[str, Any]:
        parameters = {}
        for input_name in self.inputs.keys():
            value = getattr(self, input_name)
            parameters[input_name] = value
        return parameters


    def print_input_parameters(self):
        print(f"\n=== {self.__class__.__name__} Attributes ===")
        
        # Get all action input parameters
        for input_name in self.inputs.keys():
            value = getattr(self, input_name)
            print(f"{input_name}: {value}")


    def __repr__(self):
        """
        Return a string representation of the Action instance.
        """
        return f"<{self.__class__.__name__} action_type={self.action_type.value}, description={self.description}, inputs={str(self.inputs)}>"


# Registry for Action subclasses
_action_registry: Dict[ActionType, Type['Action']] = {}


def register_action_class(action_class: Type['Action']) -> None:
    """
    Manually register an Action class in the registry.
    Useful for registering classes that might not be imported at module load time.
    
    Args:
        action_class: The Action class to register
    """
    if not issubclass(action_class, Action):
        raise TypeError(f"Can only register Action subclasses, got {type(action_class).__name__}")
    
    if not hasattr(action_class, 'action_type') or not action_class.action_type:
        raise ValueError(f"Action class {action_class.__name__} must have an action_type attribute")
    
    _action_registry[action_class.action_type] = action_class


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
        raise KeyError(f"ActionType '{action_type.value}' not found. Available types: {list(_action_registry.keys())}")
    return _action_registry[action_type]


def get_all_action_classes() -> List[Type[Action]]:
    return list(_action_registry.values())
