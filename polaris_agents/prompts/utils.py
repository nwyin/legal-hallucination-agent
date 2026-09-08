"""
Utility functions for prompt generation.

These are shared utilities used across different prompt constructors.
Migrated from polaris_agents/agents/prompts.py for the new prompts module.
"""

import logging
from typing import List, Dict, Any

from ..agents.action import ActionType, get_action_class

logger = logging.getLogger(__name__)


def get_canonical_theta_definition() -> str:
    """
    Get the canonical definition of task parameters (θ).
    
    This is the shared definition used consistently across all prompts.
    
    Returns:
        Canonical definition string for θ
    """
    return (
        "Instance-specific information required to make an accurate prediction for this particular task instance — "
        "the facts, evidence, and signals that determine the correct answer."
    )


def get_canonical_design_definition() -> str:
    """
    Get the canonical definition of design parameters (D).
    
    This is the shared definition used consistently across all prompts.
    
    Returns:
        Canonical definition string for D
    """
    return (
        "Meta-level methodology: how experts make predictions in this domain — the factors they consider, "
        "how they model key decision-makers (if any are relevant to the outcome), and how they weigh and interpret signals. "
        "Domain-level understanding: key concepts, frameworks, theories, and doctrines relevant to this problem class; "
        "background knowledge that provides context; and problem-solving strategies/techniques that generalize across similar problems. "
        "Information-gathering strategies: effective tool use, query formulation, what search terms/approaches work, "
        "how to refine failed queries, and what sources are most valuable for this type of problem."
    )


def build_action_guidelines(action_space: List[ActionType]) -> str:
    """
    Build action guidelines based on available actions.
    
    Includes action descriptions and additional guidance for specific action types
    (e.g., search actions) when they are present.
    
    Args:
        action_space: List of available action types
        
    Returns:
        Formatted string with action guidelines
    """
    guidelines = ["## Action Guidelines"]
    
    for action_type in action_space:
        try:
            action_class = get_action_class(action_type)
            description = getattr(action_class, 'description', '')
            
            # Warn if description is missing - this will cause suboptimal agent behavior
            if not description:
                logger.warning(
                    f"Action class {action_class.__name__} (action_type={action_type.value}) "
                    f"does not have a 'description' attribute defined. This will cause suboptimal "
                    f"agent behavior as the agent won't understand what this action does."
                )
                description = f"[WARNING: No description defined for {action_type.value}]"
            
            # Warn if inputs are missing - this will also cause issues
            if not hasattr(action_class, 'inputs') or not action_class.inputs:
                logger.warning(
                    f"Action class {action_class.__name__} (action_type={action_type.value}) "
                    f"does not have 'inputs' defined. This will cause suboptimal agent behavior "
                    f"as the agent won't know what parameters this action requires."
                )
        except KeyError:
            logger.warning(
                f"Could not find action class for action_type={action_type.value}. "
                f"This will cause suboptimal agent behavior."
            )
            description = f"[WARNING: Action class not found for {action_type.value}]"
        
        guidelines.append(f"- **{action_type.name}**: {description}")
    
    # Add detailed guidance for search actions if present
    if ActionType.OPEN_WEB_SEARCH in action_space:
        guidelines.append("")
        guidelines.append("**About OPEN_WEB_SEARCH:**")
        guidelines.append("- OPEN_WEB_SEARCH performs Google search on the open internet (web, news, and Google Scholar). Use this to find current information, news, scholarly articles, and web content.")
    if ActionType.COURTLISTENER_CITATION_LOOKUP in action_space:
        guidelines.append("")
        guidelines.append("**About COURTLISTENER_CITATION_LOOKUP:**")
        guidelines.append("- For reporter-style citations (e.g. '965 F.2d 962', '143 S. Ct. 1196'), use **COURTLISTENER_CITATION_LOOKUP** first with the `cite` parameter. Use OPEN_COURTLISTENER_SEARCH only if citation lookup fails or you have only a case name.")
    
    return "\n".join(guidelines)


def create_selection_actions_description(action_space: List[ActionType]) -> str:
    """
    Create a description of all available actions with type, description, and input parameters.
    
    Args:
        action_space: List of available action types
        
    Returns:
        Formatted string describing available actions
    """
    actions_desc = "You must select an action from the following list and provide the parameters for the selected action:\n"
    
    for i, action_type in enumerate(action_space):
        action_class = get_action_class(action_type)
        
        if action_class:
            actions_desc += f"{i}. action: {action_class.action_type.value} - {action_class.description}\n"
            
            if action_class.inputs:
                required_params = []
                optional_params = []
                
                for param_name, param_spec in action_class.inputs.items():
                    param_type = param_spec.get('type', 'string')
                    param_description = param_spec.get('description', '')
                    is_required = param_spec.get('required', True)
                    
                    param_info = f"{param_name} ({param_type}): {param_description}"
                    
                    if is_required:
                        required_params.append(param_info)
                    else:
                        optional_params.append(param_info)
                
                if required_params:
                    actions_desc += "   required parameters:\n"
                    for param_info in required_params:
                        actions_desc += f"   - {param_info}\n"
                
                if optional_params:
                    actions_desc += "   optional parameters:\n"
                    for param_info in optional_params:
                        actions_desc += f"   - {param_info}\n"
            actions_desc += "\n"
    
    return actions_desc


def create_actions_parameters_description(action_space: List[ActionType]) -> str:
    """
    Create a description of available actions listing type, description,
    and input parameters (required and optional), without instruction text.
    
    Args:
        action_space: List of available action types
        
    Returns:
        Formatted string describing actions and parameters
    """
    actions_desc = ""
    for i, action_type in enumerate(action_space):
        action_class = get_action_class(action_type)
        if not action_class:
            continue
        actions_desc += f"{i}. action: {action_class.action_type.value} - {action_class.description}\n"
        if action_class.inputs:
            required_params = []
            optional_params = []
            for param_name, param_spec in action_class.inputs.items():
                param_type = param_spec.get('type', 'string')
                param_description = param_spec.get('description', '')
                is_required = param_spec.get('required', True)
                param_info = f"{param_name} ({param_type}): {param_description}"
                if is_required:
                    required_params.append(param_info)
                else:
                    optional_params.append(param_info)
            if required_params:
                actions_desc += "   required parameters:\n"
                for param_info in required_params:
                    actions_desc += f"   - {param_info}\n"
            if optional_params:
                actions_desc += "   optional parameters:\n"
                for param_info in optional_params:
                    actions_desc += f"   - {param_info}\n"
        actions_desc += "\n"
    return actions_desc


def format_action_history(history: List[Dict], max_actions: int = -1) -> str:
    """
    Format action history into a readable string.
    
    Args:
        history: List of action-observation history entries
        
    Returns:
        Formatted string of recent action history
    """
    if not history:
        return ""
    
    history_to_render = history
    if max_actions is not None and max_actions > 0:
        history_to_render = history[-max_actions:]

    history_text = "RECENT ACTIONS:\n"
    for i, step in enumerate(history_to_render):
        action = step['action']
        obs = step['observation']
        
        action_type = action.action_type.value
        parameters = action.get_input_parameters()
        
        history_text += f"Step {i+1}: {action_type}\n"
        history_text += f"Parameters: {parameters}\n"
        
        result_text = format_observation_result(obs.result)
        history_text += f"Result: {result_text}\n\n"
    
    return history_text


def format_observation_result(result: Any) -> str:
    """
    Format observation result into a readable string.
    
    Args:
        result: The observation result (string, dict, or other)
        
    Returns:
        Formatted string representation
    """
    if isinstance(result, str):
        return result[:200] + "..." if len(result) > 200 else result
    elif isinstance(result, dict):
        if 'count' in result:
            result_text = f"Found {result['count']} results"
            if 'results' in result and result['results']:
                if 'snippet' in result['results'][0]:
                    result_text += f" (showing snippets of first {result.get('snippets_shown', len(result['results']))} results)"
                    for i, result_item in enumerate(result['results'][:3]):
                        case_name = result_item.get('case_name', f'Case {i}')
                        snippet = result_item.get('snippet', '')[:100] + "..." if len(result_item.get('snippet', '')) > 100 else result_item.get('snippet', '')
                        result_text += f"\n- {case_name}: {snippet}"
                else:
                    for i, result_item in enumerate(result['results'][:3]):
                        case_name = result_item.get('case_name', f'Case {i}')
                        result_text += f"\n- {case_name}"
            return result_text
        elif 'error' in result:
            return f"Error: {result['error']}"
        else:
            return str(result)[:200] + "..."
    else:
        return str(result)[:200] + "..."


def create_action_selection_json_format() -> str:
    """
    Create JSON format instructions for action selection prompts.
    Includes reasoning field to encourage explicit reasoning.
    
    Returns:
        Formatted string with action selection JSON format
    """
    return """## Response Format
Provide your action selection as a JSON object:

```json
{
  "action": {
    "action_type": "ACTION_TYPE",
    "required_param1": "value1",
    "optional_param": "value"
  },
  "reasoning": "<brief explanation for why this action optimizes the objective>"
}
```"""
