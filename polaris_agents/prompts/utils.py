"""
Utility functions for prompt generation.

These are shared utilities used across different prompt constructors.
Migrated from polaris_agents/agents/prompts.py for the new prompts module.
"""

import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING

from ..agents.action import ActionType, get_action_class
from ..environments.base import Observation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base import DomainKnowledgeProvider


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


def get_canonical_belief_framework_text() -> str:
    """
    Get the canonical belief framework description combining D and θ.
    
    This is the shared text used consistently across all prompts that describe
    the belief framework.
    
    Returns:
        Complete belief framework description
    """
    theta_def = get_canonical_theta_definition()
    design_def = get_canonical_design_definition()
    
    return f"""You maintain Bayesian beliefs over D and θ, described in natural language:

- **Design beliefs p(D)**: Your current uncertainty about the design parameters D
- **Task beliefs p(θ)**: Your current uncertainty about the task parameters θ

These beliefs are updated after each observation. The distinction between D and θ is not always sharp - some observations inform both beliefs."""


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
    if ActionType.CLOSED_SEARCH in action_space and ActionType.OPEN_WEB_SEARCH in action_space:
        guidelines.append("")
        guidelines.append("**Choosing between CLOSED_SEARCH and OPEN_WEB_SEARCH:**")
        guidelines.append("- Use **CLOSED_SEARCH** to search the fixed set of indexed documents. If you get no results or poor results, the information may not exist in the index, or you may need to rephrase your query.")
        guidelines.append("- Use **OPEN_WEB_SEARCH** to search the open internet via Google (web, news, and Google Scholar). Use this for current information, news, scholarly articles, or content that may not be in the closed document index.")
    elif ActionType.CLOSED_SEARCH in action_space:
        guidelines.append("")
        guidelines.append("**About CLOSED_SEARCH:**")
        guidelines.append("- CLOSED_SEARCH searches a fixed set of indexed documents (closed world). If you get no results or poor results, the information may not exist in the indexed documents, or your query may need to be rephrased with different terminology or more specific details.")
    elif ActionType.OPEN_WEB_SEARCH in action_space:
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


def create_explicit_eig_json_format() -> str:
    """
    Create JSON format instructions for explicit EIG estimation.
    Asks for numerical EIG estimates (in bits) for candidate actions.
    
    Based on approach from Arumugam & Griffiths (2025) "Toward Efficient Exploration 
    by Large Language Model Agents" https://arxiv.org/pdf/2504.20997
    
    Returns:
        Formatted string with explicit EIG estimation JSON format
    """
    return """## Response Format
You are acting as a **conservative information gain estimator** to help select the next action.

For each candidate action, estimate the Expected Information Gain (EIG) **in bits**.  
EIG measures how much the observation from that action will reduce your uncertainty about D and θ.

Your estimates should be **conservative**: it is fine to underestimate, but you should avoid overestimating the true information gain.

Keep in mind:
- Suboptimal or incorrect actions can still be informative – you can gain information about the optimal prediction without selecting it
- Once the optimal answer is effectively known under your beliefs, information gain should be (approximately) 0 for all actions

Whenever possible, show brief approximate calculations or concrete numbers in your justification.

Provide your response as a JSON object:

```json
{
  "candidate_actions": [
    {
      "action": {
        "action_type": "ACTION_TYPE_1",
        "param": "value"
      },
      "eig_estimate_bits": <float>  // conservative estimate in bits (rounded to ≤ 3 decimal places)
    },
    {
      "action": {
        "action_type": "ACTION_TYPE_2",
        "param": "value"
      },
      "eig_estimate_bits": <float>  // conservative estimate in bits (rounded to ≤ 3 decimal places)
    }
  ],
  "selected_action": {
    "action_type": "ACTION_TYPE",
    "param": "value"
  },
  "reasoning": "<brief explanation for why this selected action is preferred under the EIG objective>"
}
```"""


def create_prediction_json_format() -> str:
    """
    Create JSON format instructions for prediction prompts.
    Includes action, confidence, and reasoning fields.
    
    This is a generic format - the specific content requirements for the `response` field
    should be specified in the task-specific response requirements.
    
    Returns:
        Formatted string with prediction JSON format
    """
    return """## Response Format
Provide your prediction as a JSON object:

```json
{
    "action": {
        "action_type": "PROVIDE_FINAL_RESPONSE",
        "response": "<your prediction - see Response Requirements section for exact format>"
    },
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your prediction>"
}
```

**Important:** 
- The `response` field format is specified in the Response Requirements section above
- Put ALL explanations, reasoning, and narrative text in the `reasoning` field, NOT in the `response` field
- The `response` field should contain ONLY the prediction in the exact format specified in Response Requirements"""


def create_belief_update_json_format() -> str:
    """
    Create JSON format instructions for belief update prompts.
    
    Returns:
        Formatted string with belief update JSON format
    """
    return """## Response Format
Provide your updated beliefs as a JSON object:

```json
{
    "task_beliefs": "<your rich, cumulative understanding of this task instance>",
    "design_beliefs": "<your rich, cumulative understanding of how the domain works - decision-making patterns, influential factors>"
}
```

Write each belief as a **long, detailed natural language paragraph** (or multiple paragraphs). Be thorough and information-dense - include all relevant details, evidence, hypotheses, and reasoning. The text can be several paragraphs long.

**Important**: The values must be text strings (natural language), NOT nested JSON."""


def build_prediction_prompt(
    env_description: str,
    task_beliefs: str,
    action_space: List[ActionType],
    design_beliefs: Optional[str] = None,
    response_requirements: Optional[str] = None,
    history: Optional[List[Dict]] = None,
) -> str:
    """
    Build a prediction prompt that enforces PROVIDE_FINAL_RESPONSE.
    
    This is a shared utility used by all agents for consistent prediction prompts.
    
    Args:
        env_description: Environment description
        task_beliefs: Current task beliefs
        action_space: List of available action types
        design_beliefs: Optional design beliefs (for agents that have them)
        response_requirements: Optional response format requirements from environment
        history: Optional action history (for agents that want to show it)
        
    Returns:
        Complete prediction prompt string
    """
    # Use functions from this same module (no import needed)
    prompt = f"""## Environment
{env_description}

## Current State"""
    
    if task_beliefs:
        prompt += f"""

### Current Task Beliefs:
{task_beliefs}"""
    
    if design_beliefs:
        prompt += f"""

### Current Design Beliefs:
{design_beliefs}"""
    
    if history:
        prompt += f"""

### Action History:
{format_action_history(history)}"""
    
    prompt += f"""

## Available Actions
{create_actions_parameters_description(action_space)}

## Task
Based on your current beliefs and observations, choose the PROVIDE_FINAL_RESPONSE action with your best prediction.

## Response Format
Provide your prediction as a JSON object with the following structure:

RESPONSE FORMAT (JSON only)
{{
    "action": {{
        "action_type": "PROVIDE_FINAL_RESPONSE",
        "response": "<your prediction - see Response Requirements section for exact format>"
    }},
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation - put ALL narrative text here, NOT in response field>"
}}

**Important:** The `response` field format is specified in the Response Requirements section. Put ALL explanations in the `reasoning` field."""
    
    if response_requirements:
        prompt += f"\n\n## Response Requirements:\n{response_requirements}"
    
    return prompt


def build_eig_estimate_prompt(
    env_description: str,
    task_beliefs: str,
    action_space: List[ActionType],
    action_type: str,
    action_params: Dict[str, Any],
    design_beliefs: Optional[str] = None,
    history: Optional[List[Dict]] = None,
    domain_knowledge: Optional[Any] = None,
) -> str:
    """
    Build an EIG estimation prompt that is agent-agnostic.
    
    This is a shared utility used by all agents for consistent EIG estimation prompts.
    
    Args:
        env_description: Environment description
        task_beliefs: Current task beliefs
        action_space: List of available action types (unused, kept for compatibility)
        action_type: Type of action being evaluated
        action_params: Parameters of the action being evaluated
        design_beliefs: Optional design beliefs (for agents that have them)
        history: Optional action history (for agents that want to show it)
        domain_knowledge: Optional domain knowledge provider for task-specific θ/D definitions
        
    Returns:
        Complete EIG estimation prompt string
    """
    # Build belief framework section explaining IDS-OED abstraction
    theta_def = get_canonical_theta_definition()
    design_def = get_canonical_design_definition()
    
    # Build the information space section (parameters D and θ) - D first to encourage early domain exploration
    info_space_section = f"""For a given task instance, we split the information space into two components:

- **Design parameters (D)**: {design_def}
- **Task parameters (θ)**: {theta_def}"""
    
    # Add domain-specific definitions if available
    if domain_knowledge:
        if hasattr(domain_knowledge, 'get_theta_description') and hasattr(domain_knowledge, 'get_design_description'):
            theta_desc = domain_knowledge.get_theta_description()
            design_desc = domain_knowledge.get_design_description()
            info_space_section += f"""

### Domain-Specific Definitions

For this task, we provide a more specific mapping of information to D and θ:

{design_desc}

{theta_desc}"""
    
    # Build the beliefs section (p(D) and p(θ))
    beliefs_section = get_canonical_belief_framework_text()
    
    prompt = f"""## Information Space

{info_space_section}

## Beliefs

{beliefs_section}

## EIG Definition

You are evaluating actions taken by an LLM agent in an information environment. Each action returns an observation from the environment containing information, and each action-observation pair is appended to the history.

**Expected Information Gain (EIG)** measures how much the observation from taking an action is expected to reduce uncertainty about D and θ:

EIG(action; history) = I((D,θ); observation | history, action)
                     = H[p(D,θ | history)] − E[H[p(D,θ | history, action, observation)]]

Where:
- H[p(D,θ | history)] is the current entropy (uncertainty) about D and θ given history so far
- E[H[p(D,θ | history, action, observation)]] is the expected entropy after taking action and receiving observation
- The difference is how much entropy about D and θ the action is expected to reduce

ALL actions produce observations from the environment, including THINK. 
A THINK action produces a thought as its observation, which can inform both θ (task beliefs) and D (design beliefs) just like SEARCH or other actions. 
Do not assume THINK has zero EIG - evaluate based on the information content of the thought.

## Environment

{env_description}

## Current State"""
    
    if task_beliefs:
        prompt += f"""

### Current Task Beliefs (θ):
{task_beliefs}"""
    
    if design_beliefs:
        prompt += f"""

### Current Design Beliefs (D):
{design_beliefs}"""
    
    if history:
        prompt += f"""

### Action History:
{format_action_history(history)}"""
    
    prompt += f"""

## Action to Evaluate

**Action Type**: {action_type}
**Action Parameters**: 
{action_params}

## Task

Estimate the Expected Information Gain of this action **in bits**:

1. **Task EIG** (in bits): Information gain about task parameters (θ) - how much will this action reduce uncertainty about the facts, evidence, and signals needed for this task instance?
2. **Design EIG** (in bits): Information gain about design effectiveness (D) - how much will this action reduce uncertainty about which strategies, sources, and approaches are effective for this task family?
3. **Joint EIG** (in bits): Overall information value combining both task-level and design-level insights

Your estimates should be **conservative**: it is fine to underestimate, but you should avoid overestimating the true information gain.

Keep in mind:
- Suboptimal or incorrect actions can still be informative – you can gain information about D and θ even when actions don't directly provide the final answer
- Once you have reasonable information about something in the beliefs, expected information gain from further actions on that aspect should be diminishing
- Avoid hard-coded heuristics; base scores on expected posterior changes given the environment

## Response Format

Provide your EIG analysis as a JSON object with the following structure:

RESPONSE FORMAT (JSON only)
{{
    "task_eig_bits": <float>,  // conservative estimate in bits
    "design_eig_bits": <float>,  // conservative estimate in bits
    "joint_eig_bits": <float>,  // conservative estimate in bits
    "confidence": <float between 0.0 and 1.0>,  // confidence in your EIG estimates
    "reasoning": "<brief explanation of the estimates>"
}}"""
    
    return prompt
