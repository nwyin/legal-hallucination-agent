"""
BOED (Bayesian Optimal Experimental Design) Prompt Constructors.

These constructors generate prompts for the BOED agent framework, which maintains
beliefs over task parameters (θ) only, selecting actions to maximize expected
information gain about the task instance.

Unlike IDS-OED which maintains double beliefs (D and θ), BOED focuses solely on
task-level information gathering.

The constructors are task-agnostic - they use DomainKnowledgeProvider to inject
task-family specific knowledge about θ.
"""

from typing import Optional, List, Dict, Any

from ..base import (
    DomainKnowledgeProvider,
    BeliefUpdatePromptConstructor,
    ActionSelectionPromptConstructor,
    PredictionPromptConstructor,
)
from ..utils import (
    format_action_history,
    create_selection_actions_description,
    create_action_selection_json_format,
    build_action_guidelines,
)
from ...agents.action import ActionType, get_action_class
from ...environments.base import Observation


def create_boed_belief_update_json_format() -> str:
    """
    Create JSON format instructions for BOED belief update prompts.
    
    Returns:
        Formatted string with belief update JSON format
    """
    return """## Response Format
Provide your updated beliefs as a JSON object:

```json
{
    "task_beliefs": "<your rich, cumulative understanding of this task instance>"
}
```

Write the belief as a **long, detailed natural language paragraph** (or multiple paragraphs). Be thorough and information-dense - include all relevant details, evidence, hypotheses, and reasoning. The text can be several paragraphs long.

**Important**: The value must be a text string (natural language), NOT nested JSON."""


class BOEDBeliefUpdatePromptConstructor(BeliefUpdatePromptConstructor):
    """
    Prompt constructor for belief updates in BOED framework.
    
    After each action-observation pair, updates beliefs about:
    - θ (task parameters): Instance-specific information needed for prediction
    """
    
    def __init__(
        self,
        domain_knowledge: DomainKnowledgeProvider = None,
    ):
        """
        Initialize the belief update prompt constructor.
        
        Args:
            domain_knowledge: Provider for task-family specific θ definitions (None = no domain-specific prompts)
        """
        self.domain_knowledge = domain_knowledge
    
    def get_system_prompt(
        self,
        environment_description: str,
        **kwargs,
    ) -> str:
        """
        Get the system prompt for belief updating.
        
        Args:
            environment_description: Description of the task environment
            
        Returns:
            System prompt for belief updating
        """
        from ..utils import get_canonical_theta_definition
        
        theta_def = get_canonical_theta_definition()
        
        # Build framework: canonical definition + domain-specific if available
        framework = f"""## Task Parameters (θ)
{theta_def}"""
        
        if self.domain_knowledge:
            theta_desc = self.domain_knowledge.get_theta_description()
            framework += f"""

### Domain-Specific Definition
{theta_desc}"""
        
        framework += """

You maintain a Bayesian belief p(θ), described in natural language, that is updated based on observations from actions taken in the information environment."""
        
        return f"""You are an LLM agent taking actions within an information environment to solve a task. Each action you take returns an observation that provides information to help you make an accurate prediction.

You use Bayesian Optimal Experimental Design (BOED) for action selection, which focuses on reducing uncertainty about task-specific information.

{framework}

## Belief Update Process
Your role is to maintain rich, evolving beliefs about θ (task parameters). Think of your beliefs as a distribution over possible states of the world, not a single point estimate.

When you receive an observation:
- Update your task-level beliefs (θ) based on any new information that directly informs your prediction for this specific task instance.

## Environment
{environment_description}

## Principles
- Be **additive and information-dense**: build upon previous knowledge rather than replacing it
- Preserve prior beliefs unless contradicted by new evidence
- Track multiple hypotheses and interpretations, not just a single narrative
- Note the evidence supporting or contradicting different possibilities
- Evaluate source reliability and relevance
- Focus on instance-specific facts, signals, and multiple possible interpretations
- Acknowledge uncertainty and identify what information would be most valuable next"""
    
    def get_user_prompt(
        self,
        previous_beliefs: str,
        observation: Observation,
        action_type: str,
        action_parameters: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """
        Get the user prompt for belief updating.
        
        Args:
            previous_beliefs: Current belief state before update
            observation: The observation from the action
            action_type: Type of action that produced the observation
            action_parameters: Parameters of the action
            
        Returns:
            User prompt for belief updating
        """
        return f"""## Previous Beliefs
{previous_beliefs}

## New Observation
Action type: {action_type}
Action parameters: {action_parameters}
Observation: {observation}

## Task
Update your beliefs about task parameters (θ) - instance-specific information for this task.

Consider:
- What new instance-specific information was revealed?
- How does it change your understanding of this task instance?
- What are the different possible interpretations or hypotheses? Which seem more or less likely now?
- What key uncertainties remain, and what information would be most valuable to resolve them?

## Output Format
Provide your updated beliefs in natural language. Your beliefs should be **additive and information-dense** - build upon your previous knowledge rather than replacing it. Show how your understanding has grown and evolved.

**Task Beliefs (θ):** Include all relevant information you've learned about this specific task instance, showing how your understanding has developed. Track multiple hypotheses or interpretations where appropriate, noting the evidence for each and your current confidence. Identify what uncertainties remain and what information would be most valuable.

Structure your response as:
Task Beliefs: [Your rich, cumulative understanding of this task instance, with multiple hypotheses and their evidence where appropriate]

{create_boed_belief_update_json_format()}"""


class BOEDActionSelectionPromptConstructor(ActionSelectionPromptConstructor):
    """
    Prompt constructor for action selection in BOED framework.
    
    Guides the agent to choose actions that maximize expected information gain
    about task parameters (θ).
    """
    
    def __init__(
        self,
        domain_knowledge: DomainKnowledgeProvider = None,
        max_history_actions: int = 5,
    ):
        """
        Initialize the action selection prompt constructor.
        
        Args:
            domain_knowledge: Provider for task-family specific θ definitions (None = no domain-specific prompts)
            max_history_actions: Max number of recent actions to include in prompts
        """
        self.domain_knowledge = domain_knowledge
        self.max_history_actions = max_history_actions
    
    def get_system_prompt(
        self,
        action_space: List[ActionType],
        environment_description: str,
        search_capabilities: str = "",
        **kwargs,
    ) -> str:
        """
        Get the system prompt for action selection.
        
        Args:
            action_space: List of available action types
            environment_description: Description of the task environment
            search_capabilities: Optional description of available search types
            
        Returns:
            System prompt for action selection
        """
        from ..utils import get_canonical_theta_definition
        
        # Optional short task section from domain (replaces long θ + domain block when set)
        if self.domain_knowledge and hasattr(self.domain_knowledge, "get_action_selection_task_section"):
            short_section = self.domain_knowledge.get_action_selection_task_section()
            if short_section:
                theta_section = f"## Task\n{short_section}"
                # Include θ/domain description so agent sees citation knowledge when choosing actions
                theta_section += f"\n\n### Domain / task parameters (θ)\n{self.domain_knowledge.get_theta_description()}"
            else:
                short_section = None
        else:
            short_section = None
        if not short_section:
            theta_def = get_canonical_theta_definition()
            theta_section = f"""## Task Parameters (θ)
{theta_def}"""
            if self.domain_knowledge:
                theta_desc = self.domain_knowledge.get_theta_description()
                theta_section += f"""

### Domain-Specific Definition
{theta_desc}"""
            theta_section += """

You maintain a Bayesian belief p(θ) that is updated based on observations from actions."""
        
        actions_desc = create_selection_actions_description(action_space)
        
        # Build action guidelines dynamically
        action_guidelines = build_action_guidelines(action_space)
        
        # Optional task-specific action selection guidance (e.g. THINK has zero EIG, prefer search)
        action_selection_guidance = ""
        if self.domain_knowledge and hasattr(self.domain_knowledge, "get_action_selection_guidance"):
            guidance = self.domain_knowledge.get_action_selection_guidance()
            if guidance:
                action_selection_guidance = f"\n\n## Task-Specific Guidance\n{guidance}"
        
        return f"""You are an LLM agent taking actions within an information environment to solve a task. Each action you take returns an observation that provides information to help you make an accurate prediction.

You use Bayesian Optimal Experimental Design (BOED) for action selection, which focuses on reducing uncertainty about task-specific information.

{theta_section}

## Action Selection Objective
Choose the action that maximizes Expected Information Gain (EIG) about θ.
Estimate the EIG of an action by considering how much the observation from this action 
will reduce your uncertainty and help you learn information about θ.

EIG(θ | action) = I(θ; observation | action, history)

**Action Selection Process:**
1. Consider your candidate actions
2. For each action, estimate how much its observation would reduce your uncertainty about θ (task-specific facts)
3. Select the ONE action (action type and parameters) with the highest expected information gain

**Key principles:**
- Consider carefully how each action – both its action type and parameters – will inform your task beliefs (θ)
- An action that tells you little new provides low information gain
- Prefer actions that resolve the most impactful uncertainties for making an accurate prediction

## Environment
{environment_description}

## Available Actions
{actions_desc}

{search_capabilities if search_capabilities else ""}

{action_guidelines}
{action_selection_guidance}

**Important**:
- Select exactly ONE action with all required parameters
- Prefer actions for which you expect the observation to reduce the most impactful uncertainties about θ

{create_action_selection_json_format()}"""
    
    def get_user_prompt(
        self,
        observation: Optional[Observation],
        history: List,
        current_beliefs: str,
        max_steps: int,
        task_instance_description: str = "",
        response_requirements: str = "",
        **kwargs,
    ) -> str:
        """
        Get the user prompt for action selection.
        
        Args:
            observation: Current observation (None for initial state)
            history: List of previous actions
            current_beliefs: Current belief state
            max_steps: Maximum number of steps allowed
            task_instance_description: Description of the current task instance
            response_requirements: Task-specific format for PROVIDE_FINAL_RESPONSE (so voluntary final answers match)
            
        Returns:
            User prompt for action selection
        """
        steps_remaining = max_steps - len(history)
        response_requirements_block = ""
        if response_requirements:
            response_requirements_block = f"""

## Response Requirements (for PROVIDE_FINAL_RESPONSE)
When you choose PROVIDE_FINAL_RESPONSE, the "response" field must follow this format exactly:
{response_requirements}
"""
        return f"""## Current State

### Current Beliefs:
{current_beliefs}

### Current Observation:
{observation.result if observation else 'Initial state'}

### Recent Actions:
{format_action_history(history, max_actions=self.max_history_actions) if history else 'No previous actions'}

### Task Instance
{task_instance_description or 'N/A'}
{response_requirements_block}
## Task
Choose the **single next action** that will maximize expected information gain about:
- **Task parameters (θ)**: Instance-specific information for accurate prediction

Consider:
1. What you already know about θ (instance-specific evidence and signals)
2. What information would most reduce uncertainty about the correct prediction
3. Which action is most likely to provide that information

You have **{steps_remaining}** steps remaining.
{"If this is your final step, you must use PROVIDE_FINAL_RESPONSE." if steps_remaining <= 1 else ""}

Provide your response as the required JSON format."""


class BOEDPredictionPromptConstructor(PredictionPromptConstructor):
    """
    Prompt constructor for generating predictions in BOED framework.
    
    Used to query the agent's current best prediction based on accumulated
    beliefs about θ.
    """
    
    def __init__(
        self,
        domain_knowledge: DomainKnowledgeProvider = None,
        max_history_actions: int = 5,
    ):
        """
        Initialize the prediction prompt constructor.
        
        Args:
            domain_knowledge: Provider for task-family specific θ definitions (None = no domain-specific prompts)
            max_history_actions: Max number of recent actions to include in prompts
        """
        self.domain_knowledge = domain_knowledge
        self.max_history_actions = max_history_actions
    
    def get_system_prompt(
        self,
        environment_description: str,
        max_steps: int = None,
        **kwargs,
    ) -> str:
        """
        Get the system prompt for prediction.

        Args:
            environment_description: Description of the task environment
            max_steps: Maximum steps allowed (0 = direct prediction with no search)

        Returns:
            System prompt for prediction
        """
        from ..utils import get_canonical_theta_definition

        if max_steps == 0:
            return f"""You are a legal expert tasked with verifying case citations, quotes and holdings in briefs. Provide your best prediction based on the task description.

## Environment
{environment_description}"""

        theta_def = get_canonical_theta_definition()

        # Build framework: canonical definition + domain-specific if available
        theta_section = f"""## Task Parameters (θ)
{theta_def}"""

        if self.domain_knowledge:
            theta_desc = self.domain_knowledge.get_theta_description()
            theta_section += f"""

### Domain-Specific Definition
{theta_desc}"""

        return f"""You are an LLM agent that has been taking actions within an information environment to solve a task. You are now making your final prediction based on the information you have gathered.

{theta_section}

## Environment
{environment_description}"""
    
    def get_user_prompt(
        self,
        task_beliefs: str,
        task_instance_description: str = "",
        response_requirements: str = "",
        history: Optional[List[Dict]] = None,
        max_steps: int = None,
        **kwargs,
    ) -> str:
        """
        Get the user prompt for prediction.

        Args:
            task_beliefs: Current task-level beliefs (θ)
            task_instance_description: Description of the task instance
            response_requirements: Task-specific requirements for the response format
            history: Optional action history (list of action-observation pairs)
            max_steps: Maximum steps allowed (0 = direct prediction with no search)

        Returns:
            User prompt for prediction
        """
        sections = []

        if max_steps != 0:
            # Show beliefs and action history only when search steps were taken
            sections.append(f"## Current Task Beliefs (θ)\n{task_beliefs}")

            if history:
                history_text = format_action_history(history, max_actions=self.max_history_actions)
                if history_text:
                    sections.append(f"## Action History\n{history_text}")

        # Task instance
        if task_instance_description:
            sections.append(f"## Task Instance\n{task_instance_description}")

        # Response requirements
        if response_requirements:
            sections.append(f"## Response Requirements\n{response_requirements}")

        content = "\n\n".join(sections)

        task_instruction = (
            "Based on the task description, provide your best prediction."
            if max_steps == 0
            else "Based on your current beliefs and observations, provide your final prediction."
        )
        reasoning_field = (
            ""
            if max_steps == 0
            else ',\n  "reasoning": "<ALL your explanations, reasoning, and analysis go here>"'
        )

        return f"""{content}

## Task
{task_instruction}

```json
{{
  "action": {{
    "action_type": "PROVIDE_FINAL_RESPONSE",
    "response": "<ONLY your prediction in the format specified in Response Requirements>"
  }},
  "confidence": <float between 0.0 and 1.0 representing your confidence in this prediction>{reasoning_field}
}}
```"""
