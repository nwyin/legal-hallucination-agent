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


def create_citation_tracker_belief_update_json_format() -> str:
    """
    Create JSON format instructions for BOED belief update prompts.
    
    Returns:
        Formatted string with belief update JSON format
    """
    return """## Response Format
Provide your updated beliefs as a JSON object:

```json
{
    "task_beliefs": "<your list of citations, quotes, and holdings from the brief, described in words. Keep a numbered or bulleted list. For each item note: (1) the citation, quote, or holding, (2) status: pending, verified as correct, or hallucinated.>"
}
```

**Important**: The value must be a text string (natural language), NOT nested JSON."""

class BOEDCitationTrackerBeliefUpdatePromptConstructor(BeliefUpdatePromptConstructor):
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
        framework = f"""

You maintain a Bayesian belief p(θ), described in natural language, that is updated based on observations from actions taken in the information environment.
Your task is to maintain a list of citations, quotes, and holdings from the brief, described in words. 
Keep a numbered or bulleted list. For each item note: (1) the citation, quote, or holding, (2) status: pending, verified as correct, or hallucinated, (3) the associated opinion id if applicable."""
        
        domain_block = ""
        if self.domain_knowledge:
            domain_block = f"""

## Domain / task parameters (θ)
{self.domain_knowledge.get_theta_description()}"""
        
        return f"""You are an LLM agent taking actions within an information environment to solve a task. Each action you take returns an observation that provides information to help you make an accurate prediction.

You use Bayesian Optimal Experimental Design (BOED) for action selection, which focuses on reducing uncertainty about task-specific information.

{framework}
{domain_block}

## Belief Update Process
Your role is to maintain beliefs about the list of citations, quotes, and holdings from the brief. Think of your beliefs as a distribution over possible states of the world, not a single point estimate.

When you receive an observation:
- Update your beliefs about the list of citations, quotes, and holdings from the brief based on any new information that directly informs your prediction for this specific task instance.

## Environment
{environment_description}

## Principles
- Be **additive and information-dense**: build upon previous knowledge rather than replacing it
- Preserve prior beliefs unless contradicted by new evidence
- Track multiple hypotheses and interpretations, not just a single narrative
- Note the evidence supporting or contradicting different possibilities
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
Update your beliefs about the list of citations, quotes, and holdings from the brief.

Consider:
- What new instance-specific information was revealed?
- How does it change your understanding of this task instance?
- What key uncertainties remain, and what information would be most valuable to resolve them?

## Output Format
Provide your updated beliefs in natural language. Your beliefs should be **additive and information-dense** - build upon your previous knowledge rather than replacing it. Show how your understanding has grown and evolved.

Structure your response as:
Task Beliefs: [Your list of citations, quotes, and holdings from the brief, described in words. Keep a numbered or bulleted list. For each item note: (1) the citation, quote, or holding, (2) status: pending, verified as correct, or hallucinated.]

{create_citation_tracker_belief_update_json_format()}"""

class BOEDCitationTrackerPredictionPromptConstructor(PredictionPromptConstructor):
    """
    Prompt constructor for generating predictions in BOED framework.
    
    Used to query the agent's current best prediction based on accumulated
    beliefs about θ.
    """
    
    def __init__(
        self,
        domain_knowledge: DomainKnowledgeProvider = None,
    ):
        """
        Initialize the prediction prompt constructor.
        
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
        Get the system prompt for prediction.
        
        Args:
            environment_description: Description of the task environment
            
        Returns:
            System prompt for prediction
        """
        max_steps = kwargs.get("max_steps")

        if max_steps == 0:
            # Direct prediction (no search steps): simplified intro + domain knowledge only
            theta_section = ""
            if self.domain_knowledge and hasattr(self.domain_knowledge, "get_domain_knowledge_description"):
                theta_desc = self.domain_knowledge.get_domain_knowledge_description()
                theta_section = f"### Domain-Specific Definition\n{theta_desc}\n\n"
            return f"""You are a legal expert tasked with verifying case citations, quotes and holdings in briefs. Provide your best prediction based on the task description.

{theta_section}## Environment
{environment_description}"""
        else:
            from ..utils import get_canonical_theta_definition
            theta_def = get_canonical_theta_definition()
            theta_section = f"## Task Parameters (θ)\n{theta_def}"
            if self.domain_knowledge:
                theta_desc = self.domain_knowledge.get_theta_description()
                theta_section += f"\n\n### Domain-Specific Definition\n{theta_desc}"

            return f"""You are an LLM agent that has been taking actions within an information environment to solve a task. You are now making your final predictions based on the information you have gathered.

{theta_section}

## Environment
{environment_description}"""
    
    def get_user_prompt(
        self,
        task_beliefs: str,
        task_instance_description: str = "",
        response_requirements: str = "",
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        Get the user prompt for prediction.
        
        Args:
            task_beliefs: Current task-level beliefs (θ)
            task_instance_description: Description of the task instance
            response_requirements: Task-specific requirements for the response format
            history: Optional action history (list of action-observation pairs)
            
        Returns:
            User prompt for prediction
        """
        # Order: beliefs → history → task instance → response requirements
        sections = []
        
        # Show beliefs
        sections.append(f"## Current Task Beliefs (θ)\n{task_beliefs}")
        
        # Then show action history if it exists
        if history:
            history_text = format_action_history(history)
            if history_text:
                sections.append(f"## Action History\n{history_text}")
        
        # Then task instance
        if task_instance_description:
            sections.append(f"## Task Instance\n{task_instance_description}")
        
        # Response requirements
        if response_requirements:
            sections.append(f"## Response Requirements\n{response_requirements}")
        
        content = "\n\n".join(sections)
        
        return f"""{content}

## Task
Based on your current beliefs and observations, provide your final prediction.

```json
{{
  "action": {{
    "action_type": "PROVIDE_FINAL_RESPONSE",
    "response": "<list of strings; ONLY your prediction in the format specified in Response Requirements>"
  }},
  "confidence": <float between 0.0 and 1.0 representing your confidence in this prediction>,
  "reasoning": "<ALL your explanations, reasoning, and analysis go here>"
}}
```"""
