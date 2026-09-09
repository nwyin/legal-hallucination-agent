"""Prompt construction: base constructors, shared formatting helpers, task domain
knowledge for the legal hallucination checker, and the BOED prompt constructors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .actions import ActionType, get_action_class

if TYPE_CHECKING:
    from .environment import Observation

logger = logging.getLogger(__name__)


# --- Base classes for prompt construction ---


class DomainKnowledgeProvider(ABC):
    """
    Abstract base class for providing task-family specific knowledge about D and θ.

    Each task family (legal judgment, forecasting, theorem proving, etc.) should
    have a concrete implementation that provides:
    - What θ (task parameters) means for this task family
    - What D (design effectiveness) means for this task family
    - Optional examples and guidance specific to the domain

    This knowledge is injected into prompts to help the agent understand
    what information to gather and how to reason about uncertainty.
    """

    @abstractmethod
    def get_theta_description(self) -> str:
        """
        Get the description of θ (task parameters) for this task family.

        Returns:
            A description of what task-instance-specific information the agent
            should gather to make accurate predictions.
        """

    def get_action_selection_guidance(self) -> str | None:
        """
        Optional task-specific guidance for action selection (e.g. which actions
        yield information gain, how to avoid unproductive loops).

        Returns:
            Guidance string to inject into the action selection prompt, or None to skip.
        """
        return None

    def get_action_selection_task_section(self) -> str | None:
        """
        Optional short task/domain block for action selection. If provided, replaces
        the default θ + domain definition block so the prompt stays concise.

        Returns:
            A brief "task and domain" section (e.g. 2–5 lines), or None to use default.
        """
        return None


class BeliefUpdatePromptConstructor(ABC):
    """
    Abstract base class for constructing belief update prompts.

    Belief update prompts are used after each action-observation pair to
    update the agent's beliefs about θ and/or D.
    """

    @abstractmethod
    def get_system_prompt(self, **kwargs) -> str:
        """
        Get the system prompt for belief updating.

        Args:
            **kwargs: Implementation-specific arguments (e.g., environment_description)

        Returns:
            The system prompt string.
        """

    @abstractmethod
    def get_user_prompt(self, **kwargs) -> str:
        """
        Get the user prompt for belief updating.

        Args:
            **kwargs: Implementation-specific arguments (e.g., previous_beliefs,
                     observation, action)

        Returns:
            The user prompt string.
        """


class ActionSelectionPromptConstructor(ABC):
    """
    Abstract base class for constructing action selection prompts.

    Action selection prompts guide the agent to choose the next action
    based on current beliefs, history, and the EIG objective.
    """

    @abstractmethod
    def get_system_prompt(self, action_space: list[ActionType], **kwargs) -> str:
        """
        Get the system prompt for action selection.

        Args:
            action_space: List of available action types
            **kwargs: Implementation-specific arguments

        Returns:
            The system prompt string.
        """

    @abstractmethod
    def get_user_prompt(self, observation: Observation | None, **kwargs) -> str:
        """
        Get the user prompt for action selection.

        Args:
            observation: Current observation (may be None for initial state)
            **kwargs: Implementation-specific arguments (e.g., history, beliefs)

        Returns:
            The user prompt string.
        """


class PredictionPromptConstructor(ABC):
    """
    Abstract base class for constructing prediction prompts.

    Prediction prompts are used to query the agent's current best prediction
    based on accumulated beliefs and evidence.
    """

    @abstractmethod
    def get_system_prompt(self, **kwargs) -> str:
        """
        Get the system prompt for prediction.

        Args:
            **kwargs: Implementation-specific arguments

        Returns:
            The system prompt string.
        """

    @abstractmethod
    def get_user_prompt(self, **kwargs) -> str:
        """
        Get the user prompt for prediction.

        Args:
            **kwargs: Implementation-specific arguments (e.g., current_beliefs,
                     task_description)

        Returns:
            The user prompt string.
        """


# --- Shared prompt formatting helpers ---


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


def build_action_guidelines(action_space: list[ActionType]) -> str:
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
            description = getattr(action_class, "description", "")

            # Warn if description is missing - this will cause suboptimal agent behavior
            if not description:
                logger.warning(
                    f"Action class {action_class.__name__} (action_type={action_type.value}) "
                    f"does not have a 'description' attribute defined. This will cause suboptimal "
                    f"agent behavior as the agent won't understand what this action does."
                )
                description = (
                    f"[WARNING: No description defined for {action_type.value}]"
                )

            # Warn if inputs are missing - this will also cause issues
            if not hasattr(action_class, "inputs") or not action_class.inputs:
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
        guidelines.append(
            "- OPEN_WEB_SEARCH performs Google search on the open internet (web, news, and Google Scholar). Use this to find current information, news, scholarly articles, and web content."
        )
    if ActionType.COURTLISTENER_CITATION_LOOKUP in action_space:
        guidelines.append("")
        guidelines.append("**About COURTLISTENER_CITATION_LOOKUP:**")
        guidelines.append(
            "- For reporter-style citations (e.g. '965 F.2d 962', '143 S. Ct. 1196'), use **COURTLISTENER_CITATION_LOOKUP** first with the `cite` parameter. Use OPEN_COURTLISTENER_SEARCH only if citation lookup fails or you have only a case name."
        )

    return "\n".join(guidelines)


def create_selection_actions_description(action_space: list[ActionType]) -> str:
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
                    param_type = param_spec.get("type", "string")
                    param_description = param_spec.get("description", "")
                    is_required = param_spec.get("required", True)

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


def format_action_history(history: list[dict], max_actions: int = -1) -> str:
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
        action = step["action"]
        obs = step["observation"]

        action_type = action.action_type.value
        parameters = action.get_input_parameters()

        history_text += f"Step {i + 1}: {action_type}\n"
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
        if "count" in result:
            result_text = f"Found {result['count']} results"
            if result.get("results"):
                if "snippet" in result["results"][0]:
                    result_text += f" (showing snippets of first {result.get('snippets_shown', len(result['results']))} results)"
                    for i, result_item in enumerate(result["results"][:3]):
                        case_name = result_item.get("case_name", f"Case {i}")
                        snippet = (
                            result_item.get("snippet", "")[:100] + "..."
                            if len(result_item.get("snippet", "")) > 100
                            else result_item.get("snippet", "")
                        )
                        result_text += f"\n- {case_name}: {snippet}"
                else:
                    for i, result_item in enumerate(result["results"][:3]):
                        case_name = result_item.get("case_name", f"Case {i}")
                        result_text += f"\n- {case_name}"
            return result_text
        elif "error" in result:
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


# --- Legal hallucination checker: domain knowledge and task prompts ---


class LegalHallucinationCheckerDomainKnowledge(DomainKnowledgeProvider):
    """
    Domain knowledge for legal citation hallucination checker.

    θ represents task instance-specific information about which citations/sentences are hallucinated:
    - For each citation in the brief: whether it exists, is accurately quoted, has correct pincites, supports the correct holding
    - Evidence from search results and opinion fetches that confirms or refutes each citation
    """

    def get_theta_description(self) -> str:
        return """### Task Parameters (θ)
Task instance-specific information needed to identify hallucinated citations in the brief.

Information that directly updates your knowledge about which citations/sentences are hallucinated:
- For each citation: whether the case exists, whether the case name matches the case name in the citation, whether the quoted language appears in the opinion, whether the pincite is correct
- Evidence from CourtListener searches and opinion fetches that confirms or refutes each citation
- Any misattributions, fabricated quotes, or non-existent case citations you discover

### Domain knowledge: legal case citations
- **Citation format**: 
    Typical format is Case Name, Volume Reporter Page (e.g., 557 F.2d 170). The part after the comma is the reporter citation; "at" introduces a pincite (specific page).
- **Quotation format**:
    ONLY words, phrases, and sentences that are inside quotation marks are considered direct quotes. 
    (i.e. in sentence: The PLRA never takes more than 20 percent of a prisoner's assets not only supports the Courts holding of constitutionality, but was a "critical factor" in it, the quoted text is ONLY critical factor, which should appear in the original opinion, without quotes.)
    The presence of ellipses (...) or bracketed ellipses ([...]) only permits omission of intervening text.
    (i.e. in sentence: [W]here Congress explicitly enumerates certain exceptions to a general prohibition), the word [W]here would NOT appear verbatim in the opinion and should be OMITTED from string match searches.)
    Return the entire sentence as hallucinated if any visible portion of the quotation does not appear verbatim in the opinion. 
- **Holding format**: The holding the citation is usually IMMEDIATELY PRECEDES the citation.

If the case does not exist or the name does not match, the entire citation is hallucinated—no need to separately verify quotes or holdings for that citation.
- **Verification hierarchy**: 
    If a citation is already determined to be hallucinated (e.g., case does not exist or case name does not match), treat all quotes and holdings under that citation as hallucinated.
    In this case, ONLY return the citations that are hallucinated, no need to return the quotes and holdings."""

    def get_domain_knowledge_description(self) -> str:
        return """### Domain knowledge: legal case citations
- **Citation format**:
    Typical format is Case Name, Volume Reporter Page (e.g., 557 F.2d 170). The part after the comma is the reporter citation; "at" introduces a pincite (specific page).
- **Quotation format**:
    ONLY words, phrases, and sentences that are inside quotation marks are considered direct quotes.
    (i.e. in sentence: The PLRA never takes more than 20 percent of a prisoner's assets not only supports the Courts holding of constitutionality, but was a "critical factor" in it, the quoted text is ONLY critical factor, which should appear in the original opinion, without quotes.)
    The presence of ellipses (...) or bracketed ellipses ([...]) only permits omission of intervening text.
    (i.e. in sentence: [W]here Congress explicitly enumerates certain exceptions to a general prohibition), the word [W]here would NOT appear verbatim in the opinion and should be OMITTED from string match searches.)
    Return the entire sentence as hallucinated if any visible portion of the quotation does not appear verbatim in the opinion.
- **Holding format**: The holding the citation is usually IMMEDIATELY PRECEDES the citation.

If the case does not exist or the name does not match, the entire citation is hallucinated—no need to separately verify quotes or holdings for that citation.
- **Verification hierarchy**:
    If a citation is already determined to be hallucinated (e.g., case does not exist or case name does not match), treat all quotes and holdings under that citation as hallucinated.
    In this case, ONLY return the citations that are hallucinated, no need to return the quotes and holdings."""

    def get_action_selection_task_section(self) -> str | None:
        return """You are verifying citations in a legal brief for hallucinations. Your uncertainty (θ) is: which citations are hallucinated (non-existent, misquoted, or wrong pincite). You reduce that uncertainty by using search and opinion actions to gather evidence; then you submit your final list of hallucinated citations."""

    def get_action_selection_guidance(self) -> str | None:
        return """For this task you must **verify citations** by gathering evidence from external sources. Apply the following when choosing actions:

- **THINK has zero information gain**: The observation from THINK only echoes your thought
- **To reduce uncertainty about θ, you must use**: OPEN_COURTLISTENER_SEARCH (find cases), OPEN_WEB_SEARCH (existence checks), ACCESS_COURTLISTENER_OPINION (fetch opinion by ID), SEARCH_LOCAL_OPINION (search within a fetched opinion for quoted language). These actions return new evidence; THINK does not.
- **Reserve PROVIDE_FINAL_RESPONSE** until you have used search/opinion actions to verify ALL citations, or you have exhausted steps."""


def get_search_capabilities_open_search() -> str:
    """
    Get search capabilities for hallucination checker (OPEN_WEB_SEARCH + CourtListener).
    """
    return """## Search Capabilities

You have access to:

- **COURTLISTENER_CITATION_LOOKUP**: Resolve a reporter citation (e.g., `934 F.3d 53`, `143 S. Ct. 1196`) to canonical case info. Use `cite`.
- **OPEN_COURTLISTENER_SEARCH**: Search CourtListener by case name or citation text. Use `query`.
- **ACCESS_COURTLISTENER_OPINION**: Fetch an opinion by `opinion_id`. Full text is stored locally and registered for reading.
- **SEARCH_LOCAL_OPINION**: Search within a fetched opinion using `opinion_id` and `search_string`.
- **READ_DOCUMENT**: Read a fetched opinion in sections. Use `opinion_id` = `opinion_<id>` (or just the numeric `<id>`), `start_line` (0-indexed), and `num_lines`. Use this to read the full opinion in chunks when SEARCH_LOCAL_OPINION is not enough.
- **EDIT_SCRATCHPAD**: Take notes. Use `operation`: `append` (add to end), `insert` (at `position`), `replace` (at `position`), or `clear`. Use `content` for the text and `position` (0-indexed) for insert/replace.
- **OPEN_WEB_SEARCH**: Search the open web for legal information. Use `query`.

---
### Case-law search policy (important)
- If a **reporter-style citation** is present, **use COURTLISTENER_CITATION_LOOKUP first**.
- Use OPEN_COURTLISTENER_SEARCH only if citation lookup fails or only a case name is available.
- Use OPEN_WEB_SEARCH: CourtListener is inconclusive or fails to find the case.

---
### COURTLISTENER_CITATION_LOOKUP – How to use
- Input only the reporter citation AS IS from the brief.
- DO NOT omit 'at' and page numbers following it.
- If a match is returned, use the associated `opinion_id`.

---
### OPEN_COURTLISTENER_SEARCH – How to search
- Never include pincites or page numbers following “at” (they are pinpoint pages, not identifiers).
- Reduce citations to volume + reporter + first page only.
- Prefer **quoted reporter citations** (e.g., `"557 F.2d 170"`) or **case name queries**, not both.
- Do not combine multiple identifiers in one query.
- Expect reporter and punctuation variation (e.g., `F.2d` vs `F2d`).
- If no high-confidence match is found, escalate to OPEN_WEB_SEARCH.

---
### SEARCH_LOCAL_OPINION – How to search
- Search for SHORT distinctive substrings and use the returned snippet to verify the quoted language.
- Look for key noun phrases.
- DO NOT include words that are not in quotation marks and words in square brackets.
- Use READ_DOCUMENT to read the full opinion in sections if string search is inconclusive.

---
### READ_DOCUMENT – How to read the opinion
- Use the opinion_id to identify which opinion to read.
- Iteratively read the opinion in sections to find the quoted language or holdings.

---
### OPEN_WEB_SEARCH – How to search
- Do not issue strict multi-field queries (case name + docket + WL citation).
- Use loose, normalized case names without punctuation or “v.” formatting.
- Prefer either case name or reporter citation initially.
- Avoid quoting long strings unless retrying after a loose search fails.
- Use web search for existence checks, context, or recovery when CourtListener fails.

---
### General verification flow
1. Resolve the case (citation lookup → CourtListener search → web fallback).
2. Fetch the opinion by `opinion_id`.
3. Verify quoted language with SEARCH_LOCAL_OPINION.
4. Verify holdings with READ_DOCUMENT, and EDIT_SCRATCHPAD.
"""


def get_response_requirements() -> str:
    """
    Get task-specific response requirements for hallucination checker predictions.

    Returns:
        Response requirements string (list of hallucinated citations)
    """
    return (
        "Provide a **list** of hallucinated citations, case names,quotes, and holdings.\n\n"
        '**Format:** A JSON array of only the hallucinated segments, e.g. `["string1", "string2"]`.\n\n'
        "**Completeness (required):** Your list must include **every** citation, quote, or holding that you have labeled as hallucinated in your Current Task Beliefs. "
        "Do not omit any segment you believe is hallucinated — the response is evaluated against the full set. If you have N items marked hallucinated in your beliefs, your response must contain exactly those N segments (or the citation alone when sub-items are implied).\n\n"
        "**Important**: If a citation itself is hallucinated, it is assumed that all the quotes and holdings within that citation are hallucinated as well so there is no need to return them separately. "
        "If no hallucinations are found, return an empty list: `[]`.\n\n"
        "**No reasoning in the final answer:** The `response` field must contain **only** the list —- no reasoning, explanation, or prose. Put all reasoning, analysis, and explanation in the `reasoning` field, not in `response`."
        " Only return the hallucinated segments: If a holding is hallucinated, return the FULL sentence of the holding."
    )


# --- BOED prompt constructors (beliefs over the task parameter only) ---


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
        action_parameters: dict | None = None,
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
        action_space: list[ActionType],
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

        # Optional short task section from domain (replaces long θ + domain block when set)
        if self.domain_knowledge and hasattr(
            self.domain_knowledge, "get_action_selection_task_section"
        ):
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
        if self.domain_knowledge and hasattr(
            self.domain_knowledge, "get_action_selection_guidance"
        ):
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
        observation: Observation | None,
        history: list,
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
{observation.result if observation else "Initial state"}

### Recent Actions:
{format_action_history(history, max_actions=self.max_history_actions) if history else "No previous actions"}

### Task Instance
{task_instance_description or "N/A"}
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
        max_steps: int | None = None,
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
        history: list[dict] | None = None,
        max_steps: int | None = None,
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
                history_text = format_action_history(
                    history, max_actions=self.max_history_actions
                )
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


# --- BOED citation-tracker prompt constructors ---


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

        # Build framework: canonical definition + domain-specific if available
        framework = """

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
        action_parameters: dict | None = None,
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
            if self.domain_knowledge and hasattr(
                self.domain_knowledge, "get_domain_knowledge_description"
            ):
                theta_desc = self.domain_knowledge.get_domain_knowledge_description()
                theta_section = f"### Domain-Specific Definition\n{theta_desc}\n\n"
            return f"""You are a legal expert tasked with verifying case citations, quotes and holdings in briefs. Provide your best prediction based on the task description.

{theta_section}## Environment
{environment_description}"""
        else:
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
        history: list[dict] | None = None,
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
