"""Agents: the base agent and the BOED variants.

- BayesianOptimalExperimentalDesignAgent: BOED with beliefs over the task
  parameter (θ) only, selecting actions to maximise expected information gain.
- BOEDCitationTrackerAgent: BOED without domain knowledge, where task knowledge
  is a list of citations, quotes, and holdings described in words.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .tracing import langfuse, observe
from .actions import Action, ActionType, get_action_class
from .environment import Environment, Observation
from .evaluation import parse_predictions
from .llm import ModelAPI
from pydantic import ValidationError

from .parsing import (
    parse_action_response,
    normalize_action_parameters_for_construction,
    parse_prediction_response,
    normalize_yes_no_answers,
    parse_json_from_text,
)
from .prompts import (
    BeliefUpdatePromptConstructor,
    ActionSelectionPromptConstructor,
    PredictionPromptConstructor,
    BOEDBeliefUpdatePromptConstructor,
    BOEDActionSelectionPromptConstructor,
    BOEDPredictionPromptConstructor,
    BOEDCitationTrackerBeliefUpdatePromptConstructor,
    BOEDCitationTrackerPredictionPromptConstructor,
)

logger = logging.getLogger(__name__)


# Type for prompt logging callback: (system_prompt, user_prompt, prompt_type, step) -> None
PromptCallback = Callable[[str, str, str, int], None]

class Agent:
    """
    Base agent class that provides common functionality for LLM-powered agents.
    """
    
    def __init__(self, 
                 environment: Environment, 
                 model_api: ModelAPI,
                 model_id: str,
                 max_tokens: int = 1000,
                 temperature: float = 0.3,
                 temperature_action_selection: float = None,
                 seed: int = None,
                 thinking_enabled: bool = True,
                 open_web_search_enabled: bool = False,
                courtlistener_search_enabled: bool = False,
                courtlistener_opinion_access_enabled: bool = False,
                 prompt_callback: PromptCallback = None):
        """
        Initialize the base agent.
        
        Args:
            environment: The environment to interact with
            model_api: The model API to use for LLM calls
            model_id: The specific model ID to use
            max_tokens: Maximum tokens for LLM responses
            temperature: Default temperature for LLM generation
            temperature_action_selection: Temperature for action selection (defaults to temperature if not set)
            seed: Seed for reproducible LLM outputs
            thinking_enabled: Whether to allow thinking actions
            open_web_search_enabled: Whether to allow open web search actions
            courtlistener_search_enabled: Whether to allow CourtListener search actions
            courtlistener_opinion_access_enabled: Whether to allow CourtListener opinion fetch by ID
            prompt_callback: Optional callback for logging prompts. Called with (system_prompt, user_prompt, prompt_type, step)
        """
        self.environment = environment
        self.model_api = model_api
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.temperature_action_selection = temperature_action_selection if temperature_action_selection is not None else temperature
        self.seed = seed
        self.thinking_enabled = thinking_enabled
        self.open_web_search_enabled = open_web_search_enabled
        self.courtlistener_search_enabled = courtlistener_search_enabled
        self.courtlistener_opinion_access_enabled = courtlistener_opinion_access_enabled
        self.prompt_callback = prompt_callback

        # Initialize the available agent actions from environment
        self.action_space = self.environment.action_space.copy()
        env_action_space = self.environment.action_space  # Original for validation
        
        # Validate that enabled flags don't request actions the environment doesn't support
        self._validate_action_flags(env_action_space)
        
        # Filter actions based on enabled flags
        if not self.thinking_enabled:
            if ActionType.THINK in self.action_space:
                self.action_space.remove(ActionType.THINK)
        
        if not self.open_web_search_enabled:
            if ActionType.OPEN_WEB_SEARCH in self.action_space:
                self.action_space.remove(ActionType.OPEN_WEB_SEARCH)
        
        if not self.courtlistener_search_enabled:
            if ActionType.OPEN_COURTLISTENER_SEARCH in self.action_space:
                self.action_space.remove(ActionType.OPEN_COURTLISTENER_SEARCH)
        
        if not self.courtlistener_opinion_access_enabled:
            if ActionType.ACCESS_COURTLISTENER_OPINION in self.action_space:
                self.action_space.remove(ActionType.ACCESS_COURTLISTENER_OPINION)
        
        self.history: List[Dict[str, Any]] = []
        self.current_step = 0
        
        # Log final action space
        logger.info(f"Initialized agent with model: {model_id}")
        logger.info(f"Agent action space: {[a.name for a in self.action_space]}")
    
    def _validate_action_flags(self, env_action_space: List[ActionType]):
        """
        Validate that agent's enabled flags don't request actions 
        that the environment doesn't support.
        
        Raises:
            ValueError: If agent enables an action not supported by the environment.
        """
        errors = []
        env_actions_str = [a.name for a in env_action_space]
        
        if self.thinking_enabled and ActionType.THINK not in env_action_space:
            errors.append(
                f"thinking_enabled=True but environment doesn't support THINK"
            )
        
        if self.open_web_search_enabled and ActionType.OPEN_WEB_SEARCH not in env_action_space:
            errors.append(
                f"open_web_search_enabled=True but environment doesn't support OPEN_WEB_SEARCH"
            )
        
        if self.courtlistener_search_enabled and ActionType.OPEN_COURTLISTENER_SEARCH not in env_action_space:
            errors.append(
                f"courtlistener_search_enabled=True but environment doesn't support OPEN_COURTLISTENER_SEARCH"
            )
        
        if self.courtlistener_opinion_access_enabled and ActionType.ACCESS_COURTLISTENER_OPINION not in env_action_space:
            errors.append(
                f"courtlistener_opinion_access_enabled=True but environment doesn't support ACCESS_COURTLISTENER_OPINION"
            )
        
        if errors:
            raise ValueError(
                f"Agent action flag mismatch with environment:\n"
                f"  - {chr(10).join('- ' + e for e in errors)}\n"
                f"  Environment action_space: {env_actions_str}\n"
                f"  Fix: Either disable the flag in agent config, or add the action to the environment's action_space."
            )

    def select_action(self, observation: Optional[Observation]) -> Action:
        """
        Select the next action using the LLM.
        This should be overridden by subclasses to provide specific action selection logic.
        """
        raise NotImplementedError("Subclasses must implement select_action")
    
    def update_state(self, action: Action, observation: Observation):
        """
        Update the agent's internal state based on the action and observation.
        This should be overridden by subclasses to provide specific state management.
        """
        raise NotImplementedError("Subclasses must implement update_state")
    
    def log_prompts(self, system_prompt: str, user_prompt: str, prompt_type: str):
        """
        Log prompts via callback if one is registered.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt  
            prompt_type: Type of prompt (e.g., "ACTION_SELECTION", "BELIEF_UPDATE", "PREDICTION")
        """
        if self.prompt_callback:
            self.prompt_callback(system_prompt, user_prompt, prompt_type, self.current_step + 1)


def extract_action_parameters(action: Any) -> Dict[str, Any]:
    """
    Extract parameters from an action using the action class `inputs` schema.
    
    Args:
        action: Action instance to extract parameters from
        
    Returns:
        Dictionary of parameter names to values (only non-None values)
        
    Rules:
        - Include all fields defined in the `inputs` schema (required and optional)
        - For each field, read the value from the action instance when set (not None)
        - Omit fields that are unset (None) to avoid redundant defaults
    """
    params: Dict[str, Any] = {}
    inputs_schema = getattr(action.__class__, 'inputs', None)
    if not isinstance(inputs_schema, dict):
        return params
        
    for name, _spec in inputs_schema.items():
        if hasattr(action, name):
            value = getattr(action, name)
            if value is not None:
                params[name] = value
                
    return params


UNIFORM_TASK_PRIOR = "You have no initial knowledge about the specific task parameters."

# Errors from parsing/validating a model response that warrant re-asking the model.
REASK_ERRORS = (ValueError, ValidationError, TypeError, KeyError)


class BayesianOptimalExperimentalDesignAgent(Agent):
    """
    Bayesian Optimal Experimental Design agent with single-layer beliefs (θ only).
    
    This is a task-agnostic agent that can be configured for different domains
    by injecting appropriate prompt constructors via dependency injection.
    
    The agent:
    - Maintains beliefs about task parameters (θ)
    - Updates beliefs after each observation
    - Selects actions to maximize expected information gain about θ
    - Can make predictions based on accumulated beliefs
    
    Unlike IDS-OED, this agent does not maintain meta-level design beliefs (D).
    """

    MAX_ACTION_REASKS = 3
    
    def __init__(
        self,
        environment: Environment,
        model_api: ModelAPI,
        model_id: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        seed: int = None,
        thinking_enabled: bool = True,
        open_web_search_enabled: bool = False,
        courtlistener_search_enabled: bool = False,
        courtlistener_opinion_access_enabled: bool = False,
        task_belief_prior: str = UNIFORM_TASK_PRIOR,
        belief_update_prompt_constructor: BeliefUpdatePromptConstructor = None,
        action_selection_prompt_constructor: ActionSelectionPromptConstructor = None,
        prediction_prompt_constructor: PredictionPromptConstructor = None,
        max_tokens_config: Optional[Dict[str, int]] = None,
        belief_update_model_id: Optional[str] = None,
        belief_update_temperature: Optional[float] = None,
    ):
        """
        Initialize the BOED agent.
        
        Args:
            environment: The environment to interact with
            model_api: API for making LLM calls
            model_id: Model identifier
            max_tokens: Maximum tokens for LLM responses
            temperature: Temperature for LLM generation
            seed: Seed for reproducible LLM outputs
            thinking_enabled: Whether to allow thinking actions
            open_web_search_enabled: Whether to allow open web search actions
            courtlistener_search_enabled: Whether to allow CourtListener search actions
            courtlistener_opinion_access_enabled: Whether to allow CourtListener opinion fetch by ID
            task_belief_prior: Initial belief about task parameters (θ)
            belief_update_prompt_constructor: Constructor for belief update prompts
            action_selection_prompt_constructor: Constructor for action selection prompts
            prediction_prompt_constructor: Constructor for prediction prompts
            belief_update_model_id: Optional model override for belief updates only
            belief_update_temperature: Optional temperature override for belief updates only
        """
        super().__init__(
            environment=environment,
            model_api=model_api,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            thinking_enabled=thinking_enabled,
            open_web_search_enabled=open_web_search_enabled,
            courtlistener_search_enabled=courtlistener_search_enabled,
            courtlistener_opinion_access_enabled=courtlistener_opinion_access_enabled,
        )
        
        # Initialize beliefs
        self.task_beliefs = task_belief_prior
        self.task_belief_prior = task_belief_prior
        
        # For tasks that output a list (e.g. hallucination checker), store the raw list
        self.last_final_response_list: Optional[List[str]] = None
        
        # Environment description for prompts
        self.environment_description = self.environment.get_environment_description()
        
        # Prompt constructors - default to method-specific constructors if not injected
        if belief_update_prompt_constructor is None:
            logger.warning(
                "No BOED belief_update_prompt_constructor injected; "
                "defaulting to BOEDBeliefUpdatePromptConstructor(domain_knowledge=None)."
            )
        if action_selection_prompt_constructor is None:
            logger.warning(
                "No BOED action_selection_prompt_constructor injected; "
                "defaulting to BOEDActionSelectionPromptConstructor(domain_knowledge=None)."
            )
        if prediction_prompt_constructor is None:
            logger.warning(
                "No BOED prediction_prompt_constructor injected; "
                "defaulting to BOEDPredictionPromptConstructor(domain_knowledge=None)."
            )
        self.belief_update_prompt_constructor = (
            belief_update_prompt_constructor or BOEDBeliefUpdatePromptConstructor()
        )
        self.action_selection_prompt_constructor = (
            action_selection_prompt_constructor or BOEDActionSelectionPromptConstructor()
        )
        self.prediction_prompt_constructor = (
            prediction_prompt_constructor or BOEDPredictionPromptConstructor()
        )
        self.max_tokens_config = max_tokens_config or {}
        self.belief_update_model_id = belief_update_model_id or self.model_id
        self.belief_update_temperature = (
            self.temperature if belief_update_temperature is None else belief_update_temperature
        )
        
        # Track last action for belief updates
        self.last_action = None
        
        logger.info(f"Initialized BayesianOptimalExperimentalDesignAgent")
        logger.info(f"Environment: {self.environment.__class__.__name__}")

    def _call_with_reask(
        self,
        messages: List[Dict[str, str]],
        parse: Callable[[Optional[str]], Any],
        reask: Union[str, Callable[[Exception, Optional[str]], str]],
        attempts: int,
        **model_kwargs,
    ) -> Any:
        """Call the model and parse its response, re-asking on parse failure.

        `parse(response)` returns the parsed value or raises one of REASK_ERRORS.
        On failure the conversation becomes `messages + [assistant: response,
        user: reask]` (where `reask` is a string or `(error, response) -> str`)
        and the model is called again, up to `attempts` calls in total. The
        first call sends `messages` unchanged. Raises the last parse error once
        every attempt has failed.
        """
        conversation = messages
        for attempt in range(attempts):
            response = self.model_api(prompt=conversation, **model_kwargs)
            try:
                return parse(response)
            except REASK_ERRORS as e:
                with langfuse.start_as_current_observation(
                    name="validate-model-response", input=response,
                    output={"error": str(e), "attempt": attempt + 1, "exhausted": attempt + 1 == attempts},
                    level="WARNING",
                ):
                    pass
                logger.warning(
                    "Model response could not be used (attempt %s/%s): %s. Response: %r",
                    attempt + 1, attempts, e, response,
                )
                if attempt + 1 == attempts:
                    raise
                conversation = messages + [
                    {"role": "assistant", "content": response or ""},
                    {"role": "user", "content": reask(e, response) if callable(reask) else reask},
                ]

    @observe(name="update-beliefs", capture_input=False)
    def update_beliefs(self, observation: Observation, action: Action) -> str:
        """
        Update task-level beliefs based on the observation.
        
        Args:
            observation: The observation from the last action
            action: The action that produced the observation
            
        Returns:
            String representation of updated beliefs
        """
        langfuse.update_current_span(input=str(observation), metadata={"step": self.current_step})
        # Skip belief update for initial observation (no action taken yet)
        if not observation or not observation.metadata or action is None:
            logger.info("Skipping belief update for initial observation (no action taken yet)")
            return f"Task Beliefs: {self.task_beliefs}"
        
        if self.belief_update_prompt_constructor is None:
            logger.warning("No belief update prompt constructor configured")
            return f"Task Beliefs: {self.task_beliefs}"
        
        # Build prompts
        system_prompt = self.belief_update_prompt_constructor.get_system_prompt(
            environment_description=self.environment_description
        )
        user_prompt = self.belief_update_prompt_constructor.get_user_prompt(
            previous_beliefs=f"Task Beliefs: {self.task_beliefs}",
            observation=observation,
            action_type=action.action_type.value,
            action_parameters=extract_action_parameters(action),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        belief_update_max_tokens = self.max_tokens_config.get('belief_update', self.max_tokens)
        max_belief_update_attempts = 3

        def non_empty(response: Optional[str]) -> str:
            text = str(response).strip() if response is not None else ""
            if not text:
                raise ValueError("empty belief-update response")
            return text

        logger.info("Calling LLM for BOED belief update using model=%s", self.belief_update_model_id)
        try:
            updated_beliefs = self._call_with_reask(
                messages,
                non_empty,
                reask=(
                    "Your last belief-update response was empty. "
                    "Return a non-empty JSON object with a non-empty 'task_beliefs' string."
                ),
                attempts=max_belief_update_attempts,
                model_id=self.belief_update_model_id,
                max_tokens=belief_update_max_tokens,
                temperature=self.belief_update_temperature,
                seed=self.seed,
            )
        except ValueError as e:
            raise RuntimeError(
                f"Belief update returned empty response after {max_belief_update_attempts} attempts."
            ) from e

        logger.info(f"LLM belief update response: {updated_beliefs}")
        
        # Parse task beliefs from response
        self._parse_belief_update(updated_beliefs)
        
        logger.info(f"Updated task beliefs: {self.task_beliefs}")
        
        return f"Task Beliefs: {self.task_beliefs}"
    
    # Markers after which the task beliefs text starts, tried in order:
    # markdown heading, bold label, plain label with colon.
    _TASK_BELIEFS_MARKERS = (
        r"(?:^|\n)#+\s*Task\s*Beliefs[^\n]*\n",              # ### Task Beliefs ...
        r"(?:^|\n)\s*\*\*\s*Task\s*Beliefs[^\n]*\*\*\s*\n",  # **Task Beliefs:** ...
        r"Task\s*Beliefs\s*:\s*",                            # Task Beliefs: ...
    )

    def _parse_belief_update(self, text: str) -> None:
        task_section = None

        for marker in self._TASK_BELIEFS_MARKERS:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                task_section = text[match.end():].strip()
                break

        # Try JSON format: {"task_beliefs": ...}
        if task_section is None:
            try:
                data = parse_json_from_text(text, "belief update")
                if data:
                    task_raw = data.get("task_beliefs", "")
                    # Handle both string and nested object
                    if isinstance(task_raw, dict):
                        task_section = task_raw.get("summary", str(task_raw))
                    else:
                        task_section = str(task_raw) if task_raw else None
            except Exception:
                pass
        
        # Assign parsed section or fall back to full text
        if task_section is not None:
            self.task_beliefs = task_section
        else:
            if not text.strip():
                logger.warning("Empty belief update response from LLM, keeping previous beliefs")
            else:
                # Use full response as task beliefs
                self.task_beliefs = text
    
    @observe(name="select-action", capture_input=False, capture_output=False)
    def select_action(self, observation: Optional[Observation]) -> Action:
        """
        Select the next action using the BOED framework.
        
        Chooses actions that maximize expected information gain over task parameters (θ).
        On the final step, uses the prediction prompt (get_current_prediction) to force PROVIDE_FINAL_RESPONSE.
        
        Args:
            observation: Current observation (None for initial state)
            
        Returns:
            The selected action
        """
        langfuse.update_current_span(input=str(observation), metadata={"step": self.current_step + 1})
        logger.info(f"BOED action selection (attempting step {self.current_step + 1})")
        
        # Step 1: Update beliefs based on current observation
        if observation:
            self.update_beliefs(observation, self.last_action)
        
        # Check if this is the final step - if so, use prediction prompt to force PROVIDE_FINAL_RESPONSE
        is_final_step = (self.current_step + 1) >= self.environment.max_steps
        
        if is_final_step:
            logger.info("Final step reached - using prediction prompt to force PROVIDE_FINAL_RESPONSE")
            max_prediction_retries = 3
            prediction, confidence = None, None
            for attempt in range(max_prediction_retries):
                prediction, confidence = self.get_current_prediction()
                is_valid = prediction is not None and (
                    (prediction if isinstance(prediction, str) else str(prediction)).strip()
                    and not (isinstance(prediction, list) and len(prediction) == 0)
                )
                if is_valid:
                    break
                logger.warning(f"Final response empty or invalid (attempt {attempt + 1}/{max_prediction_retries}), retrying...")
            # Store list for evaluation (handles JSON array strings and plain text)
            self.store_final_response(prediction)
            # Create PROVIDE_FINAL_RESPONSE action from prediction
            action_class = get_action_class(ActionType.PROVIDE_FINAL_RESPONSE)
            action = action_class(response=prediction)
        else:
            # Build action selection prompts and get action from LLM. When the
            # model chooses PROVIDE_FINAL_RESPONSE, _call_llm_for_action_selection
            # stores the parsed list for evaluation.
            messages = self._build_action_selection_prompts(observation)
            if messages is None:
                return None
            action = self._call_llm_for_action_selection(messages)

        if action:
            # Increment step counter after successful parsing
            self.current_step += 1
            self.last_action = action
            logger.info(f"Selected action: {action.action_type.value} (step {self.current_step})")
            langfuse.update_current_span(output={
                "action_type": action.action_type.value,
                "parameters": action.get_input_parameters(),
            })
        else:
            langfuse.update_current_span(level="ERROR", status_message="Action selection returned no action")
        
        return action
    
    def _build_action_selection_prompts(self, observation: Optional[Observation]) -> Optional[List[Dict[str, str]]]:
        """
        Build prompts for action selection.
        
        Args:
            observation: Current observation (None for initial state)
            
        Returns:
            List of message dictionaries for LLM call, or None if constructor not configured
        """
        if self.action_selection_prompt_constructor is None:
            logger.error("No action selection prompt constructor configured")
            return None
        
        system_prompt = self.action_selection_prompt_constructor.get_system_prompt(
            action_space=self._get_available_action_types(),
            environment_description=self.environment.get_action_selection_environment_description(),
            search_capabilities=self.environment.get_search_capabilities(),
        )

        user_prompt = self.action_selection_prompt_constructor.get_user_prompt(
            observation=observation,
            history=self.history,
            current_beliefs=f"Task beliefs: {self.task_beliefs}",
            max_steps=self.environment.max_steps,
            # The environment has no per-instance description; the prompt renders "" as "N/A".
            task_instance_description="",
            response_requirements=self.environment.get_response_requirements(),
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _call_llm_for_action_selection(self, messages: List[Dict[str, str]]) -> Optional[Action]:
        """
        Call the LLM for action selection and parse the response into an Action.

        On a malformed or invalid response, re-ask with the validation error
        (up to MAX_ACTION_REASKS times) before giving up.

        Returns:
            Parsed Action object, or None if every attempt fails
        """
        logger.info("Calling LLM for BOED action selection")
        action_selection_max_tokens = self.max_tokens_config.get('action_selection', self.max_tokens)

        def parse_action(response: Optional[str]) -> Action:
            action_type, parameters = parse_action_response(response)
            if ActionType(action_type) not in self.action_space:
                raise ValueError(
                    f"{action_type} is not available in this run. Available actions: "
                    f"{[a.value for a in self.action_space]}"
                )
            if action_type == ActionType.PROVIDE_FINAL_RESPONSE.value:
                self.store_final_response(parameters.get("response"))
            else:
                self.last_final_response_list = None
            parameters = normalize_action_parameters_for_construction(action_type, parameters)
            return get_action_class(ActionType(action_type))(**parameters)

        def reask(error: Exception, response: Optional[str]) -> str:
            return (
                "Your previous response was not a valid action. Error:\n"
                f"{error}\n\n"
                "Respond again with a single JSON object of the form "
                '{"action": {"action_type": "...", <required parameters>}, "reasoning": "..."} '
                "using one of the available actions and non-empty required parameters."
            )

        try:
            return self._call_with_reask(
                messages,
                parse_action,
                reask,
                attempts=self.MAX_ACTION_REASKS + 1,
                model_id=self.model_id,
                temperature=self.temperature,
                max_tokens=action_selection_max_tokens,
                seed=self.seed,
            )
        except REASK_ERRORS as e:
            logger.error(f"Action parsing failed after {self.MAX_ACTION_REASKS + 1} attempts: {e}")
            return None

    def _get_available_action_types(self) -> List[ActionType]:
        return self.action_space.copy()
    
    def store_final_response(self, response: Any) -> None:
        """Record the final response as the list of predicted hallucinations.

        A JSON list (a list value or a string containing one) becomes one item
        per element; any other non-empty string counts as a single prediction.
        """
        self.last_final_response_list = parse_predictions(response)

    def update_state(self, action: Action, observation: Observation):
        """
        Update the agent's internal state based on the action and observation.
        
        Args:
            action: The action that was taken
            observation: The resulting observation
        """
        self.history.append({
            'step': self.current_step,
            'action': action,
            'observation': observation,
            'task_beliefs': self.task_beliefs
        })
        logger.info(f"Updated state - Step {self.current_step}: {action.action_type.value}")
    
    def get_current_beliefs(self) -> Dict[str, str]:
        return {
            'task_beliefs': self.task_beliefs,
            'design_beliefs': ''  # Empty for BOED - only has task beliefs
        }
    
    @observe(name="predict-hallucinations", capture_input=False)
    def get_current_prediction(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Get the agent's current best prediction based on accumulated beliefs.
        
        Returns:
            Tuple of (prediction, confidence) where confidence is 0.0-1.0
        """
        try:
            env_description = self.environment.get_environment_description()
            response_requirements = self.environment.get_response_requirements()
            # The environment has no per-instance description; the prompt omits the section for "".
            task_instance_description = ""

            # Build prediction prompt using the prediction prompt constructor
            max_steps = self.environment.max_steps
            if max_steps == 0:
                response_requirements = response_requirements.replace(
                    "that you have labeled as hallucinated in your Current Task Beliefs",
                    "that you determine to be hallucinated based on the task description",
                ).replace(
                    "If you have N items marked hallucinated in your beliefs, your response must contain exactly those N segments (or the citation alone when sub-items are implied).",
                    "Your response must contain exactly those segments (or the citation alone when sub-items are implied).",
                )
            system_prompt = self.prediction_prompt_constructor.get_system_prompt(
                environment_description=env_description,
                max_steps=max_steps,
            )

            user_prompt = self.prediction_prompt_constructor.get_user_prompt(
                task_beliefs=self.task_beliefs,
                task_instance_description=task_instance_description,
                response_requirements=response_requirements,
                history=self.history,
                max_steps=max_steps,
            )
            
            prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            langfuse.update_current_span(input=prompt, metadata={"step": self.current_step})
            
            # Get prediction from the agent's own model; re-ask once if the
            # response cannot be parsed.
            prediction_max_tokens = self.max_tokens_config.get('prediction', self.max_tokens)
            try:
                prediction, confidence = self._call_with_reask(
                    prompt,
                    parse_prediction_response,
                    reask=(
                        "Your last response was empty or could not be parsed. "
                        "Return a non-empty JSON object with an 'action' field containing a non-empty 'response' string "
                        "and a 'confidence' field (0.0-1.0)."
                    ),
                    attempts=2,
                    model_id=self.model_id,
                    max_tokens=prediction_max_tokens,
                    temperature=0.1,  # Low temperature for consistent predictions
                    seed=self.seed,
                )
            except REASK_ERRORS:
                langfuse.update_current_span(level="ERROR", status_message="Prediction parsing failed after retry")
                return None, None

            # Normalize Yes/No answers if applicable
            try:
                prediction = normalize_yes_no_answers(prediction)
            except ValueError:
                logger.warning(f"Could not normalize prediction format: {prediction}")

            logger.info(f"Current prediction: {prediction} (confidence: {confidence:.3f})")
            return prediction, confidence
                
        except Exception as e:
            logger.error(f"Error getting current prediction: {e}")
            langfuse.update_current_span(level="ERROR", status_message=type(e).__name__)
            return None, None
    
    def reset(self):
        super().reset() if hasattr(super(), 'reset') else None
        self.task_beliefs = self.task_belief_prior
        self.last_action = None
        self.history = []
        self.current_step = 0
        logger.info("Reset BOED agent to initial state")


CITATION_TRACKER_PRIOR = """Your task knowledge is a list of citations, quotes, and holdings from the brief, described in words. Keep a numbered or bulleted list. For each item note: (1) the citation, quote, or holding, (2) status: pending, verified as correct, or hallucinated. When you update your beliefs after an observation, never delete items—only add new items or update the status of items you just verified. Current list: None yet. Add items as you identify them from the brief and update status as you verify them."""


class BOEDCitationTrackerAgent(BayesianOptimalExperimentalDesignAgent):
    """
    BOED agent with no domain knowledge. Task knowledge is a list of citations,
    quotes, and holdings from the brief—described in words (natural language).
    The prior and belief-update instructions tell the model to keep the full list
    and never remove items, only add or update status, so it is less likely to
    forget previously checked citations.
    """

    def __init__(
        self,
        environment: Environment,
        model_api: ModelAPI,
        model_id: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        seed: int = None,
        thinking_enabled: bool = True,
        open_web_search_enabled: bool = False,
        courtlistener_search_enabled: bool = False,
        courtlistener_opinion_access_enabled: bool = False,
        task_belief_prior: str = "",
        belief_update_prompt_constructor: BeliefUpdatePromptConstructor = None,
        action_selection_prompt_constructor: ActionSelectionPromptConstructor = None,
        prediction_prompt_constructor: PredictionPromptConstructor = None,
        max_tokens_config: Optional[Dict[str, int]] = None,
        belief_update_model_id: Optional[str] = None,
        belief_update_temperature: Optional[float] = None,
    ):
        # No domain knowledge; use prior that asks for list described in words
        no_domain = None
        belief_constructor = belief_update_prompt_constructor or BOEDCitationTrackerBeliefUpdatePromptConstructor(no_domain)
        action_constructor = action_selection_prompt_constructor or BOEDActionSelectionPromptConstructor(no_domain)
        pred_constructor = prediction_prompt_constructor or BOEDCitationTrackerPredictionPromptConstructor(no_domain)

        super().__init__(
            environment=environment,
            model_api=model_api,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            thinking_enabled=thinking_enabled,
            open_web_search_enabled=open_web_search_enabled,
            courtlistener_search_enabled=courtlistener_search_enabled,
            courtlistener_opinion_access_enabled=courtlistener_opinion_access_enabled,
            task_belief_prior=task_belief_prior or CITATION_TRACKER_PRIOR,
            belief_update_prompt_constructor=belief_constructor,
            action_selection_prompt_constructor=action_constructor,
            prediction_prompt_constructor=pred_constructor,
            max_tokens_config=max_tokens_config,
            belief_update_model_id=belief_update_model_id,
            belief_update_temperature=belief_update_temperature,
        )
        logger.info(
            "Initialized BOEDCitationTrackerAgent (no domain knowledge; task list in words)"
        )
        
