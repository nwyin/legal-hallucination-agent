"""Agents: the base agent and the BOED variants.

- BayesianOptimalExperimentalDesignAgent: BOED with beliefs over the task
  parameter (θ) only, selecting actions to maximise expected information gain.
- BOEDCitationTrackerAgent: BOED without domain knowledge, where task knowledge
  is a list of citations, quotes, and holdings described in words.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .actions import Action, ActionType, get_action_class
from .environment import Environment, Observation
from .llm import ModelAPI
from .parsing import (
    parse_action_output_with_fallback,
    create_unified_action_guard,
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
                 provider: str,
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
        self.provider = provider
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
    
    def __init__(
        self,
        environment: Environment,
        model_api: ModelAPI,
        model_id: str,
        provider: str,
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
        belief_update_provider: Optional[str] = None,
        belief_update_temperature: Optional[float] = None,
    ):
        """
        Initialize the BOED agent.
        
        Args:
            environment: The environment to interact with
            model_api: API for making LLM calls
            model_id: Model identifier
            provider: Model provider (openrouter only)
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
            belief_update_provider: Optional provider override for belief updates only
            belief_update_temperature: Optional temperature override for belief updates only
        """
        normalized_provider = (provider or "openrouter").lower().strip()
        if normalized_provider != "openrouter":
            logger.warning(
                f"Provider '{provider}' is not supported in this project. "
                "Forcing provider to 'openrouter' for single-provider setup."
            )
            normalized_provider = "openrouter"

        normalized_belief_update_provider = (
            (belief_update_provider or normalized_provider).lower().strip()
        )
        if normalized_belief_update_provider != "openrouter":
            logger.warning(
                f"belief_update_provider '{belief_update_provider}' is not supported. "
                "Forcing belief update provider to 'openrouter'."
            )
            normalized_belief_update_provider = "openrouter"

        super().__init__(
            environment=environment,
            model_api=model_api,
            model_id=model_id,
            provider=normalized_provider,
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
        self.belief_update_provider = normalized_belief_update_provider
        self.belief_update_temperature = (
            self.temperature if belief_update_temperature is None else belief_update_temperature
        )
        
        # Track last action for belief updates
        self.last_action = None
        
        logger.info(f"Initialized BayesianOptimalExperimentalDesignAgent")
        logger.info(f"Environment: {self.environment.__class__.__name__}")
    
    def update_beliefs(self, observation: Observation, action: Action) -> str:
        """
        Update task-level beliefs based on the observation.
        
        Args:
            observation: The observation from the last action
            action: The action that produced the observation
            
        Returns:
            String representation of updated beliefs
        """
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
        max_belief_update_retries = 3
        updated_beliefs = ""
        for attempt in range(max_belief_update_retries):
            if attempt == 0:
                attempt_messages = messages
            else:
                reask_msg = (
                    "Your last belief-update response was empty. "
                    "Return a non-empty JSON object with a non-empty 'task_beliefs' string."
                )
                attempt_messages = messages + [{"role": "user", "content": reask_msg}]

            logger.info(
                "Calling LLM for BOED belief update (attempt %s/%s) using model=%s provider=%s",
                attempt + 1,
                max_belief_update_retries,
                self.belief_update_model_id,
                self.belief_update_provider,
            )
            response = self.model_api(
                model_id=self.belief_update_model_id,
                prompt=attempt_messages,
                max_attempts=1,
                provider=self.belief_update_provider,
                max_tokens=belief_update_max_tokens,
                temperature=self.belief_update_temperature,
                seed=self.seed
            )
            if response is None:
                updated_beliefs = ""
            else:
                updated_beliefs = str(response).strip()
            if updated_beliefs:
                break

        if not updated_beliefs:
            raise RuntimeError(
                f"Belief update returned empty response after {max_belief_update_retries} attempts."
            )

        logger.info(f"LLM belief update response: {updated_beliefs}")
        
        # Parse task beliefs from response
        self._parse_belief_update(updated_beliefs)
        
        logger.info(f"Updated task beliefs: {self.task_beliefs}")
        
        return f"Task Beliefs: {self.task_beliefs}"
    
    def _parse_belief_update(self, text: str) -> None:
        task_section = None
        
        # Try markdown headings format: ### Task Beliefs ...
        try:
            task_heading = re.search(r"(?:^|\n)#+\s*Task\s*Beliefs[^\n]*\n", text, flags=re.IGNORECASE)
            if task_heading:
                task_section = text[task_heading.end():].strip()
        except Exception:
            pass
        
        # Try markdown bold labels: **Task Beliefs:** ...
        if task_section is None:
            try:
                task_bold = re.search(r"(?:^|\n)\s*\*\*\s*Task\s*Beliefs[^\n]*\*\*\s*\n", text, flags=re.IGNORECASE)
                if task_bold:
                    task_section = text[task_bold.end():].strip()
            except Exception:
                pass
        
        # Try label with colon format: Task Beliefs: ...
        if task_section is None:
            try:
                m = re.search(r"Task\s*Beliefs\s*:\s*([\s\S]*)", text, flags=re.IGNORECASE)
                if m:
                    task_section = m.group(1).strip()
            except Exception:
                pass
        
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
        logger.info(f"BOED action selection (attempting step {self.current_step + 1})")
        
        # Step 1: Update beliefs based on current observation
        if observation:
            self.update_beliefs(observation, self.last_action)
        
        # Check if this is the final step - if so, use prediction prompt to force PROVIDE_FINAL_RESPONSE
        max_steps = getattr(self.environment, 'max_steps', 10)
        is_final_step = (self.current_step + 1) >= max_steps
        
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
            # Store list for evaluation (handles JSON array strings, semicolon-separated, etc.)
            self._store_final_response_list_if_applicable(
                ActionType.PROVIDE_FINAL_RESPONSE.value,
                {"response": prediction}
            )
            # Create PROVIDE_FINAL_RESPONSE action from prediction
            action_class = get_action_class(ActionType.PROVIDE_FINAL_RESPONSE)
            action = action_class(response=prediction)
        else:
            # Build action selection prompts and get action from LLM
            messages = self._build_action_selection_prompts(observation)
            if messages is None:
                return None
            action = self._call_llm_for_action_selection(messages)
            # When agent chooses PROVIDE_FINAL_RESPONSE (before final step), store for evaluation
            if action and action.action_type == ActionType.PROVIDE_FINAL_RESPONSE:
                self._store_final_response_list_if_applicable(
                    ActionType.PROVIDE_FINAL_RESPONSE.value,
                    action.get_input_parameters() if hasattr(action, 'get_input_parameters') else {"response": getattr(action, 'response', None)}
                )
        
        if action:
            # Increment step counter after successful parsing
            self.current_step += 1
            self.last_action = action
            logger.info(f"Selected action: {action.action_type.value} (step {self.current_step})")
        
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
        
        action_space = self._get_available_action_types()
        
        # Get optional search capabilities from environment
        search_capabilities = ""
        if hasattr(self.environment, 'get_search_capabilities'):
            search_capabilities = self.environment.get_search_capabilities()
        
        # Get task instance description and response requirements (for PROVIDE_FINAL_RESPONSE format)
        task_instance_description = ""
        if hasattr(self.environment, 'get_task_instance_description'):
            task_instance_description = self.environment.get_task_instance_description()
        response_requirements = ""
        if hasattr(self.environment, 'get_response_requirements'):
            response_requirements = self.environment.get_response_requirements()
        
        action_selection_environment_description = self.environment_description
        if hasattr(self.environment, "get_action_selection_environment_description"):
            action_selection_environment_description = self.environment.get_action_selection_environment_description()

        system_prompt = self.action_selection_prompt_constructor.get_system_prompt(
            action_space=action_space,
            environment_description=action_selection_environment_description,
            search_capabilities=search_capabilities,
        )
        
        user_prompt = self.action_selection_prompt_constructor.get_user_prompt(
            observation=observation,
            history=self.history,
            current_beliefs=f"Task beliefs: {self.task_beliefs}",
            max_steps=getattr(self.environment, 'max_steps', 10),
            task_instance_description=task_instance_description,
            response_requirements=response_requirements,
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _call_llm_for_action_selection(self, messages: List[Dict[str, str]]) -> Optional[Action]:
        """
        Call LLM for action selection and parse the response.
        
        Args:
            messages: List of message dictionaries for LLM call
            
        Returns:
            Parsed Action object, or None if parsing fails
        """
        logger.info("Calling LLM for BOED action selection")
        
        action_selection_max_tokens = self.max_tokens_config.get('action_selection', self.max_tokens)
        
        def build_action(validated_output: Dict[str, Any]) -> Action:
            action_type = validated_output["action_type"]
            parameters = validated_output["parameters"]
            self._store_final_response_list_if_applicable(action_type, parameters)
            parameters = normalize_action_parameters_for_construction(action_type, parameters)
            action_class = get_action_class(ActionType(action_type))
            return action_class(**parameters)

        try:
            # Try unified guard first (handles formatting and validation)
            unified_guard = create_unified_action_guard(
                model_api=self.model_api,
                model_id=self.model_id,
                provider=self.provider,
                max_tokens=action_selection_max_tokens,
                temperature=self.temperature,
                num_reasks=3
            )
            guard_response = unified_guard(messages)
            validated_output = guard_response.validated_output
            if validated_output:
                try:
                    return build_action(validated_output)
                except ValueError as ve:
                    if "query" in str(ve).lower() or "non-empty" in str(ve).lower():
                        reask_msg = (
                            "Your previous response had an empty search query. "
                            "For OPEN_COURTLISTENER_SEARCH or OPEN_WEB_SEARCH you must provide a non-empty 'query' "
                            "(e.g. a citation like '965 F.2d 962' or a case name). Please respond again with a valid query."
                        )
                        guard_response2 = unified_guard(messages + [{"role": "user", "content": reask_msg}])
                        if guard_response2.validated_output:
                            return build_action(guard_response2.validated_output)
                    raise
            raise ValueError("Guard returned no validated output")
        except Exception as e:
            logger.warning(f"Unified guard parsing failed: {e}, trying fallback approach")
            try:
                response = self.model_api(
                    model_id=self.model_id,
                    prompt=messages,
                    max_attempts=1,
                    provider=self.provider,
                    max_tokens=action_selection_max_tokens,
                    temperature=self.temperature,
                    seed=self.seed
                )
                validated_output = parse_action_output_with_fallback(response)
                action_type = validated_output["action_type"]
                parameters = validated_output["parameters"]
                try:
                    return build_action(validated_output)
                except ValueError as ve:
                    if "query" in str(ve).lower() or "non-empty" in str(ve).lower():
                        reask_content = (
                            "Your previous response had an empty search query. "
                            "For OPEN_COURTLISTENER_SEARCH or OPEN_WEB_SEARCH you must provide a non-empty 'query' "
                            "(e.g. a citation or case name). Reply with a new JSON action including a valid query."
                        )
                        response2 = self.model_api(
                            model_id=self.model_id,
                            prompt=messages + [{"role": "user", "content": reask_content}],
                            max_attempts=1,
                            provider=self.provider,
                            max_tokens=action_selection_max_tokens,
                            temperature=self.temperature,
                            seed=self.seed
                        )
                        validated_output2 = parse_action_output_with_fallback(response2)
                        return build_action(validated_output2)
                    raise
            except Exception as fallback_e:
                logger.warning(f"Fallback parsing also failed: {fallback_e}, returning None")
                return None
    
    def _get_available_action_types(self) -> List[ActionType]:
        return self.action_space.copy()
    
    def _store_final_response_list_if_applicable(self, action_type: str, parameters: Dict[str, Any]) -> None:
        self.last_final_response_list = None
        if action_type != ActionType.PROVIDE_FINAL_RESPONSE.value:
            return
        r = parameters.get("response")
        if isinstance(r, list):
            self.last_final_response_list = [str(x) for x in r]
        elif isinstance(r, str) and r:
            # Try JSON array (e.g. '["cite1", "cite2"]'); otherwise treat whole string as single item (no ";" split)
            r_stripped = r.strip()
            if r_stripped.startswith("["):
                try:
                    parsed = json.loads(r_stripped)
                    if isinstance(parsed, list):
                        self.last_final_response_list = [str(x) for x in parsed]
                    else:
                        self.last_final_response_list = [r]
                except (json.JSONDecodeError, TypeError):
                    self.last_final_response_list = [r]
            else:
                # Single string (semicolon no longer used as separator to avoid splitting citation text)
                self.last_final_response_list = [r_stripped] if r_stripped else []
        else:
            self.last_final_response_list = []
    
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
    
    def get_current_prediction(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Get the agent's current best prediction based on accumulated beliefs.
        
        Returns:
            Tuple of (prediction, confidence) where confidence is 0.0-1.0
        """
        try:
            # Get environment description
            env_description = ""
            if hasattr(self.environment, 'get_environment_description'):
                env_description = self.environment.get_environment_description()
            elif hasattr(self.environment, 'get_task_instance_description'):
                env_description = self.environment.get_task_instance_description()
            
            # Get response requirements if available
            response_requirements = ""
            if hasattr(self.environment, 'get_response_requirements'):
                response_requirements = self.environment.get_response_requirements()
            
            # Get task instance description if available
            task_instance_description = ""
            if hasattr(self.environment, 'get_task_instance_description'):
                task_instance_description = self.environment.get_task_instance_description()
            
            # Build prediction prompt using the prediction prompt constructor
            max_steps = getattr(self.environment, 'max_steps', None)
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
            
            # Get prediction from the agent's own model
            prediction_max_tokens = self.max_tokens_config.get('prediction', self.max_tokens)
            prediction_kwargs = dict(
                model_id=self.model_id,
                prompt=prompt,
                max_attempts=3,
                provider=self.provider,
                max_tokens=prediction_max_tokens,
                temperature=0.1,  # Low temperature for consistent predictions
                seed=self.seed,
            )
            response = self.model_api(**prediction_kwargs)

            # Parse the response; if it fails, retry once with a re-ask message.
            try:
                prediction, confidence = parse_prediction_response(response)
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse prediction response: {response}, error: {e}")
                reask_msg = (
                    "Your last response was empty or could not be parsed. "
                    "Return a non-empty JSON object with an 'action' field containing a non-empty 'response' string "
                    "and a 'confidence' field (0.0-1.0)."
                )
                retry_prompt = prompt + [
                    {"role": "assistant", "content": response or ""},
                    {"role": "user", "content": reask_msg},
                ]
                logger.info("Retrying prediction with re-ask message")
                retry_kwargs = {**prediction_kwargs, "prompt": retry_prompt}
                try:
                    response = self.model_api(**retry_kwargs)
                    prediction, confidence = parse_prediction_response(response)
                except (ValueError, KeyError) as e2:
                    logger.warning(f"Retry also failed: {response}, error: {e2}")
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
        provider: str,
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
        belief_update_provider: Optional[str] = None,
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
            provider=provider,
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
            belief_update_provider=belief_update_provider,
            belief_update_temperature=belief_update_temperature,
        )
        logger.info(
            "Initialized BOEDCitationTrackerAgent (no domain knowledge; task list in words)"
        )
        
