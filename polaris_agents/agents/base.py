"""
Base Agent class for LLM-powered agents.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Callable
from .action import Action, ActionType
from ..environments.base import Environment, Observation
from ..models.llm import ModelAPI

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
                 closed_search_enabled: bool = True,
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
            closed_search_enabled: Whether to allow closed search actions
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
        self.closed_search_enabled = closed_search_enabled
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
        
        if not self.closed_search_enabled:
            if ActionType.CLOSED_SEARCH in self.action_space:
                self.action_space.remove(ActionType.CLOSED_SEARCH)
        
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
        
        if self.closed_search_enabled and ActionType.CLOSED_SEARCH not in env_action_space:
            errors.append(
                f"closed_search_enabled=True but environment doesn't support CLOSED_SEARCH"
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
