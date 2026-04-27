"""
Base classes for prompt construction.

This module defines:
- EIGFormulation: Enum for different EIG objective formulations
- DomainKnowledgeProvider: ABC for task-family specific θ/D definitions
- Prompt constructor ABCs for belief update, action selection, and prediction
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any

from ..agents.action import ActionType
from ..environments.base import Observation


class EIGFormulation(Enum):
    """
    Different formulations for the Expected Information Gain objective.
    
    - JOINT: Maximize EIG(θ,D | action) - joint information gain about task and design
    - ADDITIVE: EIG(θ | action) + EIG(D | action) - explicit decomposition
    - IDS_RATIO: Minimize (ExpectedRegret)² / EIG(θ | action) - Information-Directed Sampling
    
    Reference: Russo & Van Roy (2017) "Learning to Optimize Via Information-Directed Sampling"
    https://arxiv.org/pdf/1403.5556
    """
    JOINT = "joint"
    ADDITIVE = "additive"
    IDS_RATIO = "ids_ratio"


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
        pass
    
    @abstractmethod
    def get_design_description(self) -> str:
        """
        Get the description of D (design effectiveness) for this task family.
        
        Returns:
            A description of meta-level knowledge about strategies, heuristics,
            and sources that improve performance across tasks of this type.
        """
        pass
    
    def get_action_selection_guidance(self) -> Optional[str]:
        """
        Optional task-specific guidance for action selection (e.g. which actions
        yield information gain, how to avoid unproductive loops).
        
        Returns:
            Guidance string to inject into the action selection prompt, or None to skip.
        """
        return None

    def get_action_selection_task_section(self) -> Optional[str]:
        """
        Optional short task/domain block for action selection. If provided, replaces
        the default θ + domain definition block so the prompt stays concise.
        
        Returns:
            A brief "task and domain" section (e.g. 2–5 lines), or None to use default.
        """
        return None

    def get_belief_framework_description(self) -> str:
        """
        Get the full belief framework description combining D and θ.
        
        Returns:
            Formatted string describing the two belief layers.
        """
        theta_desc = self.get_theta_description()
        design_desc = self.get_design_description()
        
        description = f"""## Belief Framework
You maintain beliefs over two kinds of uncertainty:

{theta_desc}

{design_desc}

The boundary between design parameters (D) and task parameters (θ) is sometimes blurry. 
Some information can inform both belief spaces."""
        
        return description


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
        pass
    
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
        pass


class ActionSelectionPromptConstructor(ABC):
    """
    Abstract base class for constructing action selection prompts.
    
    Action selection prompts guide the agent to choose the next action
    based on current beliefs, history, and the EIG objective.
    """
    
    @abstractmethod
    def get_system_prompt(self, action_space: List[ActionType], **kwargs) -> str:
        """
        Get the system prompt for action selection.
        
        Args:
            action_space: List of available action types
            **kwargs: Implementation-specific arguments
            
        Returns:
            The system prompt string.
        """
        pass
    
    @abstractmethod
    def get_user_prompt(self, observation: Optional[Observation], **kwargs) -> str:
        """
        Get the user prompt for action selection.
        
        Args:
            observation: Current observation (may be None for initial state)
            **kwargs: Implementation-specific arguments (e.g., history, beliefs)
            
        Returns:
            The user prompt string.
        """
        pass


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
        pass
    
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
        pass
