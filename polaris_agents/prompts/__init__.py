"""
Prompt module for LLM agent prompts.

This module provides:
- Base classes for prompt construction (DomainKnowledgeProvider, PromptConstructors)
- EIG formulation definitions
- Agent-specific prompt constructors (in agents/)
- Environment-specific prompts and domain knowledge (in environments/)

Structure:
- prompts/agents/ - BOED and BOED citation-tracker prompt constructors
- prompts/environments/ - Task-specific prompts and domain knowledge providers
"""

from .base import (
    EIGFormulation,
    DomainKnowledgeProvider,
    BeliefUpdatePromptConstructor,
    ActionSelectionPromptConstructor,
    PredictionPromptConstructor,
)
from .eig_formulations import get_eig_objective_text
from .utils import (
    format_action_history,
    format_observation_result,
    create_selection_actions_description,
    create_actions_parameters_description,
    create_action_selection_json_format,
    create_explicit_eig_json_format,
    create_prediction_json_format,
    create_belief_update_json_format,
)

# Agent-specific prompt constructors (from agents/)
from .agents import (
    # BOED (single beliefs: θ only)
    BOEDBeliefUpdatePromptConstructor,
    BOEDActionSelectionPromptConstructor,
    BOEDPredictionPromptConstructor,
)

__all__ = [
    # Base classes
    "EIGFormulation",
    "DomainKnowledgeProvider",
    "BeliefUpdatePromptConstructor",
    "ActionSelectionPromptConstructor",
    "PredictionPromptConstructor",
    # EIG
    "get_eig_objective_text",
    # BOED prompt constructors (single beliefs)
    "BOEDBeliefUpdatePromptConstructor",
    "BOEDActionSelectionPromptConstructor",
    "BOEDPredictionPromptConstructor",
    # Utilities
    "format_action_history",
    "format_observation_result",
    "create_selection_actions_description",
    "create_actions_parameters_description",
    # JSON format templates
    "create_action_selection_json_format",
    "create_explicit_eig_json_format",
    "create_prediction_json_format",
    "create_belief_update_json_format",
]
