"""
Prompt module for LLM agent prompts.

This module provides:
- Base classes for prompt construction (DomainKnowledgeProvider, PromptConstructors)
- Agent-specific prompt constructors (in agents/)
- Environment-specific prompts and domain knowledge (in environments/)

Structure:
- prompts/agents/ - BOED and BOED citation-tracker prompt constructors
- prompts/environments/ - Task-specific prompts and domain knowledge providers
"""

from .base import (
    DomainKnowledgeProvider,
    BeliefUpdatePromptConstructor,
    ActionSelectionPromptConstructor,
    PredictionPromptConstructor,
)
from .utils import (
    format_action_history,
    format_observation_result,
    create_selection_actions_description,
    create_actions_parameters_description,
    create_action_selection_json_format,
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
    "DomainKnowledgeProvider",
    "BeliefUpdatePromptConstructor",
    "ActionSelectionPromptConstructor",
    "PredictionPromptConstructor",
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
]
