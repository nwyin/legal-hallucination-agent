"""
Polaris Agents - A framework for strategic exploration using various LLM providers.

This package provides tools for building AI agents that can perform strategic exploration
tasks, including legal research, data analysis, and decision-making processes.
"""

__version__ = "0.1.0"
__author__ = "Strategic Exploration Team"
__email__ = "your-email@example.com"

# Import main components for easy access
from .agents.boed import BayesianOptimalExperimentalDesignAgent
from .agents.boed_citation_tracker import BOEDCitationTrackerAgent
from .actions import Action, ActionType
from .llm import ModelAPI
from .environments.base import Environment, Observation

# Import CourtListener search functionality
from .agents.action.courtlistener_search.main import (
    search_courtlistener,
    search_opinions,
    search_cases,
    search_dockets,
    search_filings,
    search_judges,
    search_oral_arguments,
    execute_courtlistener_search,
)

__all__ = [
    # Core agents
    "BayesianOptimalExperimentalDesignAgent",
    "BOEDCitationTrackerAgent",
    "Action",
    "ActionType",
    "ModelAPI",

    # Environments
    "Environment",
    "Observation",

    # CourtListener search functions
    "search_courtlistener",
    "search_opinions",
    "search_cases",
    "search_dockets",
    "search_filings",
    "search_judges",
    "search_oral_arguments",
    "execute_courtlistener_search",

    # Version info
    "__version__",
    "__author__",
    "__email__",
]