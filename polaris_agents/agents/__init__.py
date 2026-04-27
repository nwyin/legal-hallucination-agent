"""
Agents module - task-agnostic agent implementations.

Agent types:
- BayesianOptimalExperimentalDesignAgent: BOED with single beliefs (θ only)
- BOEDCitationTrackerAgent: BOED without domain knowledge; task knowledge is a list of citations/quotes/holdings
"""

# Task-agnostic agents
from .boed import BayesianOptimalExperimentalDesignAgent
from .boed_citation_tracker import BOEDCitationTrackerAgent

# Base agent
from .base import Agent

__all__ = [
    "BayesianOptimalExperimentalDesignAgent",
    "BOEDCitationTrackerAgent",
    "Agent",
]

