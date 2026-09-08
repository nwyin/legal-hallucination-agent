"""
BOED Citation Tracker Agent: BOED without domain knowledge; task knowledge in words.

Task knowledge is a list of citations, quotes, and holdings from the brief—described
in words (natural language), not a data structure. The belief-update instructions
tell the model to keep a full list in words and never remove items, only add or
update status, so the agent is less likely to forget previously checked citations.
"""

import logging
from typing import Optional, Dict, Any

from ..environments.base import Environment, Observation
from ..llm import ModelAPI
from ..prompts import (
    BeliefUpdatePromptConstructor,
    ActionSelectionPromptConstructor,
    PredictionPromptConstructor,
    BOEDActionSelectionPromptConstructor,
    BOEDCitationTrackerBeliefUpdatePromptConstructor,
    BOEDCitationTrackerPredictionPromptConstructor,
)

from .boed import BayesianOptimalExperimentalDesignAgent

logger = logging.getLogger(__name__)

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
        