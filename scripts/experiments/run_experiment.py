#!/usr/bin/env python3
"""
Generic Experiment Runner for Polaris Agents.

Runs any task (environment) with any agent method and collects metrics.
Examples are loaded from a JSONL dataset file.

Usage:
    # Single example - specify example_id
    python scripts/run_experiment.py data.example_id=23-477
    
    # Batch mode - run all examples
    python scripts/run_experiment.py
    
    # Batch with limit
    python scripts/run_experiment.py data.limit=5
    
    # Override method
    python scripts/run_experiment.py data.example_id=23-477 method=boed
"""

import os
import sys
import logging
import token
import hydra
from typing import Dict, Any, Type, List, Optional
from omegaconf import DictConfig, OmegaConf
import json
import random
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add the project root to the path (run_experiment.py is in scripts/experiments/)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
# Add scripts directory to path for episode_logging import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from episode_logging import (
    log_initial_state,
    log_step_header,
    log_action_basic,
    log_final_response,
    log_think_action,
    log_search_action,
    log_generic_observation,
    log_beliefs,
)

from polaris_agents.models.llm import ModelAPI
from polaris_agents.environments.base import Environment, Observation
from polaris_agents.agents.base import Agent
from polaris_agents.evaluation import (
    create_metrics_collector,
    MetricsCollector
)
from dotenv import load_dotenv

load_dotenv()

# Disable OpenTelemetry tracing (prevents 429 "Too Many Requests" from trace exporter)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

logger = logging.getLogger(__name__)

# =============================================================================
# TASK REGISTRY
# =============================================================================

TASK_NAMES = ["scotus_judgment", "multi_armed_bandit", "legal_hallucination_checker"]


def extract_hallucination_ground_truth(data: Dict[str, Any]) -> Any:
    """
    Resolve hallucination ground-truth field from known dataset variants.

    Supported aliases:
    - list_hallucinations
    - listed_hallucinations
    - list_hallucinationss (legacy typo seen in some rows)
    """
    if not data:
        return []
    for key in ("list_hallucinations", "listed_hallucinations", "list_hallucinationss"):
        if key in data and data.get(key) is not None:
            return data.get(key)
    return []


def _load_task_config(task: str) -> Dict[str, Any]:
    """Load config for a single task (lazy import to avoid Java/pyserini for tasks that don't need it)."""
    if task == "legal_hallucination_checker":
        from polaris_agents.environments.legal_hallucination_checker import HallucinationCheckerEnvironment
        from polaris_agents.prompts.environments.legal_hallucination_checker import LegalHallucinationCheckerDomainKnowledge
        return {
            "environment_class": HallucinationCheckerEnvironment,
            "domain_knowledge": LegalHallucinationCheckerDomainKnowledge(),
            "id_field": "filename",
            "task_belief_prior": (
                "You are verifying citations in a legal brief for hallucinations. "
                "θ = the set of citations and sentences that are hallucinated (fabricated, misquoted, or non-existent). "
                "You begin with no knowledge of which citations are hallucinated."
            ),
        }
    else:
        raise ValueError(f"Unknown task: {task}. Available: {TASK_NAMES}")


def get_task_registry(task: str) -> Dict[str, Dict[str, Any]]:
    """Returns task config for the given task. Loads only that task to avoid heavy deps (e.g. pyserini/Java for scotus)."""
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task: {task}. Available: {TASK_NAMES}")
    return {task: _load_task_config(task)}

# =============================================================================
# AGENT REGISTRY
# =============================================================================

def get_agent_registry() -> Dict[str, Type[Agent]]:
    """Returns the agent registry mapping method names to agent classes."""
    from polaris_agents.agents.boed import BayesianOptimalExperimentalDesignAgent
    from polaris_agents.agents.boed_citation_tracker import BOEDCitationTrackerAgent

    return {
        "boed": BayesianOptimalExperimentalDesignAgent,
        "boed_citation_tracker": BOEDCitationTrackerAgent,
    }

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_logging(log_config: Dict[str, Any]):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

def setup_experiment_logging(dataset: str, method: str, model_id: str, example_id: str, experiments_dir: str = "outputs/experiments"):
    """
    Add a file handler to write logs to experiments directory.
    
    Logs are written to: {experiments_dir}/{dataset}/{model_id}/{method}/{example_id}_{timestamp}.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{example_id}_{timestamp}.log"
    
    log_dir = os.path.join(experiments_dir, dataset, model_id, method)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Add file handler to root logger
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Experiment log: {log_filepath}")
    return log_filepath

def setup_api_keys():
    """Check and log available API keys."""
    keys = {
        "OPENROUTER_API_KEY": "OpenRouter",
        "GEMINI_API_KEY": "Gemini",
        "OPENAI_API_KEY": "OpenAI",
        "AI_SANDBOX_KEY": "Sandbox",
        "COURTLISTENER_API_KEY": "CourtListener",
    }
    for env_var, name in keys.items():
        if os.getenv(env_var):
            print(f"✓ {name} API key found")
        else:
            print(f"⚠ {env_var} not found")

def check_required_api_key(provider: str) -> bool:
    """Check if required API key is available for the provider."""
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "sandbox": "AI_SANDBOX_KEY",
    }
    required_key = key_map.get(provider.lower()) if provider else None
    if required_key and not os.getenv(required_key):
        logger.error(f"{required_key} is required but not found")
        print(f"Please set: export {required_key}='your_key'")
        return False
    return True


def resolve_agent_model_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Resolve the model config used for agent inference.

    Preferred source is `agent.model`. Top-level `model` is treated as legacy fallback.
    If both are present and differ, log a warning to avoid ambiguous config behavior.
    """
    top_level_model = OmegaConf.to_container(cfg.get('model', {}), resolve=True) if 'model' in cfg else {}

    agent_model = {}
    if 'agent' in cfg and cfg.agent is not None:
        agent_model = OmegaConf.to_container(cfg.agent.get('model', {}), resolve=True)

    if agent_model and top_level_model and agent_model != top_level_model:
        logger.warning(
            "Both `agent.model` and top-level `model` are set and differ. "
            "Using `agent.model` for agent inference; top-level `model` is ignored."
        )

    return agent_model or top_level_model

# =============================================================================
# DATA LOADING
# =============================================================================

def load_examples(dataset_path: str) -> List[Dict[str, Any]]:
    """Load examples from a JSONL dataset file."""
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def get_example_by_id(examples: List[Dict[str, Any]], example_id: str, id_field: str) -> Optional[Dict[str, Any]]:
    """Find an example by its ID field."""
    for example in examples:
        if str(example.get(id_field)) == str(example_id):
            return example
    return None

def _serialize_action_history(history: list) -> list:
    """Convert action history to JSON-serializable list."""
    if not history:
        return []
    result = []
    for step in history:
        action = step.get('action')
        if action and hasattr(action, 'action_type') and hasattr(action, 'get_input_parameters'):
            result.append({
                "action_type": action.action_type.value,
                "parameters": action.get_input_parameters(),
            })
        else:
            result.append({"action_type": "unknown", "parameters": {}})
    return result


def get_completed_examples(metrics_dir: str, dataset: str, model_id: str, method: str) -> set:
    """Get set of already completed example IDs.
    
    Looks for metrics at: {metrics_dir}/{dataset}/{model_id}/{method}/{example_id}.json
    """
    completed = set()
    method_metrics_dir = Path(metrics_dir) / dataset / model_id / method
    if method_metrics_dir.exists():
        for metrics_file in method_metrics_dir.glob("*.json"):
            # Extract example_id from filename (e.g., "23-477.json" -> "23-477")
            completed.add(metrics_file.stem)
    return completed


def clear_opinion_cache(opinion_cache_dir: str) -> None:
    """Clear the opinion cache directory (remove cached opinion JSON files) so the next example doesn't reuse them."""
    if not opinion_cache_dir or not os.path.isdir(opinion_cache_dir):
        return
    try:
        for name in os.listdir(opinion_cache_dir):
            path = os.path.join(opinion_cache_dir, name)
            if os.path.isfile(path) and name.endswith(".json"):
                os.remove(path)
    except OSError as e:
        logger.warning(f"Failed to clear opinion cache at {opinion_cache_dir}: {e}")


# =============================================================================
# PATH TEMPLATE SUBSTITUTION
# =============================================================================

def substitute_path_templates(paths_config: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute path templates with example-specific values.
    
    Supported placeholders:
        {docket} - The case docket number
        {year_prefix} - First two characters of docket (e.g., "23" from "23-477")
        {base} - The base path
    """
    docket = example.get('docket', '')
    year_prefix = docket.split('-')[0] if '-' in docket else docket[:2]
    base = paths_config.get('base', '')
    
    def substitute(value: Any) -> Any:
        if isinstance(value, str):
            return value.format(
                docket=docket,
                year_prefix=year_prefix,
                base=base
            )
        elif isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute(v) for v in value]
        return value
    
    return substitute(paths_config)

def build_environment_config(
    example: Dict[str, Any],
    paths_config: Dict[str, Any],
    search_config: Dict[str, Any],
    env_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the full environment config for an example."""
    # Substitute path templates
    paths = substitute_path_templates(paths_config, example)
    
    # Build environment config
    # Extract resolution date from resolution_entry_raw (first entry's date)
    # This is the actual date the case was resolved, NOT freeze_date (which is just when data was collected)
    resolution_date = ''
    resolution_entry_raw = example.get('resolution_entry_raw', [])
    if resolution_entry_raw and len(resolution_entry_raw) > 0:
        resolution_date = resolution_entry_raw[0].get('date', '')
    
    if not resolution_date:
        logger.warning(f"⚠️  No resolution_date found in resolution_entry_raw for case {example.get('docket', 'unknown')} - date filter will not be applied!")
    
    env = {
        # From example (JSONL)
        'case_docket': example.get('docket', ''),
        'case_title': example.get('case_title', ''),
        'resolution_date': resolution_date,
        'legal_questions': example.get('legal_questions', []),
        'legal_answers': example.get('legal_answers', []),
        'facts_of_the_case': example.get('facts_of_the_case', '') or example.get('oyez_facts_of_the_case', ''),
        
        # From paths (substituted)
        'indexes': paths.get('indexes', {}),
        'task_specific_documents_dir': paths.get('task_specific_documents_dir', ''),
        'metadocuments_dir': paths.get('metadocuments_dir', ''),
        
        # From search config
        'search': search_config,
        
        # From env config
        'max_steps': env_config.get('max_steps', 20),
    }
    
    return env

# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def create_environment(task: str, env_config: Dict[str, Any]) -> Environment:
    """Create an environment based on task name and config."""
    task_registry = get_task_registry(task)
    task_info = task_registry[task]
    env_class = task_info["environment_class"]
    
    if task == "scotus_judgment":
        search_config = env_config.get('search', {})
        if 'indexes' in env_config:
            search_config['indexes'] = env_config['indexes']
        
        return env_class(
            case_info=env_config,
            max_steps=env_config.get('max_steps', 10),
            search_config=search_config,
            embedding_model_name=search_config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B'),
            device=search_config.get('embedding_device', 'cpu'),  # Get device from search config
            task_specific_documents_dir=env_config.get('task_specific_documents_dir'),
            metadocuments_dir=env_config.get('metadocuments_dir'),
            metadata_filter_max_tokens=search_config.get('metadata_filter_max_tokens', 10000)
        )
    
    elif task == "multi_armed_bandit":
        return env_class(
            n_bandits=env_config.get('n_bandits', 3),
            thetas=env_config.get('thetas', [0.3, 0.5, 0.7]),
            max_steps=env_config.get('max_steps', 10),
            seed=env_config.get('seed'),
            search_index_path=env_config.get('search_index_path'),
            search_top_k=env_config.get('search_top_k', 3),
            enable_think=env_config.get('enable_think', True),
            enable_closed_search=env_config.get('enable_closed_search', False),  # Usually no search for bandits
            enable_web_search=env_config.get('enable_web_search', False),
        )
    elif task == "legal_hallucination_checker":
        return env_class(
            brief_info=env_config.get('brief_info', {}),
            brief_text=env_config.get('brief_text', ''),
            max_steps=env_config.get('max_steps', 30),
            search_top_k=env_config.get('search_top_k', 3),
            opinion_cache_dir=env_config.get('opinion_cache_dir'),
        )
    else:
        return env_class(
            question=env_config.get('question', ''),
            key=env_config.get('key', ''),
            max_steps=env_config.get('max_steps', 10),
        )

# =============================================================================
# AGENT FACTORY
# =============================================================================

def create_agent(
    method: str,
    task: str,
    environment: Environment,
    model_api: ModelAPI,
    model_config: Dict[str, Any],
    agent_config: Dict[str, Any]
) -> Agent:
    """Create an agent based on method name and task."""
    from polaris_agents.prompts.agents.boed import (
        BOEDBeliefUpdatePromptConstructor,
        BOEDActionSelectionPromptConstructor,
        BOEDPredictionPromptConstructor,
    )
    from polaris_agents.prompts.agents.boed_citation_tracker import (
        BOEDCitationTrackerBeliefUpdatePromptConstructor,
        BOEDCitationTrackerPredictionPromptConstructor
    )
    
    agent_registry = get_agent_registry()
    task_registry = get_task_registry(task)
    
    if method not in agent_registry:
        raise ValueError(f"Unknown method: {method}. Available: {list(agent_registry.keys())}")
    
    agent_class = agent_registry[method]
    
    # Get task-specific domain knowledge (or None for generic)
    # Set to None to use generic prompts, or provide a DomainKnowledgeProvider for task-specific prompts
    domain_knowledge = task_registry.get(task, {}).get('domain_knowledge')
    
    # Extract agent max_tokens config (must be a dict)
    agent_max_tokens_config = agent_config.get('max_tokens', {})
    if not isinstance(agent_max_tokens_config, dict):
        raise ValueError("agent.max_tokens must be a dictionary with keys: action_selection, belief_update, prediction, eig_estimate")
    
    # Get default max_tokens from action_selection (used as fallback in agents)
    max_tokens_default = agent_max_tokens_config.get('action_selection', 16000)
    
    # Common params for BOED / BOEDCitationTracker
    common_params = {
        'environment': environment,
        'model_api': model_api,
        'model_id': model_config.get('model_id'),
        'provider': model_config.get('provider'),
        'max_tokens': max_tokens_default,
        'max_tokens_config': agent_max_tokens_config,
        'temperature': model_config.get('temperature', 0.7),
        'seed': model_config.get('seed'),  # For reproducible LLM outputs
        'thinking_enabled': agent_config.get('thinking_enabled', True),
        'closed_search_enabled': agent_config.get('closed_search_enabled', True),
        'open_web_search_enabled': agent_config.get('open_web_search_enabled', False),
        'courtlistener_search_enabled': agent_config.get('courtlistener_search_enabled', False),
        'courtlistener_opinion_access_enabled': agent_config.get('courtlistener_opinion_access_enabled', False),
        # Optional overrides for belief update calls only
        'belief_update_model_id': model_config.get('belief_update_model_id'),
        'belief_update_provider': model_config.get('belief_update_provider'),
        'belief_update_temperature': model_config.get('belief_update_temperature'),
    }
    
    # Add method-specific prompt constructors with domain knowledge
    task_info = task_registry.get(task, {})
    if method == 'boed':
        common_params['belief_update_prompt_constructor'] = BOEDBeliefUpdatePromptConstructor(domain_knowledge)
        common_params['action_selection_prompt_constructor'] = BOEDActionSelectionPromptConstructor(domain_knowledge)
        common_params['prediction_prompt_constructor'] = BOEDPredictionPromptConstructor(domain_knowledge)
        if 'task_belief_prior' in task_info:
            common_params['task_belief_prior'] = task_info['task_belief_prior']
    elif method == 'boed_citation_tracker':
        common_params['belief_update_prompt_constructor'] = BOEDCitationTrackerBeliefUpdatePromptConstructor(domain_knowledge)
        common_params['action_selection_prompt_constructor'] = BOEDActionSelectionPromptConstructor(domain_knowledge)
        common_params['prediction_prompt_constructor'] = BOEDCitationTrackerPredictionPromptConstructor(domain_knowledge)

    return agent_class(**common_params)


def _create_hallucination_checker_agent(
    environment,
    model_api,
    model_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    agent_class,
):
    """Create HallucinationCheckerAgent with task-specific params."""
    from polaris_agents.models.local_llm import init_local_model
    
    brief_name = agent_config.get('brief_name', 'unknown')
    provider = model_config.get('provider', 'sandbox')
    
    llm, tokenizer = None, None
    if provider == "local":
        model_path = model_config.get('model_path', '')
        tp_size = model_config.get('tp_size', 1)
        llm, tokenizer = init_local_model(model_path=model_path, tp_size=tp_size, logger=logger)
    
    return agent_class(
        llm=llm,
        tokenizer=tokenizer,
        brief_name=brief_name,
        logger=logger,
        environment=environment,
        model_api=model_api,
        model_id=model_config.get('model_id', 'gpt-4-turbo'),
        provider=provider,
        max_tokens=model_config.get('max_tokens', 16000),
        temperature=model_config.get('temperature', 0.7),
        thinking_enabled=agent_config.get('thinking_enabled', True),
        closed_search_enabled=agent_config.get('closed_search_enabled', False),
        open_web_search_enabled=agent_config.get('open_web_search_enabled', True),
        courtlistener_search_enabled=agent_config.get('courtlistener_search_enabled', True),
        courtlistener_opinion_access_enabled=agent_config.get('courtlistener_opinion_access_enabled', True),
    )


# =============================================================================
# EPISODE RUNNER
# =============================================================================

def run_episode(
    agent: Agent,
    environment: Environment,
    metrics_collector,
    task_name: str
) -> Dict[str, Any]:
    """Run a complete episode and collect metrics."""
    def _prediction_to_list(prediction: Any) -> Optional[List[str]]:
        """Best-effort conversion of model prediction into a list of hallucinations."""
        if prediction is None:
            return None
        if isinstance(prediction, list):
            return [str(x) for x in prediction]
        if isinstance(prediction, str):
            value = prediction.strip()
            if not value:
                return []
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except (json.JSONDecodeError, TypeError):
                    pass
            return [value]
        return [str(prediction)]

    environment.reset()
    observation = environment.get_initial_observation()
    
    log_initial_state(agent, environment, observation, task_name)
    
    final_response = None
    is_correct = None
    final_accuracy = None
    predicted_hallucinations = None
    
    if getattr(environment, "max_steps", None) == 0:
        logger.info("max_steps=0 detected - skipping action selection and requesting direct prediction.")
        max_prediction_retries = 3
        final_response = None
        for attempt in range(max_prediction_retries):
            try:
                final_response, _ = agent.get_current_prediction()
            except Exception as e:
                logger.error(f"Direct prediction failed for max_steps=0 (attempt {attempt + 1}/{max_prediction_retries}): {e}")
                final_response = None
            is_valid = final_response is not None and (
                (final_response if isinstance(final_response, str) else str(final_response)).strip()
                and not (isinstance(final_response, list) and len(final_response) == 0)
            )
            if is_valid:
                break
            logger.warning(f"Direct prediction empty or invalid (attempt {attempt + 1}/{max_prediction_retries}), retrying...")
        if task_name == "legal_hallucination_checker":
            predicted_hallucinations = getattr(agent, "last_final_response_list", None)
            if predicted_hallucinations is None:
                predicted_hallucinations = _prediction_to_list(final_response)
    else:
        max_action_parse_retries = 2
        max_final_response_retries = 2
        final_response_retry_count = 0

        while not environment.is_terminated() and not environment.is_truncated():
            step_num = agent.current_step + 1
            log_step_header(step_num, observation)
            
            action = None
            for attempt_idx in range(max_action_parse_retries + 1):
                action = agent.select_action(observation)
                if action is not None:
                    break
                if attempt_idx < max_action_parse_retries:
                    logger.warning(
                        "Action parsing failed at step %s (attempt %s/%s). Retrying select_action.",
                        step_num,
                        attempt_idx + 1,
                        max_action_parse_retries + 1,
                    )

            if action is None:
                logger.error(
                    "Action parsing failed after %s attempts at step %s - agent returned None. Terminating episode.",
                    max_action_parse_retries + 1,
                    step_num,
                )
                break
            if task_name == "legal_hallucination_checker":
                time.sleep(random.uniform(1, 3))  # Polite delay to avoid rate limits
            observation = environment.step(action)
            agent.update_state(action, observation)
            
            action_type = log_action_basic(action)
            
            # Action-specific logging
            if action_type == "PROVIDE_FINAL_RESPONSE":
                final_response, is_correct, final_accuracy = log_final_response(action, observation)
                if task_name == "legal_hallucination_checker":
                    predicted_hallucinations = getattr(agent, "last_final_response_list", None)
                    if not final_response and final_response_retry_count < max_final_response_retries:
                        final_response_retry_count += 1
                        logger.warning(
                            "PROVIDE_FINAL_RESPONSE returned empty or unparseable list "
                            "(attempt %s/%s). Prompting agent to retry.",
                            final_response_retry_count,
                            max_final_response_retries,
                        )
                        environment.terminated = False
                        observation = Observation(
                            result=(
                                "Your response was empty or could not be parsed as a JSON list. "
                                "Please provide your final answer as a valid JSON array of hallucinated strings, "
                                'e.g. ["citation1", "citation2"]. If no hallucinations were found, return [].'
                            ),
                            metadata={"action_type": "PROVIDE_FINAL_RESPONSE", "error": "empty_or_unparseable"},
                        )
                        final_response = None
                        predicted_hallucinations = None
            elif action_type == "THINK":
                log_think_action(action)
            elif action_type in ["CLOSED_SEARCH", "OPEN_WEB_SEARCH", "OPEN_COURTLISTENER_SEARCH"]:
                log_search_action(action, observation, action_type)
            elif action_type == "ACCESS_COURTLISTENER_OPINION":
                opinion_id = getattr(action, "opinion_id", "")
                logger.info(f"  [ACCESS_COURTLISTENER_OPINION] opinion_id={opinion_id}")
            elif action_type == "SEARCH_LOCAL_OPINION":
                opinion_id = getattr(action, "opinion_id", "")
                search_string = getattr(action, "search_string", "")[:60]
                logger.info(f"  [SEARCH_LOCAL_OPINION] opinion_id={opinion_id} search_string={search_string}...")
            elif action_type == "READ_DOCUMENT":
                doc_id = getattr(action, "opinion_id", "")
                start = getattr(action, "start_line", 0)
                num = getattr(action, "num_lines", 0)
                logger.info(f"  [READ_DOCUMENT] opinion_id={doc_id} start_line={start} num_lines={num}")
            elif action_type == "EDIT_SCRATCHPAD":
                op = getattr(action, "operation", "")
                logger.info(f"  [EDIT_SCRATCHPAD] operation={op}")
            else:
                log_generic_observation(observation)
            
            log_beliefs(agent)
            logger.info("")
            
            metrics_collector.record_step(
                action=action,
                agent=agent,
                reward=None,
                metadata={
                    'observation_result': observation.result,
                    'observation_metadata': observation.metadata,
                    'current_beliefs': agent.get_current_beliefs() if hasattr(agent, 'get_current_beliefs') else None,
                    'step_number': agent.current_step
                }
            )
    
    current_beliefs = agent.get_current_beliefs() if hasattr(agent, 'get_current_beliefs') else None
    
    # Get accuracy from the final observation
    final_accuracy = None
    if final_response and agent.history:
        last_step = agent.history[-1]
        last_obs = last_step.get('observation')
        if last_obs and last_obs.metadata:
            final_accuracy = last_obs.metadata.get('accuracy')
    
    if predicted_hallucinations is None and task_name == "legal_hallucination_checker":
        predicted_hallucinations = getattr(agent, "last_final_response_list", None)

    # Hallucination checker: compute precision, recall, F1
    precision = recall = f1 = None
    if task_name == "legal_hallucination_checker":
        from polaris_agents.evaluation.hallucination_checker_evaluator import (
            evaluate_entry,
            compute_metrics,
        )
        ground_truth_list = environment.key if isinstance(environment.key, list) else []
        gt_found, gt_total, correct_pred, pred_total = evaluate_entry(
            ground_truth_list, predicted_hallucinations
        )
        metrics = compute_metrics(gt_found, gt_total, correct_pred, pred_total)
        precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]
        if final_accuracy is None:
            final_accuracy = f1  # Use F1 as accuracy for hallucination checker

    summary = {
        'total_steps': agent.current_step,
        'terminated': environment.is_terminated(),
        'truncated': environment.is_truncated(),
        'final_response': final_response,
        'predicted_hallucinations': predicted_hallucinations,
        'is_correct': is_correct,  # Boolean: fully correct (all questions right)
        'accuracy': final_accuracy,  # Float: ratio / F1 for hallucination checker
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_answer': environment.key,
        'question': environment.question,
        'history': agent.history,
        'model_used': agent.model_id,
        'task_beliefs': current_beliefs.get('task_beliefs') if current_beliefs else None,
        'design_beliefs': current_beliefs.get('design_beliefs') if current_beliefs else None,
    }
    
    # Log completion; for hallucination checker include P/R/F1
    if task_name == "legal_hallucination_checker" and precision is not None:
        logger.info(
            f"Episode completed - Steps: {agent.current_step}, "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}, Fully Correct: {is_correct}"
        )
    else:
        accuracy_str = f", Accuracy: {final_accuracy:.3f}" if final_accuracy is not None else ""
        logger.info(f"Episode completed - Steps: {agent.current_step}, Accuracy: {final_accuracy if final_accuracy is not None else 'N/A'}{accuracy_str}, Fully Correct: {is_correct}")
    return summary

# =============================================================================
# RESULTS LOGGING
# =============================================================================

def log_results(summary: Dict[str, Any], task: str, method: str, example_id: str = None):
    """Log execution results."""
    header = f"{task.upper()} - {method.upper()}"
    if example_id:
        header += f" - {example_id}"
    
    logger.info("\n" + "=" * 60)
    logger.info(header)
    logger.info("=" * 60)
    
    logger.info(f"Model: {summary.get('model_used', 'Unknown')}")
    logger.info(f"Total Steps: {summary['total_steps']}")
    logger.info(f"Terminated: {summary['terminated']}")
    
    if summary.get('final_response'):
        logger.info(f"\nPrediction: {summary['final_response']}")
        if task == "legal_hallucination_checker":
            p, r, f = summary.get('precision'), summary.get('recall'), summary.get('f1')
            if p is not None and r is not None and f is not None:
                logger.info(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")
            accuracy = summary.get('accuracy')
            if accuracy is not None:
                logger.info(f"Accuracy (F1): {accuracy:.3f}")
        else:
            accuracy = summary.get('accuracy')
            if accuracy is not None:
                logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}% of questions correct)")
        logger.info(f"Fully Correct: {summary.get('is_correct')}")
    
    # Log belief state if available
    if summary.get('task_beliefs') or summary.get('design_beliefs'):
        logger.info("")
        logger.info("FINAL BELIEF STATE:")
        logger.info("-" * 40)
        if summary.get('task_beliefs'):
            logger.info(f"Task Beliefs: {summary['task_beliefs'][:200]}...")
        if summary.get('design_beliefs'):
            logger.info(f"Design Beliefs: {summary['design_beliefs'][:200]}...")
    
    # Log detailed action history
    if summary.get('history'):
        logger.info("")
        logger.info("DETAILED ACTION HISTORY:")
        logger.info("-" * 40)
        for i, step in enumerate(summary['history']):
            action = step.get('action')
            obs = step.get('observation')
            if action and obs:
                action_type = action.action_type.value
                logger.info(f"Step {i+1}: {action_type}")
                logger.info(f"  Action Parameters: {str(action.get_input_parameters())}")
                
                # Special handling for PROVIDE_FINAL_RESPONSE to show prediction results
                if action_type == "PROVIDE_FINAL_RESPONSE":
                    # Get prediction from action parameters
                    prediction = action.get_input_parameters().get('response', 'N/A')
                    
                    # Get accuracy and is_correct from observation metadata, fall back to summary
                    accuracy_val = None
                    is_correct_val = None
                    if obs and hasattr(obs, 'metadata') and obs.metadata:
                        accuracy_val = obs.metadata.get('accuracy')
                        is_correct_val = obs.metadata.get('is_correct')
                    # Fall back to summary values
                    if accuracy_val is None:
                        accuracy_val = summary.get('accuracy')
                    if is_correct_val is None:
                        is_correct_val = summary.get('is_correct')
                    
                    # Log prediction and metrics
                    logger.info(f"  Prediction: {prediction}")
                    if task == "legal_hallucination_checker":
                        p, r, f = summary.get('precision'), summary.get('recall'), summary.get('f1')
                        if p is not None and r is not None and f is not None:
                            logger.info(f"  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")
                        if accuracy_val is not None:
                            logger.info(f"  Accuracy (F1): {accuracy_val:.3f}")
                    else:
                        accuracy_str = f"{accuracy_val:.3f}" if accuracy_val is not None else "N/A"
                        logger.info(f"  Accuracy: {accuracy_str}")
                    logger.info(f"  Fully Correct: {is_correct_val}")
                    logger.info("")
                    continue
                
                # Special handling for CLOSED_SEARCH and OPEN_WEB_SEARCH to show detailed results
                if action_type in ['CLOSED_SEARCH', 'OPEN_WEB_SEARCH']:
                    metadata = obs.metadata or {}
                    num_results = metadata.get('num_results', 0)
                    search_results = metadata.get('search_results', [])
                    query = metadata.get('query', action.get_input_parameters().get('query', 'N/A'))
                    search_type = metadata.get('search_type', action.get_input_parameters().get('search_type', 'N/A'))
                    
                    logger.info(f"  Query: {query}")
                    logger.info(f"  Search Type: {search_type}")
                    logger.info(f"  Number of Results: {num_results}")
                    
                    if num_results > 0 and search_results:
                        # Show details for each result (up to 5)
                        for i, result in enumerate(search_results[:5]):
                            # Handle both dict and object results
                            if isinstance(result, dict):
                                title = result.get('title', 'Unknown')
                                result_id = result.get('result_id', str(i+1))
                                # Score is in metadata dict
                                if 'metadata' in result and isinstance(result['metadata'], dict):
                                    score = result['metadata'].get('score', 'N/A')
                                else:
                                    score = 'N/A'
                            else:
                                # SearchResult object
                                title = getattr(result, 'title', 'Unknown')
                                result_id = getattr(result, 'result_id', str(i+1))
                                score = result.metadata.get('score', 'N/A') if hasattr(result, 'metadata') and result.metadata else 'N/A'
                            
                            # Truncate title for display
                            title_display = title[:100] + '...' if len(title) > 100 else title
                            logger.info(f"    Result {i+1} (ID: {result_id}): {title_display} (score: {score})")
                        
                        if num_results > 5:
                            logger.info(f"    ... and {num_results - 5} more result(s)")
                    elif num_results == 0:
                        logger.info(f"  No results found for query: '{query}'")
                    else:
                        # Fallback: parse from result text
                        result_text = obs.result
                        logger.info(f"  Result: {result_text[:500]}...")
                else:
                    # For other actions, show truncated result safely
                    preview = obs.result
                    if isinstance(preview, dict):
                        preview = json.dumps(preview, default=str)
                    elif preview is None:
                        preview = "None"
                    else:
                        preview = str(preview)
                    logger.info(f"  Result: {preview[:150]}...")
                
                logger.info("")
    
    logger.info("=" * 60)

# =============================================================================
# SINGLE EXAMPLE RUNNER
# =============================================================================

def run_single_example(
    task: str,
    method: str,
    example: Dict[str, Any],
    paths_config: Dict[str, Any],
    search_config: Dict[str, Any],
    env_settings: Dict[str, Any],
    model_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    eval_model_config: Dict[str, Any],
    eval_settings: Dict[str, Any],
    output_dir: str,
    metrics_dir: str,
    example_id: str,
    dataset: str,
    experiment_logs: bool = False,
    model_api: ModelAPI = None,
) -> Dict[str, Any]:
    """Run a single example and return results."""
    if model_api is None:
        model_api = ModelAPI()
    
    model_id = model_config.get('model_id', 'unknown')
    
    # Set up experiment logging if enabled
    if experiment_logs:
        setup_experiment_logging(dataset, method, model_id, example_id)
    
    # Build environment config from example + templates
    if task == "multi_armed_bandit":
        env_config = example.copy()
        env_config['max_steps'] = env_settings.get('max_steps', example.get('max_steps', 10))
    elif task == "legal_hallucination_checker":
        opinion_cache_base = paths_config.get('opinion_cache_dir') or os.path.join(output_dir, 'opinion_cache')
        _model_id = model_config.get('model_id', 'unknown')
        _max_steps = env_settings.get('max_steps', 30)
        _method_with_steps = f"{method}_steps{_max_steps}"
        safe_example_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(example_id))
        opinion_cache = os.path.join(opinion_cache_base, _model_id, _method_with_steps, safe_example_id)
        gt = extract_hallucination_ground_truth(example)
        env_config = {
            'brief_info': {
                'list_hallucinations': gt,
                'list_hallucination_types': example.get('list_hallucination_types', []),
            },
            'brief_text': example.get('text', ''),
            'max_steps': env_settings.get('max_steps', 30),
            'search_top_k': search_config.get('top_k', 3),
            'opinion_cache_dir': opinion_cache,
        }
        agent_config = dict(agent_config)
        agent_config['brief_name'] = example.get('filename', example_id)
    else:
        env_config = build_environment_config(example, paths_config, search_config, env_settings)
    
    # Create environment
    environment = create_environment(task, env_config)
    
    # Create agent
    agent = create_agent(
        method=method,
        task=task,
        environment=environment,
        model_api=model_api,
        model_config=model_config,
        agent_config=agent_config
    )
    
    # Create metrics collector
    has_beliefs = method in ['ids_oed', 'boed', 'boed_citation_tracker']
    model_id = model_config.get('model_id', 'unknown')

    # Metrics go to: metrics/{dataset}/{model_id}/{method}_steps{max_steps}/
    max_steps = env_config.get('max_steps')
    method_with_steps = f"{method}_steps{max_steps}" if max_steps is not None else method
    example_metrics_dir = os.path.join(metrics_dir, dataset, model_id, method_with_steps)
    
    task_registry = get_task_registry(task)
    
    # Check evaluation settings - 'enabled' is master switch for all evaluation
    eval_enabled = eval_settings.get('enabled', True)
    # EIG estimation can be disabled separately (extra LLM call per step when on)
    enable_eig = eval_settings.get('enable_eig_estimation', eval_enabled)
    # Action classification (task vs design focus scoring) can be disabled separately
    enable_action_classification = eval_settings.get('enable_action_classification', eval_enabled)
    # Belief evolution tracking (complexity analysis per step; extra LLM call)
    enable_belief_evolution = eval_settings.get('enable_belief_evolution_tracking', eval_enabled)
    
    metrics_collector = create_metrics_collector(
        model_api=model_api,
        model_name=eval_model_config.get('model_id', model_config.get('model_id')),
        provider=eval_model_config.get('provider', model_config.get('provider')),
        max_tokens=eval_model_config.get('max_tokens', 10000),
        save_dir=example_metrics_dir,
        enable_action_classification=enable_action_classification,
        enable_task_performance_tracking=eval_enabled,
        enable_eig_estimation=enable_eig,
        enable_belief_evolution_tracking=has_beliefs and enable_belief_evolution,
        domain_knowledge=task_registry.get(task, {}).get('domain_knowledge')
    )
    
    # Start metrics collection
    metrics_collector.start_episode(
        episode_id=example_id,
        task_name=task,
        agent_type=agent.__class__.__name__,
        environment_info=getattr(environment, 'get_case_info', lambda: {})(),
        method=method,
        model_id=model_config.get('model_id')
    )
    
    # Set ground truth (pass environment for consistent is_correct() evaluation)
    if task == "legal_hallucination_checker":
        ground_truth = env_config.get('list_hallucinations') or env_config.get('brief_info', {}).get('list_hallucinations')
    else:
        ground_truth = env_config.get('legal_answers')
    if ground_truth is not None:
        metrics_collector.set_task_performance_ground_truth(ground_truth, environment)
    
    # Run episode
    summary = run_episode(agent, environment, metrics_collector, task)
    
    # Evaluate final prediction once (no per-step prediction evaluation)
    metrics_collector.record_final_prediction(agent, environment)
    
    # End metrics collection; for legal_hallucination_checker, embed ground truth + prediction + raw response for evaluation
    extra_data = None
    if task == "legal_hallucination_checker":
        extra_data = {
            "list_hallucinations": ground_truth or [],
            "predicted_hallucinations": summary.get("predicted_hallucinations"),
            "final_response_raw": summary.get("final_response"),  # actual agent response string before parsing
        }
    if task == "legal_hallucination_checker" and extra_data.get("final_response_raw") is None:
        metrics_filepath = None
    else:
        metrics_filepath = metrics_collector.end_episode(extra_data=extra_data)
    summary['metrics_filepath'] = metrics_filepath
    summary['episode_id'] = example_id
    summary['example_id'] = example_id
    
    # Log results
    log_results(summary, task, method, example_id)

    return summary

# =============================================================================
# MAIN
# =============================================================================

@hydra.main(version_base=None, config_path="../../configs", config_name="scotus_judgment")
def main(cfg: DictConfig):
    """
    Main entry point for running experiments.
    
    Config should specify:
        - task: Task name
        - method: Agent method
        - data.dataset_path: Path to JSONL dataset
        - data.example_id: (Optional) Specific example to run
        - paths: Path templates
        - model/agent/search: Settings
    """
    log_config = OmegaConf.to_container(cfg.get('logging', {}), resolve=True)
    setup_logging(log_config)
    setup_api_keys()
    
    task = cfg.get('task', 'legal_hallucination_checker')
    method = cfg.get('method', 'boed_citation_tracker')
    # Dataset names the result folder (metrics/plots/output); defaults to task for backward compatibility
    dataset = cfg.get('dataset') or task
    logger.info(f"Starting experiment: task={task}, method={method}, dataset={dataset}")
    
    # Convert configs to dicts (some sections are optional)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    paths_config = OmegaConf.to_container(cfg.paths, resolve=True) if 'paths' in cfg else {}
    search_config = OmegaConf.to_container(cfg.search, resolve=True) if 'search' in cfg else {}
    env_settings = OmegaConf.to_container(cfg.environment, resolve=True)
    agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
    model_config = resolve_agent_model_config(cfg)
    eval_model_config = OmegaConf.to_container(cfg.get('evaluation_model', model_config), resolve=True)
    eval_settings = OmegaConf.to_container(cfg.get('evaluation', {}), resolve=True)
    
    # Check API key
    if not check_required_api_key(model_config.get('provider')):
        return
    output_dir = cfg.get('output_dir', 'outputs')
    metrics_dir = cfg.get('metrics_dir', 'metrics')

    if cfg.get('test_run', False):
        logger.info("TEST RUN mode enabled")
        metrics_dir = os.path.join(metrics_dir, 'test')
        output_dir = os.path.join(output_dir, 'test')
    
    # Load examples from dataset (or create synthetic example for config-based tasks)
    dataset_path = data_config.get('dataset_path')
    
    # Some tasks (like multi_armed_bandit) don't need a JSONL dataset
    # They get their config directly from the config file
    if task == "multi_armed_bandit":
        # Create a synthetic example from config
        example_id = data_config.get('example_id', 'bandit_run')
        examples = [{
            'id': example_id,
            'n_bandits': env_settings.get('n_bandits', 5),
            'thetas': env_settings.get('thetas', [0.1, 0.2, 0.3, 0.4, 0.5]),
            'max_steps': env_settings.get('max_steps', 10),
            'seed': env_settings.get('seed'),
            'search_index_path': env_settings.get('search_index_path'),
            'enable_think': env_settings.get('enable_think', True),
            'enable_closed_search': env_settings.get('enable_closed_search', False),
            'enable_web_search': env_settings.get('enable_web_search', False),
        }]
        logger.info(f"Multi-armed bandit config: n_bandits={examples[0]['n_bandits']}, thetas={examples[0]['thetas']}")
    elif not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    else:
        logger.info(f"Loading examples from: {dataset_path}")
        examples = load_examples(dataset_path)
        logger.info(f"Loaded {len(examples)} examples")
    
    # Get ID field for this task
    task_registry = get_task_registry(task)
    id_field = data_config.get('id_field') or task_registry.get(task, {}).get('id_field', 'id')
    
    # Filter to specific example if requested
    example_id = data_config.get('example_id')
    if example_id:
        example = get_example_by_id(examples, example_id, id_field)
        if not example:
            logger.error(f"Example not found: {example_id}")
            return
        examples = [example]
        logger.info(f"Running single example: {example_id}")
    else:
        # Batch mode
        limit = data_config.get('limit')
        if limit:
            examples = examples[:limit]
            logger.info(f"Limited to {limit} examples")

        # Skip completed
        if data_config.get('skip_completed', True):
            model_id = model_config.get('model_id', 'unknown')
            max_steps_val = env_settings.get('max_steps')
            method_key = f"{method}_steps{max_steps_val}" if max_steps_val is not None else method
            completed = get_completed_examples(metrics_dir, dataset, model_id, method_key)
            original_count = len(examples)
            examples = [ex for ex in examples if str(ex.get(id_field)) not in completed]
            logger.info(f"Skipping {original_count - len(examples)} completed examples")

        # Shard: partition remaining examples after skip_completed for even load distribution.
        # All shards must be launched at the same time so they see the same remaining list.
        # Usage: data.shard=[shard_index],[num_shards]  e.g. data.shard=0,4
        shard = data_config.get('shard')
        if shard:
            if isinstance(shard, (list, tuple)):
                shard_index, num_shards = int(shard[0]), int(shard[1])
            else:
                shard_index, num_shards = [int(x) for x in str(shard).split(',')]
            examples = examples[shard_index::num_shards]
            logger.info(f"Shard {shard_index}/{num_shards}: {len(examples)} examples")
    
    if not examples:
        logger.info("No examples to run!")
        return
    
    # Create shared model API
    model_api = ModelAPI()
    
    # Run examples
    results = []

    for example in tqdm(examples, desc=f"Running {task}/{method}"):
        ex_id = str(example.get(id_field, f"example_{len(results)}"))

        try:
            summary = run_single_example(
                task=task,
                method=method,
                example=example,
                paths_config=paths_config,
                search_config=search_config,
                env_settings=env_settings,
                model_config=model_config,
                agent_config=agent_config,
                eval_model_config=eval_model_config,
                eval_settings=eval_settings,
                output_dir=output_dir,
                metrics_dir=metrics_dir,
                example_id=ex_id,
                dataset=dataset,
                experiment_logs=log_config.get('experiment_logs', False),
                model_api=model_api,
            )
            results.append(summary)

        except Exception as e:
            logger.error(f"Failed on example {ex_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'example_id': ex_id,
                'error': str(e),
                'is_correct': None,
            })
        finally:
            # Clear this episode's opinion cache after each data point to avoid accumulating disk/memory use.
            # Each episode uses its own subdirectory so parallel agents don't clear each other's cached opinions.
            if task == "legal_hallucination_checker":
                opinion_cache_dir = paths_config.get('opinion_cache_dir') or os.path.join(output_dir, 'opinion_cache')
                _model_id = model_config.get('model_id', 'unknown')
                _max_steps = env_settings.get('max_steps', 30)
                _method_with_steps = f"{method}_steps{_max_steps}"
                safe_example_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(ex_id))
                clear_opinion_cache(os.path.join(opinion_cache_dir, _model_id, _method_with_steps, safe_example_id))
    
    # Log batch summary
    if len(results) > 1:
        if task == "legal_hallucination_checker":
            from polaris_agents.evaluation.hallucination_checker_evaluator import aggregate_metrics, evaluate_hallucination_entry
            entries = [
                {"list_hallucinations": r.get("true_answer") or [], "predicted_hallucinations": r.get("predicted_hallucinations")}
                for r in results
            ]
            per_entry = [evaluate_hallucination_entry(e) for e in entries]
            agg = aggregate_metrics(per_entry)
            logger.info(f"\n{'='*60}")
            logger.info(
                f"BATCH COMPLETE (legal_hallucination_checker): "
                f"P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f}"
            )
            logger.info(
                f"  ({agg['correct_predictions']:.0f}/{agg['total_predictions']:.0f} pred matched, "
                f"{agg['ground_truth_found']:.0f}/{agg['total_ground_truth']:.0f} GT found)"
            )
            logger.info(f"{'='*60}")
        else:
            correct = sum(1 for r in results if r.get('is_correct') == True)
            total = sum(1 for r in results if r.get('is_correct') is not None)
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH COMPLETE: {correct}/{total} correct ({100*correct/total:.1f}%)" if total > 0 else "BATCH COMPLETE: No predictions made")
            logger.info(f"{'='*60}")
        
        # Save batch results
        batch_results_path = os.path.join(output_dir, dataset, method, "batch_results.json")
        os.makedirs(os.path.dirname(batch_results_path), exist_ok=True)
        serializable_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != 'history'}
            serializable_results.append(sr)
        batch_data = {"results": serializable_results}
        if task == "legal_hallucination_checker" and len(results) > 1:
            from polaris_agents.evaluation.hallucination_checker_evaluator import aggregate_metrics, evaluate_hallucination_entry
            entries = [
                {"list_hallucinations": r.get("true_answer") or [], "predicted_hallucinations": r.get("predicted_hallucinations")}
                for r in results
            ]
            agg = aggregate_metrics([evaluate_hallucination_entry(e) for e in entries])
            batch_data["aggregate_metrics"] = agg
        with open(batch_results_path, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        logger.info(f"Batch results saved to: {batch_results_path}")


if __name__ == "__main__":
    main()
