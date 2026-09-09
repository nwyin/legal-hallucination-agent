#!/usr/bin/env python3
"""
Citation benchmark runner for the legal hallucination checker task.

Loads examples from a JSONL dataset, runs a citation-focused agent, and writes
episode metrics.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .tracing import langfuse, observe, propagate_attributes, flush_traces
from .agent import Agent
from .environment import Environment, HallucinationCheckerEnvironment, Observation
from .llm import ModelAPI
from .prompts import LegalHallucinationCheckerDomainKnowledge
from .evaluation import aggregate_metrics, compute_metrics, evaluate_entry, evaluate_hallucination_entry, extract_ground_truth
from .recording import (
    MetricsCollector,
    log_initial_state,
    log_step_header,
    log_action_basic,
    log_final_response,
    log_think_action,
    log_search_action,
    log_generic_observation,
    log_beliefs,
)

load_dotenv()

logger = logging.getLogger(__name__)

TASK_NAME = "legal_hallucination_checker"
TASK_ID_FIELD = "filename"
TASK_BELIEF_PRIOR = (
    "You are verifying citations in a legal brief for hallucinations. "
    "θ = the set of citations and sentences that are hallucinated (fabricated, misquoted, or non-existent). "
    "You begin with no knowledge of which citations are hallucinated."
)
TASK_DOMAIN_KNOWLEDGE = LegalHallucinationCheckerDomainKnowledge()


# =============================================================================
# AGENT REGISTRY
# =============================================================================

def get_agent_registry() -> Dict[str, Type[Agent]]:
    from .agent import BayesianOptimalExperimentalDesignAgent, BOEDCitationTrackerAgent

    return {
        "boed": BayesianOptimalExperimentalDesignAgent,
        "boed_citation_tracker": BOEDCitationTrackerAgent,
    }

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_logging(log_config: Dict[str, Any]):
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
    keys = {
        "OPENROUTER_API_KEY": "OpenRouter",
        "COURTLISTENER_API_KEY": "CourtListener",
        "SERPAPI_API_KEY": "SerpAPI",
    }
    for env_var, name in keys.items():
        if os.getenv(env_var):
            print(f"✓ {name} API key found")
        else:
            print(f"⚠ {env_var} not found")

def check_required_api_keys(agent_config: Dict[str, Any]) -> bool:
    required = ["OPENROUTER_API_KEY"]
    # Without this key every OPEN_WEB_SEARCH would fail; refuse to run rather than
    # silently benchmark an agent that cannot use one of the paper's eight actions.
    if agent_config.get("open_web_search_enabled", False):
        required.append("SERPAPI_API_KEY")
    missing = [key for key in required if not os.getenv(key)]
    for key in missing:
        logger.error(f"{key} is required but not found")
        print(f"Please set: export {key}='your_key'")
    return not missing


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
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def get_example_by_id(examples: List[Dict[str, Any]], example_id: str, id_field: str) -> Optional[Dict[str, Any]]:
    for example in examples:
        if str(example.get(id_field)) == str(example_id):
            return example
    return None

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
# ENVIRONMENT FACTORY
# =============================================================================

def create_environment(env_config: Dict[str, Any]) -> Environment:
    return HallucinationCheckerEnvironment(
        brief_info=env_config.get('brief_info', {}),
        brief_text=env_config.get('brief_text', ''),
        max_steps=env_config.get('max_steps', 30),
        search_top_k=env_config.get('search_top_k', 3),
        opinion_cache_dir=env_config.get('opinion_cache_dir'),
    )

# =============================================================================
# AGENT FACTORY
# =============================================================================

def create_agent(
    method: str,
    environment: Environment,
    model_api: ModelAPI,
    model_config: Dict[str, Any],
    agent_config: Dict[str, Any]
) -> Agent:
    """Create an agent based on method name."""
    from .prompts import (
        BOEDBeliefUpdatePromptConstructor,
        BOEDActionSelectionPromptConstructor,
        BOEDPredictionPromptConstructor,
        BOEDCitationTrackerBeliefUpdatePromptConstructor,
        BOEDCitationTrackerPredictionPromptConstructor,
    )
    
    agent_registry = get_agent_registry()
    
    if method not in agent_registry:
        raise ValueError(f"Unknown method: {method}. Available: {list(agent_registry.keys())}")
    
    agent_class = agent_registry[method]
    
    domain_knowledge = TASK_DOMAIN_KNOWLEDGE
    
    # Extract agent max_tokens config (must be a dict)
    agent_max_tokens_config = agent_config.get('max_tokens', {})
    if not isinstance(agent_max_tokens_config, dict):
        raise ValueError("agent.max_tokens must be a dictionary with keys: action_selection, belief_update, prediction")
    
    # Get default max_tokens from action_selection (used as fallback in agents)
    max_tokens_default = agent_max_tokens_config.get('action_selection', 16000)
    
    # Common params for BOED / BOEDCitationTracker
    common_params = {
        'environment': environment,
        'model_api': model_api,
        'model_id': model_config.get('model_id'),
        'max_tokens': max_tokens_default,
        'max_tokens_config': agent_max_tokens_config,
        'temperature': model_config.get('temperature', 0.7),
        'seed': model_config.get('seed'),  # For reproducible LLM outputs
        'thinking_enabled': agent_config.get('thinking_enabled', True),
        'open_web_search_enabled': agent_config.get('open_web_search_enabled', False),
        'courtlistener_search_enabled': agent_config.get('courtlistener_search_enabled', False),
        'courtlistener_opinion_access_enabled': agent_config.get('courtlistener_opinion_access_enabled', False),
        # Optional overrides for belief update calls only
        'belief_update_model_id': model_config.get('belief_update_model_id'),
        'belief_update_temperature': model_config.get('belief_update_temperature'),
    }
    
    # Add method-specific prompt constructors with domain knowledge
    if method == 'boed':
        common_params['belief_update_prompt_constructor'] = BOEDBeliefUpdatePromptConstructor(domain_knowledge)
        common_params['action_selection_prompt_constructor'] = BOEDActionSelectionPromptConstructor(domain_knowledge)
        common_params['prediction_prompt_constructor'] = BOEDPredictionPromptConstructor(domain_knowledge)
        common_params['task_belief_prior'] = TASK_BELIEF_PRIOR
    elif method == 'boed_citation_tracker':
        common_params['belief_update_prompt_constructor'] = BOEDCitationTrackerBeliefUpdatePromptConstructor(domain_knowledge)
        common_params['action_selection_prompt_constructor'] = BOEDActionSelectionPromptConstructor(domain_knowledge)
        common_params['prediction_prompt_constructor'] = BOEDCitationTrackerPredictionPromptConstructor(domain_knowledge)

    return agent_class(**common_params)


# =============================================================================
# EPISODE RUNNER
# =============================================================================

def run_episode(
    agent: Agent,
    environment: Environment,
    metrics_collector,
) -> Dict[str, Any]:
    """Run a complete episode and collect metrics."""
    def _prediction_to_list(prediction: Any) -> Optional[List[str]]:
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
    
    log_initial_state(agent, environment, observation, TASK_NAME)
    
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
            time.sleep(random.uniform(1, 3))  # Polite delay to avoid rate limits
            observation = environment.step(action)
            agent.update_state(action, observation)
            
            action_type = log_action_basic(action)
            
            # Action-specific logging
            if action_type == "PROVIDE_FINAL_RESPONSE":
                final_response, is_correct, final_accuracy = log_final_response(action, observation)
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
            elif action_type in ["OPEN_WEB_SEARCH", "OPEN_COURTLISTENER_SEARCH"]:
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
    
    if predicted_hallucinations is None:
        predicted_hallucinations = getattr(agent, "last_final_response_list", None)

    # evaluate_entry accepts both a list of spans and a {span: type} dict.
    gt_found, gt_total, correct_pred, pred_total = evaluate_entry(
        environment.key, predicted_hallucinations
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
    if precision is not None:
        logger.info(
            f"Episode completed - Steps: {agent.current_step}, "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}, Fully Correct: {is_correct}"
        )
    return summary

# =============================================================================
# RESULTS LOGGING
# =============================================================================

def log_results(summary: Dict[str, Any], method: str, example_id: str = None):
    header = f"{TASK_NAME.upper()} - {method.upper()}"
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
        p, r, f = summary.get('precision'), summary.get('recall'), summary.get('f1')
        if p is not None and r is not None and f is not None:
            logger.info(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")
        accuracy = summary.get('accuracy')
        if accuracy is not None:
            logger.info(f"Accuracy (F1): {accuracy:.3f}")
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
                    p, r, f = summary.get('precision'), summary.get('recall'), summary.get('f1')
                    if p is not None and r is not None and f is not None:
                        logger.info(f"  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")
                    if accuracy_val is not None:
                        logger.info(f"  Accuracy (F1): {accuracy_val:.3f}")
                    logger.info(f"  Fully Correct: {is_correct_val}")
                    logger.info("")
                    continue
                
                # Special handling for OPEN_WEB_SEARCH to show detailed results
                if action_type == 'OPEN_WEB_SEARCH':
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

@observe(name="verify-legal-brief", as_type="agent", capture_input=False, capture_output=False)
def run_single_example(
    method: str,
    example: Dict[str, Any],
    paths_config: Dict[str, Any],
    search_config: Dict[str, Any],
    env_settings: Dict[str, Any],
    model_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    output_dir: str,
    metrics_dir: str,
    example_id: str,
    dataset: str,
    experiment_logs: bool = False,
    model_api: ModelAPI = None,
) -> Dict[str, Any]:
    """Run a single example and return results."""
    langfuse.update_current_span(
        input=example.get("text", ""),
        metadata={"example_id": example_id, "dataset": dataset, "method": method,
                  "model": model_config.get("model_id"),
                  "max_steps": env_settings.get("max_steps", 30)},
    )
    if model_api is None:
        model_api = ModelAPI()
    
    model_id = model_config.get('model_id', 'unknown')
    
    # Set up experiment logging if enabled
    if experiment_logs:
        setup_experiment_logging(dataset, method, model_id, example_id)
    
    # Build environment config from example
    opinion_cache_base = paths_config.get('opinion_cache_dir') or os.path.join(output_dir, 'opinion_cache')
    _model_id = model_config.get('model_id', 'unknown')
    _max_steps = env_settings.get('max_steps', 30)
    _method_with_steps = f"{method}_steps{_max_steps}"
    safe_example_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(example_id))
    opinion_cache = os.path.join(opinion_cache_base, _model_id, _method_with_steps, safe_example_id)
    gt = extract_ground_truth(example)
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
    
    # Create environment
    environment = create_environment(env_config)
    
    # Create agent
    agent = create_agent(
        method=method,
        environment=environment,
        model_api=model_api,
        model_config=model_config,
        agent_config=agent_config
    )
    
    # Create metrics collector
    model_id = model_config.get('model_id', 'unknown')

    # Metrics go to: metrics/{dataset}/{model_id}/{method}_steps{max_steps}/
    max_steps = env_config.get('max_steps')
    method_with_steps = f"{method}_steps{max_steps}" if max_steps is not None else method
    example_metrics_dir = os.path.join(metrics_dir, dataset, model_id, method_with_steps)
    
    metrics_collector = MetricsCollector(save_dir=example_metrics_dir)

    # Start metrics collection
    metrics_collector.start_episode(
        episode_id=example_id,
        task_name=TASK_NAME,
        agent_type=agent.__class__.__name__,
        environment_info=getattr(environment, 'get_case_info', lambda: {})(),
        method=method,
        model_id=model_config.get('model_id')
    )
    
    ground_truth = env_config['brief_info']['list_hallucinations']

    with propagate_attributes(
        tags=["legal-hallucination-checker", method],
        metadata={"dataset": dataset, "example_id": example_id, "method": method},
    ):
        summary = run_episode(agent, environment, metrics_collector)

    # Save ground truth, prediction, raw response, and scores alongside the trajectory.
    predicted = summary.get("predicted_hallucinations")
    extra_data = {
        "list_hallucinations": ground_truth or [],
        "predicted_hallucinations": predicted,
        "final_response_raw": summary.get("final_response"),  # agent response string before parsing
        **compute_metrics(*evaluate_entry(ground_truth, predicted)),
    }
    if summary.get("final_response") is None:
        metrics_filepath = None
    else:
        metrics_filepath = metrics_collector.end_episode(extra_data=extra_data)
    summary['metrics_filepath'] = metrics_filepath
    summary['episode_id'] = example_id
    summary['example_id'] = example_id
    langfuse.update_current_span(
        output=summary.get("final_response"),
        metadata={"precision": summary.get("precision"), "recall": summary.get("recall"),
                  "f1": summary.get("f1"), "total_steps": summary.get("total_steps")},
        level="ERROR" if summary.get("final_response") is None else "DEFAULT",
        status_message="No final response" if summary.get("final_response") is None else None,
    )
    if langfuse.get_current_trace_id():
        summary["langfuse_trace_id"] = langfuse.get_current_trace_id()
    
    # Log results
    log_results(summary, method, example_id)

    return summary

# =============================================================================
# MAIN
# =============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="legal_hallucination_checker_gpt")
def main(cfg: DictConfig):
    """
    Main entry point for running experiments.
    
    Config should specify:
        - method: Agent method
        - data.dataset_path: Path to JSONL dataset
        - data.example_id: (Optional) Specific example to run
        - model/agent/search: Settings
    """
    log_config = OmegaConf.to_container(cfg.get('logging', {}), resolve=True)
    setup_logging(log_config)
    setup_api_keys()
    
    method = cfg.get('method', 'boed_citation_tracker')
    requested_task = cfg.get('task')
    if requested_task is not None and requested_task != TASK_NAME:
        raise ValueError(
            f"Unsupported task '{requested_task}'. This runner only supports '{TASK_NAME}'. "
            "Set task to 'legal_hallucination_checker' or omit task in the config."
        )
    # Dataset names the result folder (metrics/plots/output); defaults to TASK_NAME for consistency
    dataset = cfg.get('dataset') or TASK_NAME
    logger.info(f"Starting experiment: task={TASK_NAME}, method={method}, dataset={dataset}")
    
    # Convert configs to dicts (some sections are optional)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    paths_config = OmegaConf.to_container(cfg.paths, resolve=True) if 'paths' in cfg else {}
    search_config = OmegaConf.to_container(cfg.search, resolve=True) if 'search' in cfg else {}
    env_settings = OmegaConf.to_container(cfg.environment, resolve=True)
    agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
    model_config = resolve_agent_model_config(cfg)

    # Check API key
    if not check_required_api_keys(agent_config):
        return
    output_dir = cfg.get('output_dir', 'outputs')
    metrics_dir = cfg.get('metrics_dir', 'metrics')

    if cfg.get('test_run', False):
        logger.info("TEST RUN mode enabled")
        metrics_dir = os.path.join(metrics_dir, 'test')
        output_dir = os.path.join(output_dir, 'test')
    
    # Load examples from dataset
    dataset_path = data_config.get('dataset_path')
    if not dataset_path or not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    else:
        logger.info(f"Loading examples from: {dataset_path}")
        examples = load_examples(dataset_path)
        logger.info(f"Loaded {len(examples)} examples")
    
    # Get ID field for this task
    id_field = data_config.get('id_field') or TASK_ID_FIELD
    
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

    for example in tqdm(examples, desc=f"Running {TASK_NAME}/{method}"):
        ex_id = str(example.get(id_field, f"example_{len(results)}"))

        try:
            summary = run_single_example(
                method=method,
                example=example,
                paths_config=paths_config,
                search_config=search_config,
                env_settings=env_settings,
                model_config=model_config,
                agent_config=agent_config,
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
            opinion_cache_dir = paths_config.get('opinion_cache_dir') or os.path.join(output_dir, 'opinion_cache')
            _model_id = model_config.get('model_id', 'unknown')
            _max_steps = env_settings.get('max_steps', 30)
            _method_with_steps = f"{method}_steps{_max_steps}"
            safe_example_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(ex_id))
            clear_opinion_cache(os.path.join(opinion_cache_dir, _model_id, _method_with_steps, safe_example_id))
    
    # Log batch summary
    if len(results) > 1:
        entries = [
            {"list_hallucinations": r.get("true_answer") or [], "predicted_hallucinations": r.get("predicted_hallucinations")}
            for r in results
        ]
        batch_entry_metrics = [evaluate_hallucination_entry(e) for e in entries]
        agg = aggregate_metrics(batch_entry_metrics)
        logger.info(f"\n{'='*60}")
        logger.info(
            f"BATCH COMPLETE ({TASK_NAME}): "
            f"P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f}"
        )
        logger.info(
            f"  ({agg['correct_predictions']:.0f}/{agg['total_predictions']:.0f} pred matched, "
            f"{agg['ground_truth_found']:.0f}/{agg['total_ground_truth']:.0f} GT found)"
        )
        logger.info(f"{'='*60}")
        
        # Save batch results
        batch_results_path = os.path.join(output_dir, dataset, method, "batch_results.json")
        os.makedirs(os.path.dirname(batch_results_path), exist_ok=True)
        serializable_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != 'history'}
            serializable_results.append(sr)
        batch_data = {"results": serializable_results}
        if len(results) > 1:
            batch_data["aggregate_metrics"] = agg
        with open(batch_results_path, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        logger.info(f"Batch results saved to: {batch_results_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_traces()
