#!/usr/bin/env python3
"""
Citation benchmark runner for the legal hallucination checker task.

Loads examples from a JSONL dataset, runs a citation-focused agent, and writes
episode metrics.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .agent import (
    Agent,
    BayesianOptimalExperimentalDesignAgent,
    BOEDCitationTrackerAgent,
)
from .environment import Environment, HallucinationCheckerEnvironment, Observation
from .evaluation import compute_metrics, evaluate_entry, extract_ground_truth
from .llm import ModelAPI
from .prompts import (
    BOEDActionSelectionPromptConstructor,
    BOEDBeliefUpdatePromptConstructor,
    BOEDCitationTrackerBeliefUpdatePromptConstructor,
    BOEDCitationTrackerPredictionPromptConstructor,
    BOEDPredictionPromptConstructor,
    LegalHallucinationCheckerDomainKnowledge,
)
from .recording import (
    MetricsCollector,
    log_action_basic,
    log_beliefs,
    log_final_response,
    log_generic_observation,
    log_initial_state,
    log_search_action,
    log_step_header,
    log_think_action,
)
from .tracing import (
    TelemetryError,
    check_capture,
    episode_capture,
    episode_trace,
    experiment_run,
    langfuse,
    observe,
    propagate_attributes,
    publish_aggregate,
    run_context,
    score_metrics,
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

# method -> agent class. "boed" keeps free-form beliefs as an ablation of the
# paper's citation-tracker belief list.
AGENT_REGISTRY = {
    "boed": BayesianOptimalExperimentalDesignAgent,
    "boed_citation_tracker": BOEDCitationTrackerAgent,
}

FINAL_RESPONSE_REASK = (
    "Your response was empty or could not be parsed as a JSON list. "
    "Please provide your final answer as a valid JSON array of hallucinated strings, "
    'e.g. ["citation1", "citation2"]. If no hallucinations were found, return [].'
)


# --- Setup ---


def setup_logging(log_config: dict[str, Any]):
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )


def setup_experiment_logging(
    dataset: str,
    method: str,
    model_id: str,
    example_id: str,
    experiments_dir: str = "outputs/experiments",
):
    """Add a file handler writing to {experiments_dir}/{dataset}/{model_id}/{method}/{example_id}_{timestamp}.log."""
    log_dir = Path(experiments_dir) / dataset / model_id / method
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / (f"{example_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Experiment log: {log_filepath}")
    return str(log_filepath)


def check_required_api_keys(agent_config: dict[str, Any]) -> bool:
    """Report which API keys are present and refuse to run without the required ones."""
    required = {"OPENROUTER_API_KEY"}
    # Without this key every OPEN_WEB_SEARCH would fail; refuse to run rather than
    # silently benchmark an agent that cannot use one of the paper's eight actions.
    if agent_config.get("open_web_search_enabled", False):
        required.add("SERPAPI_API_KEY")
    missing = []
    for key in ("OPENROUTER_API_KEY", "SERPAPI_API_KEY", "COURTLISTENER_API_KEY"):
        if os.getenv(key):
            logger.info(f"{key} found")
        elif key in required:
            logger.error(f"{key} is required but not found. Set: export {key}='your_key'")
            missing.append(key)
        else:
            logger.info(f"{key} not set")
    return not missing


# --- Data loading and paths ---


def load_examples(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_example_by_id(examples: list[dict[str, Any]], example_id: str, id_field: str) -> dict[str, Any] | None:
    return next((ex for ex in examples if str(ex.get(id_field)) == str(example_id)), None)


def get_completed_examples(metrics_dir: str) -> set:
    """Example IDs with a metrics file in metrics_dir (file stem = example id)."""
    completed = set()
    for path in Path(metrics_dir).glob("*.json"):
        try:
            if json.loads(path.read_text()).get("langfuse_exported") is True:
                completed.add(path.stem)
        except (OSError, ValueError):
            continue
    return completed


def method_key(method: str, max_steps: int) -> str:
    """Result-folder name for a method at a step budget, e.g. boed_citation_tracker_steps30."""
    return f"{method}_steps{max_steps}"


def episode_metrics_dir(metrics_dir: str, dataset: str, model_id: str, method: str, max_steps: int) -> str:
    return str(Path(metrics_dir) / dataset / model_id / method_key(method, max_steps))


def episode_opinion_cache_dir(
    paths_config: dict[str, Any],
    output_dir: str,
    model_id: str,
    method: str,
    max_steps: int,
    example_id: str,
) -> str:
    """Per-episode opinion cache, so parallel runs never clear each other's opinions."""
    base = paths_config.get("opinion_cache_dir") or Path(output_dir) / "opinion_cache"
    safe_example_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(example_id))
    return str(Path(base) / model_id / method_key(method, max_steps) / safe_example_id)


def clear_opinion_cache(opinion_cache_dir: str) -> None:
    if not opinion_cache_dir or not Path(opinion_cache_dir).is_dir():
        return
    try:
        for path in Path(opinion_cache_dir).iterdir():
            if path.is_file() and path.name.endswith(".json"):
                path.unlink()
    except OSError as e:
        logger.warning(f"Failed to clear opinion cache at {opinion_cache_dir}: {e}")


# --- Factories ---


def create_environment(env_config: dict[str, Any]) -> Environment:
    return HallucinationCheckerEnvironment(
        brief_info=env_config.get("brief_info", {}),
        brief_text=env_config.get("brief_text", ""),
        max_steps=env_config.get("max_steps", 30),
        search_top_k=env_config.get("search_top_k", 3),
        opinion_cache_dir=env_config.get("opinion_cache_dir"),
    )


def create_agent(
    method: str,
    environment: Environment,
    model_api: ModelAPI,
    model_config: dict[str, Any],
    agent_config: dict[str, Any],
) -> Agent:
    """Create an agent based on method name."""
    if method not in AGENT_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(AGENT_REGISTRY)}")

    max_tokens_config = agent_config.get("max_tokens", {})
    if not isinstance(max_tokens_config, dict):
        raise ValueError("agent.max_tokens must be a dictionary with keys: action_selection, belief_update, prediction")

    params = {
        "environment": environment,
        "model_api": model_api,
        "model_id": model_config.get("model_id"),
        "max_tokens": max_tokens_config.get("action_selection", 16000),
        "max_tokens_config": max_tokens_config,
        "temperature": model_config.get("temperature", 0.7),
        "seed": model_config.get("seed"),
        "thinking_enabled": agent_config.get("thinking_enabled", True),
        "open_web_search_enabled": agent_config.get("open_web_search_enabled", False),
        "courtlistener_search_enabled": agent_config.get("courtlistener_search_enabled", False),
        "courtlistener_opinion_access_enabled": agent_config.get("courtlistener_opinion_access_enabled", False),
        # Optional overrides for belief update calls only
        "belief_update_model_id": model_config.get("belief_update_model_id"),
        "belief_update_temperature": model_config.get("belief_update_temperature"),
        "action_selection_prompt_constructor": BOEDActionSelectionPromptConstructor(TASK_DOMAIN_KNOWLEDGE),
    }
    if method == "boed":
        params["belief_update_prompt_constructor"] = BOEDBeliefUpdatePromptConstructor(TASK_DOMAIN_KNOWLEDGE)
        params["prediction_prompt_constructor"] = BOEDPredictionPromptConstructor(TASK_DOMAIN_KNOWLEDGE)
        params["task_belief_prior"] = TASK_BELIEF_PRIOR
    else:
        params["belief_update_prompt_constructor"] = BOEDCitationTrackerBeliefUpdatePromptConstructor(
            TASK_DOMAIN_KNOWLEDGE
        )
        params["prediction_prompt_constructor"] = BOEDCitationTrackerPredictionPromptConstructor(TASK_DOMAIN_KNOWLEDGE)
    return AGENT_REGISTRY[method](**params)


# --- Episode runner ---


def run_episode(agent: Agent, environment: Environment, metrics_collector: MetricsCollector) -> dict[str, Any]:
    """Run a complete episode and collect metrics."""
    environment.reset()
    observation = environment.get_initial_observation()
    log_initial_state(agent, environment, observation, TASK_NAME)

    final_response = None
    is_correct = None

    if environment.max_steps == 0:
        # Non-agentic baseline: one direct prediction, no actions or belief updates.
        logger.info("max_steps=0 detected - skipping action selection and requesting direct prediction.")
        for attempt in range(3):
            try:
                final_response, _ = agent.get_current_prediction()
            except Exception as e:
                logger.error(f"Direct prediction failed for max_steps=0 (attempt {attempt + 1}/3): {e}")
                final_response = None
            if final_response and str(final_response).strip():
                break
            logger.warning(f"Direct prediction empty or invalid (attempt {attempt + 1}/3), retrying...")
        if final_response is not None:
            agent.store_final_response(final_response)
    else:
        final_response_retries = 0
        while not environment.is_terminated() and not environment.is_truncated():
            step_num = agent.current_step + 1
            log_step_header(step_num, observation)

            action = None
            for attempt in range(3):
                action = agent.select_action(observation)
                if action is not None:
                    break
                logger.warning(
                    "Action parsing failed at step %s (attempt %s/3). Retrying select_action.",
                    step_num,
                    attempt + 1,
                )
            if action is None:
                logger.error(
                    "Action parsing failed after 3 attempts at step %s. Terminating episode.",
                    step_num,
                )
                break

            observation = environment.step(action)
            agent.update_state(action, observation)

            action_type = log_action_basic(action)
            if action_type == "PROVIDE_FINAL_RESPONSE":
                final_response, is_correct, _ = log_final_response(action, observation)
                if not final_response and final_response_retries < 2:
                    # Empty final answer: reopen the episode and ask again.
                    final_response_retries += 1
                    logger.warning(
                        "PROVIDE_FINAL_RESPONSE returned empty or unparseable list (attempt %s/2). Prompting agent to retry.",
                        final_response_retries,
                    )
                    environment.terminated = False
                    observation = Observation(
                        result=FINAL_RESPONSE_REASK,
                        metadata={
                            "action_type": "PROVIDE_FINAL_RESPONSE",
                            "error": "empty_or_unparseable",
                        },
                    )
                    final_response = None
            elif action_type == "THINK":
                log_think_action(action)
            elif action_type in ("OPEN_WEB_SEARCH", "OPEN_COURTLISTENER_SEARCH"):
                log_search_action(action, observation)
            else:
                log_generic_observation(observation)
            log_beliefs(agent)
            logger.info("")

            metrics_collector.record_step(
                action=action,
                reward=None,
                metadata={
                    "observation_result": observation.result,
                    "observation_metadata": observation.metadata,
                    "current_beliefs": agent.get_current_beliefs(),
                    "step_number": agent.current_step,
                },
            )

    # The agent parses its own final response into the list that gets scored.
    predicted_hallucinations = agent.last_final_response_list if final_response is not None else None
    # evaluate_entry accepts both a list of spans and a {span: type} dict.
    metrics = compute_metrics(*evaluate_entry(environment.key, predicted_hallucinations))
    current_beliefs = agent.get_current_beliefs()

    summary = {
        "total_steps": agent.current_step,
        "terminated": environment.is_terminated(),
        "truncated": environment.is_truncated(),
        "final_response": final_response,
        "predicted_hallucinations": predicted_hallucinations,
        "is_correct": is_correct,
        "accuracy": metrics["f1"],  # F1 serves as the scalar accuracy for this task
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "true_answer": environment.key,
        "question": environment.question,
        "history": agent.history,
        "model_used": agent.model_id,
        "task_beliefs": current_beliefs.get("task_beliefs"),
        "design_beliefs": current_beliefs.get("design_beliefs"),
    }
    logger.info(
        f"Episode completed - Steps: {agent.current_step}, "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}, Fully Correct: {is_correct}"
    )
    return summary


def log_results(summary: dict[str, Any], method: str, example_id: str):
    """Compact end-of-episode report; per-step detail was already logged during the run."""
    logger.info("\n" + "=" * 60)
    logger.info(f"{TASK_NAME.upper()} - {method.upper()} - {example_id}")
    logger.info("=" * 60)
    logger.info(f"Model: {summary.get('model_used', 'Unknown')}")
    logger.info(f"Total Steps: {summary['total_steps']}  Terminated: {summary['terminated']}")
    if summary.get("final_response"):
        logger.info(f"Prediction: {summary['final_response']}")
        logger.info(f"Precision: {summary['precision']:.3f}  Recall: {summary['recall']:.3f}  F1: {summary['f1']:.3f}")
    if summary.get("task_beliefs"):
        logger.info(f"Task Beliefs: {summary['task_beliefs'][:200]}...")
    for i, step in enumerate(summary.get("history", []), 1):
        logger.info(f"Step {i}: {step['action'].action_type.value} {str(step['action'].get_input_parameters())[:150]}")
    logger.info("=" * 60)


@episode_capture
@observe(
    name="verify-legal-brief",
    as_type="agent",
    capture_input=False,
    capture_output=False,
)
def run_single_example(
    method: str,
    example: dict[str, Any],
    paths_config: dict[str, Any],
    search_config: dict[str, Any],
    env_settings: dict[str, Any],
    model_config: dict[str, Any],
    agent_config: dict[str, Any],
    output_dir: str,
    metrics_dir: str,
    example_id: str,
    dataset: str,
    experiment_logs: bool = False,
    model_api: ModelAPI = None,
) -> dict[str, Any]:
    """Run a single example and return results."""
    model_id = model_config.get("model_id", "unknown")
    max_steps = env_settings.get("max_steps", 30)
    episode_trace.set(langfuse.get_current_trace_id())
    langfuse.update_current_span(
        input=example.get("text", ""),
        metadata={
            **(run_context.get() or {}),
            "example_id": example_id,
            "dataset": dataset,
            "method": method,
            "example_sha256": hashlib.sha256(json.dumps(example, sort_keys=True).encode()).hexdigest(),
            "ground_truth": extract_ground_truth(example),
            "model": model_id,
            "max_steps": max_steps,
        },
    )
    if experiment_logs:
        setup_experiment_logging(dataset, method, model_id, example_id)

    ground_truth = extract_ground_truth(example)
    environment = create_environment(
        {
            "brief_info": {
                "list_hallucinations": ground_truth,
                "list_hallucination_types": example.get("list_hallucination_types", []),
            },
            "brief_text": example.get("text", ""),
            "max_steps": max_steps,
            "search_top_k": search_config.get("top_k", 3),
            "opinion_cache_dir": episode_opinion_cache_dir(
                paths_config, output_dir, model_id, method, max_steps, example_id
            ),
        }
    )
    agent = create_agent(method, environment, model_api or ModelAPI(), model_config, agent_config)

    metrics_collector = MetricsCollector(
        save_dir=episode_metrics_dir(metrics_dir, dataset, model_id, method, max_steps)
    )
    metrics_collector.start_episode(
        episode_id=example_id,
        task_name=TASK_NAME,
        agent_type=agent.__class__.__name__,
        environment_info={},
        method=method,
        model_id=model_id,
    )

    with propagate_attributes(
        tags=["legal-hallucination-checker", method],
        metadata={"dataset": dataset, "example_id": example_id, "method": method},
    ):
        summary = run_episode(agent, environment, metrics_collector)

    trace_id = langfuse.get_current_trace_id()
    if trace_id:
        summary["langfuse_trace_id"] = trace_id
        summary["langfuse_run_id"] = run_context.get()["run_id"]
        score_metrics(
            {k: summary[k] for k in ("precision", "recall", "f1", "accuracy", "total_steps")},
            trace_id,
            example_id,
        )

    # Save ground truth, prediction, raw response, scores, and trace id alongside the trajectory.
    predicted = summary.get("predicted_hallucinations")
    extra_data = {
        "list_hallucinations": ground_truth or [],
        "predicted_hallucinations": predicted,
        "final_response_raw": summary.get("final_response"),  # agent response string before parsing
        "langfuse_trace_id": trace_id,
        "langfuse_run_id": (run_context.get() or {}).get("run_id"),
        **compute_metrics(*evaluate_entry(ground_truth, predicted)),
    }
    no_response = summary.get("final_response") is None
    summary["metrics_filepath"] = None if no_response else metrics_collector.end_episode(extra_data=extra_data)
    summary["episode_id"] = example_id
    summary["example_id"] = example_id
    langfuse.update_current_span(
        output=summary.get("final_response"),
        metadata={
            "precision": summary.get("precision"),
            "recall": summary.get("recall"),
            "f1": summary.get("f1"),
            "total_steps": summary.get("total_steps"),
        },
        level="ERROR" if no_response else "DEFAULT",
        status_message="No final response" if no_response else None,
    )
    log_results(summary, method, example_id)
    return summary


# --- Main ---


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="legal_hallucination_checker_gpt",
)
@experiment_run
def main(cfg: DictConfig):
    """Run every selected example from the dataset and write per-episode metrics plus a batch summary."""
    log_config = OmegaConf.to_container(cfg.get("logging", {}), resolve=True)
    setup_logging(log_config)
    logger.info("Langfuse run/session: %s", run_context.get()["run_id"])

    method = cfg.get("method", "boed_citation_tracker")
    requested_task = cfg.get("task")
    if requested_task is not None and requested_task != TASK_NAME:
        raise ValueError(
            f"Unsupported task '{requested_task}'. This runner only supports '{TASK_NAME}'. "
            "Set task to 'legal_hallucination_checker' or omit task in the config."
        )
    # Dataset names the result folder (metrics/plots/output); defaults to TASK_NAME for consistency
    dataset = cfg.get("dataset") or TASK_NAME
    logger.info(f"Starting experiment: task={TASK_NAME}, method={method}, dataset={dataset}")

    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    paths_config = OmegaConf.to_container(cfg.paths, resolve=True) if "paths" in cfg else {}
    search_config = OmegaConf.to_container(cfg.search, resolve=True) if "search" in cfg else {}
    env_settings = OmegaConf.to_container(cfg.environment, resolve=True)
    agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
    model_config = OmegaConf.to_container(cfg.agent.model, resolve=True)
    model_id = model_config.get("model_id", "unknown")
    max_steps = env_settings.get("max_steps", 30)

    if not check_required_api_keys(agent_config):
        raise ValueError("Required model/search credentials missing")
    output_dir = cfg.get("output_dir", "outputs")
    metrics_dir = cfg.get("metrics_dir", "metrics")
    if cfg.get("test_run", False):
        logger.info("TEST RUN mode enabled")
        metrics_dir = str(Path(metrics_dir) / "test")
        output_dir = str(Path(output_dir) / "test")

    dataset_path = data_config.get("dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    logger.info(f"Loading examples from: {dataset_path}")
    examples = load_examples(dataset_path)
    logger.info(f"Loaded {len(examples)} examples")
    id_field = data_config.get("id_field") or TASK_ID_FIELD

    example_id = data_config.get("example_id")
    if example_id:
        example = get_example_by_id(examples, example_id, id_field)
        if not example:
            logger.error(f"Example not found: {example_id}")
            return
        examples = [example]
        logger.info(f"Running single example: {example_id}")
    else:
        limit = data_config.get("limit")
        if limit:
            examples = examples[:limit]
            logger.info(f"Limited to {limit} examples")
        if data_config.get("skip_completed", True):
            completed = get_completed_examples(episode_metrics_dir(metrics_dir, dataset, model_id, method, max_steps))
            original_count = len(examples)
            examples = [ex for ex in examples if str(ex.get(id_field)) not in completed]
            logger.info(f"Skipping {original_count - len(examples)} completed examples")
        # Shard: partition remaining examples after skip_completed for even load distribution.
        # All shards must be launched at the same time so they see the same remaining list.
        # Usage: data.shard=[shard_index],[num_shards]  e.g. data.shard=0,4
        shard = data_config.get("shard")
        if shard:
            if isinstance(shard, (list, tuple)):
                shard_index, num_shards = int(shard[0]), int(shard[1])
            else:
                shard_index, num_shards = [int(x) for x in str(shard).split(",")]
            examples = examples[shard_index::num_shards]
            logger.info(f"Shard {shard_index}/{num_shards}: {len(examples)} examples")

    if not examples:
        logger.info("No examples to run!")
        return

    model_api = ModelAPI()
    results = []
    for example in tqdm(examples, desc=f"Running {TASK_NAME}/{method}"):
        ex_id = str(example.get(id_field, f"example_{len(results)}"))
        try:
            results.append(
                run_single_example(
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
                    experiment_logs=log_config.get("experiment_logs", False),
                    model_api=model_api,
                )
            )
        except TelemetryError:
            raise
        except Exception as e:
            check_capture()
            logger.exception(f"Failed on example {ex_id}: {e}")
            results.append(
                {
                    "example_id": ex_id,
                    "error": str(e),
                    "is_correct": None,
                    "true_answer": extract_ground_truth(example),
                    "langfuse_trace_id": getattr(e, "langfuse_trace_id", None),
                }
            )
        finally:
            # Each episode has its own cache subdirectory; clear it so disk use does not accumulate.
            clear_opinion_cache(episode_opinion_cache_dir(paths_config, output_dir, model_id, method, max_steps, ex_id))

    agg = publish_aggregate(results)
    if results:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"BATCH COMPLETE ({TASK_NAME}): P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f}")
        logger.info(
            f"  ({agg['correct_predictions']:.0f}/{agg['total_predictions']:.0f} pred matched, "
            f"{agg['ground_truth_found']:.0f}/{agg['total_ground_truth']:.0f} GT found)"
        )
        logger.info(f"{'=' * 60}")

        batch_results_path = Path(output_dir) / dataset / method / "batch_results.json"
        batch_results_path.parent.mkdir(parents=True, exist_ok=True)
        batch_data = {
            "results": [{k: v for k, v in r.items() if k != "history"} for r in results],
            "aggregate_metrics": agg,
        }
        with batch_results_path.open("w") as f:
            json.dump(batch_data, f, indent=2, default=str)
        logger.info(f"Batch results saved to: {batch_results_path}")


if __name__ == "__main__":
    main()
