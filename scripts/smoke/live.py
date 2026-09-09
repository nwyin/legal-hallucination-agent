"""Bounded OpenRouter + Langfuse smoke using the production experiment runner."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--steps", type=int, choices=(0, 2), default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "smoke" / str(uuid4()))
    args = parser.parse_args()
    from omegaconf import OmegaConf
    from benchmark_agent import run, llm
    from benchmark_agent.actions import ActionType
    from benchmark_agent.tracing import langfuse
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    example = json.loads((ROOT / "reference_data/openrouter_baseline/input.json").read_text())
    data = output / "synthetic.jsonl"
    data.write_text(json.dumps(example) + "\n")
    cfg = OmegaConf.load(ROOT / "configs/legal_hallucination_checker_gpt.yaml")
    cfg.data.dataset_path = str(data)
    cfg.data.skip_completed = False
    cfg.data.limit = 1
    cfg.data.id_field = "filename"
    cfg.output_dir = str(output)
    cfg.metrics_dir = str(output / "metrics")
    cfg.dataset = "live-smoke"
    cfg.agent.model.model_id = args.model
    cfg.agent.model.temperature = 0
    cfg.agent.model.seed = 42
    cfg.agent.max_tokens = {k: 1024 for k in ("action_selection", "belief_update", "prediction")}
    cfg.agent.open_web_search_enabled = False
    cfg.agent.courtlistener_search_enabled = False
    cfg.agent.courtlistener_opinion_access_enabled = False
    cfg.environment.max_steps = args.steps
    cfg.smoke = {"allowed_actions": ["THINK", "EDIT_SCRATCHPAD", "PROVIDE_FINAL_RESPONSE"],
                 "model_call_limit": 8, "request_timeout_seconds": 20, "sdk_retries": 0}
    allowed = {ActionType(a) for a in cfg.smoke.allowed_actions}
    factory = run.create_environment
    def environment(settings):
        env = factory(settings)
        env.action_space = [a for a in env.action_space if a in allowed]
        env.initial_observation.metadata["available_actions"] = [a.value for a in env.action_space]
        return env
    client = llm.get_client().with_options(timeout=20, max_retries=0)
    model = llm.ModelAPI()
    calls = 0
    def bounded_model(**kwargs):
        nonlocal calls
        calls += 1
        if calls > 8:
            raise RuntimeError("Live smoke model-call budget exceeded")
        return model(**kwargs)
    started = time.monotonic()
    logging.basicConfig(level=logging.INFO, filename=output / "execution.log")
    try:
        with patch.object(run, "create_environment", environment), patch.object(llm, "get_client", return_value=client), \
                patch.object(run, "ModelAPI", return_value=bounded_model):
            run.main(cfg)
    finally:
        client.close()
    batch = json.loads((output / "live-smoke/boed_citation_tracker/batch_results.json").read_text())
    result = batch["results"][0]
    assert result.get("final_response") is not None, "No usable final prediction"
    assert result.get("langfuse_trace_id"), "Episode trace missing"
    if args.steps:
        assert result["total_steps"] > 0, "Agentic smoke did not execute an action"
    report = {"model": args.model, "calls": calls, "seconds": round(time.monotonic() - started, 2),
              "steps": result["total_steps"], "f1": result["f1"], "run_id": result["langfuse_run_id"],
              "trace_id": result["langfuse_trace_id"], "trace_url": langfuse.get_trace_url(trace_id=result["langfuse_trace_id"]),
              "output": str(output)}
    (output / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
