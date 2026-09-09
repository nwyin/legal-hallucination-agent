"""Small live capture / strict offline replay of the existing benchmark runner."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "replay" in sys.argv:
    os.environ["OTEL_SDK_DISABLED"] = "true"
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from benchmark_agent import run as runner
from benchmark_agent.actions import ActionType
from benchmark_agent import llm as llm_module
from benchmark_agent.evaluation import evaluate_entry, compute_metrics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def serialize(value):
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if hasattr(value, "action_type"):
        return {"action_type": value.action_type.value, "parameters": value.get_input_parameters()}
    if hasattr(value, "result") and hasattr(value, "metadata"):
        return {"result": serialize(value.result), "metadata": serialize(value.metadata)}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "replay"])
    parser.add_argument("--directory", type=Path, default=ROOT / "reference_data/smoke")
    args = parser.parse_args()
    base = args.directory.resolve()
    recording = args.mode == "record"
    if not recording:
        patch("benchmark_agent.tracing.require_capture").start()
        patch("benchmark_agent.tracing.publish_aggregate").start()
    if recording and (base / "manifest.json").exists():
        raise SystemExit("Baseline already exists; choose a new --directory to avoid overwriting it.")
    if recording and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is missing")
    out = base / ("record" if recording else "replay")
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=out / "execution.log", filemode="w")
    config = {
        "method": "boed_citation_tracker",
        "model": {"model_id": "moonshotai/kimi-k2.5", "temperature": 0, "seed": 42},
        "agent": {"thinking_enabled": True, "open_web_search_enabled": False,
                  "courtlistener_search_enabled": False, "courtlistener_opinion_access_enabled": False,
                  "max_tokens": {"action_selection": 2048, "belief_update": 2048, "prediction": 2048}},
        "step_budgets": [0, 3], "search": {"top_k": 3},
        "transport_overrides": {"temperature": 0, "seed": 42, "extra_body": {"reasoning": {"enabled": False}}},
        "allowed_actions": ["THINK", "EDIT_SCRATCHPAD", "PROVIDE_FINAL_RESPONSE"],
    }
    example = {"filename": "synthetic_smoke", "text": (
        "Synthetic software test, not a real brief. The following citation is explicitly invented: "
        "Imaginary Plaintiff v. Fictional Defendant, 999999 U.S. 999999 (2099). "
        "It supposedly holds that all contracts must be written on the moon. "
        "When using actions, first THINK about the invented citation before submitting the final response."
    ), "list_hallucinations": ["Imaginary Plaintiff v. Fictional Defendant, 999999 U.S. 999999 (2099)"]}
    if recording:
        write(base / "config.json", config)
        write(base / "input.json", example)
        sources = list((ROOT / "benchmark_agent").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
        sources += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
        write(base / "manifest.json", {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "python": sys.version,
            "scope": "Synthetic OpenRouter-only smoke; not paper accuracy or retrieval coverage",
        })
    else:
        config = json.loads((base / "config.json").read_text())
        example = json.loads((base / "input.json").read_text())

    original_factory = runner.create_environment
    allowed = {ActionType(x) for x in config["allowed_actions"]}
    def create_environment(settings):
        env = original_factory(settings)
        env.action_space = [a for a in env.action_space if a in allowed]
        env.initial_observation.metadata["available_actions"] = [a.value for a in env.action_space]
        original_step = env.step
        def step(action):
            if action is not None and action.action_type not in allowed:
                raise RuntimeError("Smoke run blocked an external/retrieval action")
            return original_step(action)
        env.step = step
        return env
    runner.create_environment = create_environment
    original_client = llm_module.get_client
    if not recording:
        def no_network(*args, **kwargs):
            raise RuntimeError("Network access forbidden during replay")
        socket.socket.connect = no_network
        socket.create_connection = no_network
    results = []
    for budget in config["step_budgets"]:
        tape_path = base / f"steps_{budget}.json"
        tape = [] if recording else json.loads(tape_path.read_text())
        index = 0
        def completion(**request):
            nonlocal index
            effective = {**request, **config["transport_overrides"]}
            if index >= 12:
                raise RuntimeError("Smoke call budget exceeded")
            if recording:
                response = original_client().chat.completions.create(**effective)
                tape.append({"requested": request, "effective": effective, "response": response.model_dump(mode="json")})
                write(tape_path, tape)
                content = response.choices[0].message.content
            else:
                if index >= len(tape) or tape[index]["requested"] != request or tape[index]["effective"] != effective:
                    raise AssertionError(f"Request mismatch at steps={budget}, call={index}")
                content = tape[index]["response"]["choices"][0]["message"]["content"]
            index += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        llm_module.get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        summary = runner.run_single_example(
            method=config["method"], example=example, paths_config={}, search_config=config["search"],
            env_settings={"max_steps": budget}, model_config=config["model"], agent_config=config["agent"],
            output_dir=str(out / "outputs"), metrics_dir=str(out / "metrics"),
            example_id=example["filename"], dataset="smoke",
        )
        if not summary.get("metrics_filepath"):
            raise AssertionError("Episode did not produce a final response and metrics")
        summary.pop("metrics_filepath")
        summary = serialize(summary)
        write(out / f"steps_{budget}_summary.json", summary)
        if not recording:
            assert index == len(tape), "Unused recorded responses"
            expected = json.loads((base / "record" / f"steps_{budget}_summary.json").read_text())
            assert summary == expected, f"Summary/trajectory mismatch for {budget} steps"
        results.append({"max_steps": budget, "calls": index, "actions": [s["action"]["action_type"] for s in summary["history"]],
                        "prediction": summary["predicted_hallucinations"], "f1": summary["f1"]})
    scoring = {"empty": compute_metrics(*evaluate_entry([], [])),
               "substring": compute_metrics(*evaluate_entry(["invented citation"], ["invented citation, extra context"]))}
    write(out / "scoring.json", scoring)
    if not recording:
        assert scoring == json.loads((base / "record/scoring.json").read_text())
    write(out / "results.json", results)
    print(json.dumps({"mode": args.mode, "results": results}, indent=2))


if __name__ == "__main__":
    main()
