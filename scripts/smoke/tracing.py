"""Verify tracing using recorded synthetic model responses; --export sends to Langfuse."""
import argparse
from contextlib import chdir
import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Export synthetic traces to the configured project")
    args = parser.parse_args()
    os.environ["OTEL_SDK_DISABLED"] = "false"
    os.environ["LANGFUSE_TRACING_ENABLED"] = "true"
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "test"
    os.environ["LANGFUSE_SAMPLE_RATE"] = "1"
    if not args.export:
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-offline-test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-offline-test"
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    exported = []
    original_export = OTLPSpanExporter.export

    def collect(exporter, spans):
        exported.extend(spans)
        return original_export(exporter, spans) if args.export else SpanExportResult.SUCCESS

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network request in offline tracing check")

    # CourtListener creates a relative SQLite cache on import.
    with TemporaryDirectory() as cache_dir, chdir(cache_dir), patch.object(OTLPSpanExporter, "export", collect):
        if not args.export:
            network_patch = patch.object(socket.socket, "connect", no_network)
            network_patch.start()
        try:
            import httpx
            from benchmark_agent import run, llm
            from benchmark_agent.actions import ActionType
            from benchmark_agent.tracing import langfuse, flush_traces, enabled, observe
            assert enabled, "Langfuse credentials are required for --export"
            base = ROOT / "reference_data/openrouter_baseline"
            config = json.loads((base / "config.json").read_text())
            example = json.loads((base / "input.json").read_text())
            tape = json.loads((base / "steps_3.json").read_text())
            responses = iter(tape)
            calls = []

            def respond(request):
                calls.append(request)
                return httpx.Response(200, json=next(responses)["response"])

            client = llm.OpenAI(api_key="synthetic-openrouter-key", base_url="https://openrouter.invalid/v1",
                                http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            original_factory = run.create_environment

            def environment(settings):
                env = original_factory(settings)
                allowed = {ActionType(value) for value in config["allowed_actions"]}
                env.action_space = [action for action in env.action_space if action in allowed]
                env.initial_observation.metadata["available_actions"] = [a.value for a in env.action_space]
                return env

            with TemporaryDirectory() as output, patch.object(llm, "get_client", return_value=client), \
                    patch.object(run, "create_environment", environment), patch.object(run.time, "sleep"):
                try:
                    summary = run.run_single_example(
                        method=config["method"], example=example, paths_config={}, search_config=config["search"],
                        env_settings={"max_steps": 3}, model_config=config["model"], agent_config=config["agent"],
                        output_dir=output, metrics_dir=output,
                        example_id="synthetic-langfuse-smoke", dataset="tracing-smoke",
                    )
                finally:
                    flush_traces()
                    client.close()
            assert len(calls) == len(tape), (len(calls), len(tape))
            assert summary["f1"] == 1.0
            trace_id = summary["langfuse_trace_id"]
            spans = [s for s in exported if format(s.context.trace_id, "032x") == trace_id]
            roots = [s for s in spans if s.parent is None]
            assert len(roots) == 1 and roots[0].name == "verify-legal-brief"
            root = roots[0]
            span_ids = {s.context.span_id for s in spans}
            assert all(s.parent is None or s.parent.span_id in span_ids for s in spans), "Orphaned observation"
            assert root.attributes["langfuse.observation.input"]
            assert root.attributes["langfuse.observation.output"]
            generations = [s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"]
            assert len(generations) == len(tape), (len(generations), len(tape))
            for generation in generations:
                assert generation.parent is not None
                assert generation.attributes.get("langfuse.observation.model.name")
                assert generation.attributes.get("langfuse.observation.usage_details")
                assert generation.attributes.get("langfuse.observation.input")
                assert generation.attributes.get("langfuse.observation.output")
            action_spans = {step["action"].action_type.value.lower().replace("_", "-") for step in summary["history"]}
            expected_spans = {"select-action", "update-beliefs", *action_spans}
            if summary["total_steps"] >= 3:  # the prediction prompt only runs when the step budget forces it
                expected_spans.add("predict-hallucinations")
            for name in expected_spans:
                assert any(s.name == name for s in spans), name
            if not args.export:
                os.environ["TRACING_TEST_API_KEY"] = "secret-for-masking-check"
                try:
                    with langfuse.start_as_current_observation(name="check-masking", input=os.environ["TRACING_TEST_API_KEY"]):
                        pass
                    flush_traces()
                    masked = next(s for s in exported if s.name == "check-masking")
                    assert "secret-for-masking-check" not in str(dict(masked.attributes))
                    assert "[REDACTED]" in masked.attributes["langfuse.observation.input"]
                    with patch.dict(os.environ, {"LANGFUSE_CAPTURE_CONTENT": "false"}):
                        with langfuse.start_as_current_observation(name="check-content", input="confidential brief") as span:
                            span.update(output="confidential answer", metadata={"text": "confidential metadata"})
                        flush_traces()
                    masked = next(s for s in exported if s.name == "check-content")
                    assert "confidential" not in str(dict(masked.attributes))
                    assert masked.attributes["langfuse.observation.input"] == "[CONTENT DISABLED]"
                    @observe(name="check-error")
                    def fail():
                        raise ValueError("synthetic tracing failure")

                    try:
                        fail()
                    except ValueError:
                        pass
                    flush_traces()
                    error = next(s for s in exported if s.name == "check-error")
                    assert error.attributes["langfuse.observation.level"] == "ERROR"
                finally:
                    del os.environ["TRACING_TEST_API_KEY"]
            print(json.dumps({"trace_id": trace_id, "observations": len(spans), "generations": len(generations),
                              "exported": args.export, "url": langfuse.get_trace_url(trace_id=trace_id) if args.export else None}))
        finally:
            if not args.export:
                network_patch.stop()


if __name__ == "__main__":
    main()
