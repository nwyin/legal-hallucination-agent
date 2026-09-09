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
        os.environ["LANGFUSE_BASE_URL"] = "https://langfuse.invalid"
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
            from benchmark_agent import tracing
            scores = []
            original_score = langfuse.create_score
            from langfuse._task_manager.score_ingestion_consumer import ScoreIngestionConsumer
            score_events = []
            if not args.export:
                patch.object(ScoreIngestionConsumer, "_upload_batch", lambda self, batch: score_events.extend(batch)).start()
            def score(**kwargs):
                scores.append(kwargs)
                original_score(**kwargs)
            patch.object(langfuse, "create_score", score).start()
            if not args.export:
                patch.object(langfuse, "auth_check", return_value=True).start()
            assert enabled, "Langfuse credentials are required for --export"
            base = ROOT / "reference_data/openrouter_baseline"
            config = json.loads((base / "config.json").read_text())
            example = json.loads((base / "input.json").read_text())
            tape = json.loads((base / "steps_3.json").read_text())
            responses = iter(tape)
            calls = []

            def respond(request):
                calls.append(request)
                recorded = next(responses)
                assert json.loads(request.content) == recorded["requested"], "Model-facing request changed"
                return httpx.Response(200, json=recorded["response"])

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
            assert any(s["name"] == "f1" and s["trace_id"] == summary["langfuse_trace_id"] for s in scores)
            assert any(s["name"] == "example_count" and s["trace_id"] != summary["langfuse_trace_id"] for s in scores)
            assert all(s["metadata"]["run_id"] == summary["langfuse_run_id"] for s in scores)
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
                    try:
                        with langfuse.start_as_current_observation(name="check-exception-masking"):
                            raise ValueError(os.environ["TRACING_TEST_API_KEY"])
                    except ValueError:
                        pass
                    flush_traces()
                    error_span = next(s for s in exported if s.name == "check-exception-masking")
                    assert "secret-for-masking-check" not in str(error_span.status.description)
                    assert "secret-for-masking-check" not in str([dict(e.attributes) for e in error_span.events])
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
            # Real retrieval clients, mocked only at HTTP: raw evidence must survive cache deletion.
            from benchmark_agent import courtlistener, web_search
            from benchmark_agent.actions import AccessCourtListenerOpinion, OpenWebSearch, OpenCourtListenerSearch
            from benchmark_agent.environment import HallucinationCheckerEnvironment
            from requests import Response
            def http_response(payload):
                response = Response()
                response.status_code = 200
                response._content = json.dumps(payload).encode()
                return response
            opinion = {"id": 123, "plain_text": "Opinion text. " * 500 + "RAW_OPINION_TAIL", "html": "<p>Raw HTML</p>"}
            search_payload = {"count": 2, "results": [{"id": 1, "snippet": "visible result"}], "unfiltered": "RAW_SEARCH_FIELD"}
            web_payload = {"organic_results": [{"title": "Synthetic", "link": "https://example.invalid", "snippet": "visible"}],
                           "extra": "RAW_WEB_FIELD"}
            with TemporaryDirectory() as cache, langfuse.start_as_current_observation(name="synthetic-retrieval") as root:
                env = HallucinationCheckerEnvironment(brief_text="synthetic", brief_info={}, max_steps=5, opinion_cache_dir=cache)
                with patch.object(courtlistener.requests, "get", return_value=http_response(opinion)), patch.object(courtlistener.time, "sleep"):
                    observation = env.step(AccessCourtListenerOpinion(opinion_id="123"))
                assert "RAW_OPINION_TAIL" not in str(observation)
                assert "RAW_OPINION_TAIL" in (Path(cache) / "123.json").read_text()
                with patch.object(courtlistener.requests, "get", return_value=http_response(search_payload)), patch.object(courtlistener.time, "sleep"):
                    env.step(OpenCourtListenerSearch(query="synthetic"))
                with patch.dict(os.environ, {"SERPAPI_API_KEY": "synthetic-serp-secret"}), patch.object(web_search.requests, "get", return_value=http_response(web_payload)):
                    env.step(OpenWebSearch(query="synthetic"))
            flush_traces()
            raw_spans = [s for s in exported if s.name in {"request-courtlistener", "request-serpapi"}]
            content = str([dict(s.attributes) for s in raw_spans])
            assert all(marker in content for marker in ("RAW_OPINION_TAIL", "RAW_SEARCH_FIELD", "RAW_WEB_FIELD"))
            assert "synthetic-serp-secret" not in content
            if not args.export:
                # Exercise the real episode decorator on an exception before model calls.
                with TemporaryDirectory() as output, patch.object(run, "create_environment", side_effect=ValueError("synthetic setup error")):
                    try:
                        run.run_single_example(method=config["method"], example=example, paths_config={},
                            search_config=config["search"], env_settings={"max_steps": 0}, model_config=config["model"],
                            agent_config=config["agent"], output_dir=output, metrics_dir=output,
                            example_id="failed-synthetic", dataset="tracing-smoke")
                    except ValueError as error:
                        failed_id = error.langfuse_trace_id
                    else:
                        raise AssertionError("Failed episode did not raise")
                    assert any(s["trace_id"] == failed_id and s["name"] == "error_count" for s in scores)
                    assert any(format(s.context.trace_id, "032x") == failed_id for s in exported)
                    (Path(output) / "unexported.json").write_text('{"langfuse_exported": false}')
                    (Path(output) / "exported.json").write_text('{"langfuse_exported": true}')
                    assert run.get_completed_examples(output) == {"exported"}
                # Failures must flush ended spans, and export failures must reach the caller.
                with patch.object(langfuse, "flush", wraps=langfuse.flush) as flush:
                    try:
                        with tracing.export_lifecycle():
                            with langfuse.start_as_current_observation(name="failed-episode"):
                                raise ValueError("synthetic episode failure")
                    except ValueError:
                        pass
                    assert flush.call_count == 1
                assert any(s.name == "failed-episode" for s in exported)
                with patch.object(OTLPSpanExporter, "export", return_value=SpanExportResult.FAILURE):
                    try:
                        with tracing.export_lifecycle():
                            with langfuse.start_as_current_observation(name="failed-export"):
                                pass
                    except tracing.TelemetryError:
                        pass
                    else:
                        raise AssertionError("Export rejection was swallowed")
                tracing.capture_failures.failures.clear()
                with patch.object(langfuse, "flush", side_effect=RuntimeError("synthetic flush failure")):
                    try:
                        with tracing.export_lifecycle():
                            raise ValueError("original error")
                    except ExceptionGroup as errors:
                        assert isinstance(errors.exceptions[0], ValueError)
                        assert isinstance(errors.exceptions[1], tracing.TelemetryError)
                    else:
                        raise AssertionError("Failure lifecycle lost exceptions")
                tracing.capture_failures.failures.clear()
                assert score_events, "No scores reached the SDK ingestion queue"
                assert any(e["body"]["traceId"] == trace_id and e["body"]["name"] == "f1" for e in score_events)
                with patch.object(ScoreIngestionConsumer, "_upload_batch", side_effect=RuntimeError("synthetic score export failure")):
                    original_score(name="failed-score", value=0, trace_id=trace_id)
                    try:
                        flush_traces()
                    except tracing.TelemetryError:
                        pass
                    else:
                        raise AssertionError("Score export failure was swallowed")
                tracing.capture_failures.failures.clear()
                for setting, value in (("LANGFUSE_SECRET_KEY", ""), ("LANGFUSE_CAPTURE_CONTENT", "false"),
                                       ("LANGFUSE_SAMPLE_RATE", "0.5"), ("OTEL_SDK_DISABLED", "true")):
                    with patch.dict(os.environ, {setting: value}):
                        try:
                            tracing.require_capture()
                        except tracing.TelemetryError:
                            pass
                        else:
                            raise AssertionError(f"Capture accepted {setting}={value}")
            print(json.dumps({"trace_id": trace_id, "observations": len(spans), "generations": len(generations),
                              "exported": args.export, "url": langfuse.get_trace_url(trace_id=trace_id) if args.export else None}))
        finally:
            if not args.export:
                network_patch.stop()


if __name__ == "__main__":
    main()
