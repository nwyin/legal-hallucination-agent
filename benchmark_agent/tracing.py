"""Required experiment telemetry, independent of the agent's runtime stores."""

import base64
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse, propagate_attributes
from langfuse import observe as observe
from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Status


class TelemetryError(RuntimeError):
    """Capture is incomplete; this run must not be treated as a successful export."""


def redact(value):
    """Remove credential fields and configured secret values without truncating content."""
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if any(
                p in str(k).upper()
                for p in (
                    "API_KEY",
                    "SECRET",
                    "PASSWORD",
                    "AUTHORIZATION",
                    "ACCESS_TOKEN",
                )
            )
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if secret and any(
                p in key.upper() for p in ("API_KEY", "SECRET", "TOKEN", "PASSWORD")
            ):
                value = value.replace(secret, "[REDACTED]")
    return value


def mask_otel_spans(*, params):
    patches = {}
    for identifier, span in params.spans.items():
        replacements = {}
        for key, value in span.attributes.items():
            masked = redact(value)
            if masked != value:
                replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches)


class CaptureFailures(logging.Handler):
    """SDK background workers log some capture/score/media failures instead of raising."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.failures = []

    def emit(self, record):
        # Keep only safe diagnostic categories; SDK messages can contain credentials.
        self.failures.append(f"{record.name}:{record.levelname}")


capture_failures = CaptureFailures()
logging.getLogger("langfuse").addHandler(capture_failures)
logging.getLogger("opentelemetry.sdk").addHandler(capture_failures)


class CheckedExporter(OTLPSpanExporter):
    """Remember export failures even when the batch processor swallows them."""

    def export(self, spans):
        # SDK masking covers attributes; exception events and OTEL status need masking too.
        spans = [
            ReadableSpan(
                name=redact(s.name),
                context=s.context,
                parent=s.parent,
                resource=s.resource,
                attributes=s.attributes,
                kind=s.kind,
                start_time=s.start_time,
                end_time=s.end_time,
                instrumentation_scope=s.instrumentation_scope,
                links=s.links,
                events=[
                    Event(e.name, redact(dict(e.attributes or {})), e.timestamp)
                    for e in s.events
                ],
                status=Status(s.status.status_code, redact(s.status.description)),
            )
            for s in spans
        ]
        try:
            result = super().export(spans)
        except Exception:
            capture_failures.failures.append("OTLP export exception")
            raise
        if result != SpanExportResult.SUCCESS:
            capture_failures.failures.append("OTLP export rejected")
        return result


enabled = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
) and (
    os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true"
    and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
)
base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
exporter = None
if enabled and base_url:
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    exporter = CheckedExporter(
        endpoint=f"{base_url.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
        timeout=30,
    )
langfuse = Langfuse(
    tracing_enabled=enabled,
    base_url=base_url,
    sample_rate=1.0,
    mask=lambda data: redact(data),
    mask_otel_spans=mask_otel_spans,
    span_exporter=exporter,
    environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
)


def require_capture():
    """Fail before model spending; offline checks mock this boundary explicitly."""
    missing = [
        k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.getenv(k)
    ]
    if not base_url:
        missing.append("LANGFUSE_BASE_URL (or LANGFUSE_HOST)")
    if missing:
        raise TelemetryError(
            "Required Langfuse configuration missing: " + ", ".join(missing)
        )
    if (
        not enabled
        or os.getenv("LANGFUSE_CAPTURE_CONTENT", "true").lower() != "true"
        or os.getenv("LANGFUSE_SAMPLE_RATE", "1") not in ("1", "1.0")
        or os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false"
        or os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
    ):
        raise TelemetryError(
            "Experiments require full, unsampled Langfuse capture; remove telemetry disable/redaction settings"
        )
    try:
        if not langfuse.auth_check():
            raise TelemetryError("Langfuse credential check failed")
    except Exception as exc:
        raise TelemetryError(
            "Langfuse authentication/connectivity check failed"
        ) from exc
    check_capture()


def check_capture():
    if capture_failures.failures:
        raise TelemetryError(
            "Langfuse capture/export failed: " + ", ".join(capture_failures.failures)
        )


def flush_traces():
    if enabled:
        try:
            langfuse.flush()
        except Exception as exc:
            capture_failures.failures.append("Langfuse flush exception")
            raise TelemetryError("Could not flush Langfuse telemetry") from exc
        check_capture()


@contextmanager
def export_lifecycle():
    """Flush ended spans on success and failure, retaining both kinds of failure."""
    try:
        yield
    except BaseException as application_error:
        try:
            flush_traces()
        except Exception as export_error:
            raise BaseExceptionGroup(
                "Execution and telemetry export failed",
                [application_error, export_error],
            ) from None
        raise
    else:
        flush_traces()


run_context = ContextVar("experiment_capture", default=None)
episode_trace = ContextVar("episode_trace", default=None)


def provenance(config):
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    files = sorted((root / "benchmark_agent").glob("*.py")) + sorted(
        (root / "configs").glob("*.yaml")
    )
    files += [
        *sorted((root / "scripts").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    return redact(
        {
            "effective_config": config,
            "provider": "openrouter",
            "code_revision": git("rev-parse", "HEAD"),
            "code_dirty": bool(git("status", "--porcelain")),
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            "python": sys.version,
            "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        }
    )


@contextmanager
def experiment(config):
    require_capture()
    metadata = provenance(config)
    metadata["run_id"] = str(uuid4())
    dataset_path = config.get("data", {}).get("dataset_path")
    if dataset_path and Path(dataset_path).is_file():
        metadata["dataset_revision"] = hashlib.sha256(
            Path(dataset_path).read_bytes()
        ).hexdigest()
    token = run_context.set(metadata)
    try:
        with (
            export_lifecycle(),
            propagate_attributes(
                session_id=metadata["run_id"], metadata={"run_id": metadata["run_id"]}
            ),
        ):
            logging.getLogger(__name__).info(
                "Langfuse run/session: %s", metadata["run_id"]
            )
            yield metadata
    finally:
        run_context.reset(token)


def experiment_run(fn):
    @wraps(fn)
    def wrapped(cfg):
        from omegaconf import OmegaConf

        with experiment(OmegaConf.to_container(cfg, resolve=True)):
            return fn(cfg)

    return wrapped


def episode_capture(fn):
    """Place lifecycle outside @observe so the episode root ends before flushing."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        token = episode_trace.set(None)
        bound = inspect.signature(fn).bind(*args, **kwargs)

        def invoke():
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                exc.langfuse_trace_id = episode_trace.get()
                if episode_trace.get() and not capture_failures.failures:
                    from .evaluation import (
                        compute_metrics,
                        evaluate_entry,
                        extract_ground_truth,
                    )

                    metrics = compute_metrics(
                        *evaluate_entry(
                            extract_ground_truth(bound.arguments["example"]), None
                        )
                    )
                    score_metrics(
                        {**metrics, "error_count": 1},
                        episode_trace.get(),
                        bound.arguments["example_id"],
                    )
                raise

        try:
            if run_context.get() is not None:
                with export_lifecycle():
                    result = invoke()
            else:
                config = {k: v for k, v in bound.arguments.items() if k != "model_api"}
                with experiment(config):
                    result = invoke()
                    publish_aggregate([result])
            # Only successfully exported episodes can be skipped by subsequent runs.
            if enabled and result.get("metrics_filepath"):
                path = Path(result["metrics_filepath"])
                metrics = json.loads(path.read_text())
                metrics["langfuse_exported"] = True
                path.write_text(json.dumps(metrics, indent=2, default=str))
            return result
        except Exception as exc:
            exc.langfuse_trace_id = episode_trace.get()
            raise
        finally:
            episode_trace.reset(token)

    return wrapped


def score_metrics(metrics, trace_id, example_id=None):
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            langfuse.create_score(
                name=name,
                value=float(value),
                data_type="NUMERIC",
                trace_id=trace_id,
                metadata=redact(
                    {"run_id": run_context.get()["run_id"], "example_id": example_id}
                ),
            )
    check_capture()


def publish_aggregate(results):
    from .evaluation import aggregate_metrics, evaluate_hallucination_entry

    metrics = aggregate_metrics(
        [
            evaluate_hallucination_entry(
                {
                    "list_hallucinations": r.get("true_answer") or [],
                    "predicted_hallucinations": r.get("predicted_hallucinations"),
                }
            )
            for r in results
        ]
    )
    metrics["example_count"] = len(results)
    metrics["error_count"] = sum(
        bool(r.get("error")) or r.get("final_response") is None for r in results
    )
    with langfuse.start_as_current_observation(
        name="experiment-summary",
        input=[r["example_id"] for r in results],
        metadata=run_context.get(),
        output=metrics,
    ) as span:
        span.update(
            metadata={
                "episode_trace_ids": [r.get("langfuse_trace_id") for r in results]
            }
        )
        score_metrics(metrics, langfuse.get_current_trace_id())
    return metrics


@contextmanager
def capture_http(name, method, url, **request):
    """Record raw response bodies before parsing/filtering, never request auth headers."""
    with langfuse.start_as_current_observation(
        name=name,
        as_type="retriever",
        input=redact({"method": method, "url": url, **request}),
    ) as span:

        def capture(response):
            span.update(
                output=redact(
                    {
                        "status_code": response.status_code,
                        "body": response.text,
                        "from_cache": bool(getattr(response, "from_cache", False)),
                    }
                ),
                level="ERROR" if response.status_code >= 400 else "DEFAULT",
            )

        yield capture
