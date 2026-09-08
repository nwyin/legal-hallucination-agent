"""Shared Langfuse setup; load credentials before importing instrumented clients."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse, observe, propagate_attributes
from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

logger = logging.getLogger(__name__)


def mask_otel_spans(*, params):
    """Redact configured credentials, including credentials embedded in errors.

    Content capture can be disabled for confidential briefs. This is not a
    general-purpose PII detector; content capture otherwise preserves legal text.
    """
    secrets = tuple(
        value for key, value in os.environ.items()
        if value and len(value) >= 8
        and any(part in key.upper() for part in ("API_KEY", "SECRET", "TOKEN", "PASSWORD"))
    )
    capture = os.getenv("LANGFUSE_CAPTURE_CONTENT", "true").lower() == "true"

    def redact(value):
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
        elif isinstance(value, (list, tuple)):
            value = tuple(redact(item) for item in value)
        return value

    patches = {}
    for identifier, span in params.spans.items():
        replacements = {}
        for key, value in span.attributes.items():
            if not capture and any(part in key for part in (
                ".input", ".output", ".metadata", ".status_message",
                "gen_ai.prompt", "gen_ai.completion",
            )):
                replacements[key] = "[CONTENT DISABLED]"
            else:
                masked = redact(value)
                if masked != value:
                    replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches)


# Respect offline checks and explicit operator settings. No credentials means
# tracing is disabled; the benchmark can still run normally.
enabled = (
    bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
    and os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true"
)
if not enabled:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

langfuse = Langfuse(
    tracing_enabled=enabled,
    mask_otel_spans=mask_otel_spans,
    environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
)


def flush_traces():
    """Drain queued observations without replacing an application exception."""
    if enabled:
        try:
            langfuse.flush()
        except Exception:
            logger.warning("Could not flush Langfuse traces", exc_info=True)
