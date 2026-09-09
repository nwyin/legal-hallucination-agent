"""OpenRouter model calls with SDK retries and Langfuse tracing."""

import logging
import os

from .tracing import langfuse, observe  # Initialize before the OpenAI integration.

# isort: split
from langfuse.openai import OpenAI

logger = logging.getLogger(__name__)


def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=3,
    )


class ModelAPI:
    """Injectable callable returning the first OpenRouter completion's text."""

    @observe(name="request-model", capture_input=False, capture_output=False)
    def __call__(self, model_id: str, prompt: list[dict[str, str]], **kwargs):
        langfuse.update_current_span(
            input={"messages": prompt, "model": model_id, **kwargs},
            metadata={"provider": "openrouter"},
        )
        try:
            response = get_client().chat.completions.create(messages=prompt, model=model_id, **kwargs)
            langfuse.update_current_span(
                output=response.model_dump(mode="json")
                if hasattr(response, "model_dump")
                else response.choices[0].message.content
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("OpenRouter query failed (model_id=%s)", model_id)
            raise
