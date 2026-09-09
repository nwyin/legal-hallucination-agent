"""OpenRouter model calls with SDK retries and Langfuse tracing."""

import logging
import os

from .tracing import langfuse  # Initialize before the OpenAI integration.
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

    def __call__(self, model_id: str, prompt: list[dict[str, str]], **kwargs):
        try:
            response = get_client().chat.completions.create(
                messages=prompt, model=model_id, **kwargs
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("OpenRouter query failed (model_id=%s)", model_id)
            raise
