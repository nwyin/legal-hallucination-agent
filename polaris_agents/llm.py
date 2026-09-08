"""Model API: provider protocol, OpenRouter client, and the routing entry point."""

import logging
import os
from typing import Optional, Protocol, Union

import attrs
from .tracing import langfuse  # Initialize before the OpenAI integration.
from langfuse.openai import OpenAI

logger = logging.getLogger(__name__)


class LLMResponse:
    model_id: str
    completion: str
    stop_reason: str
    cost: Optional[float] = None
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    logprobs: Optional[list[dict[str, float]]] = None


class ModelAPIProtocol(Protocol):
    def __call__(self, model_id, prompt, max_attempts, **kwargs):
        raise NotImplementedError


def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=3  # SDK handles retries with exponential backoff
    )

OpenRouterChatPrompt = list[dict[str, str]]
OpenRouterBasePrompt = Union[str, list[str]]

@attrs.define
class OpenRouterModel(ModelAPIProtocol):
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        raise NotImplementedError
    
    def __call__(self, model_ids, prompt, num_candidates, max_attempts=3, **kwargs):
        # OpenRouter uses OpenAI SDK which handles retries automatically
        return self._make_api_call(prompt, model_ids, num_candidates, **kwargs)
    
class OpenRouterChatModel(OpenRouterModel):
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        client = get_client()
        api_response = client.chat.completions.create(messages=prompt, model=model_id, **params)
        return api_response.choices[0].message.content


def _format_llm_exception(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc!s}"]

    # Common HTTP-ish attributes across SDKs
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")

    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status is not None and status_code is None:
            parts.append(f"response.status_code={resp_status}")
        try:
            body_text = getattr(response, "text", None)
            if body_text:
                parts.append(f"response.text={body_text}")
        except Exception:
            pass
        try:
            body_json = response.json()
            if body_json is not None:
                parts.append(f"response.json={body_json}")
        except Exception:
            pass

    # Some SDK exceptions include useful data in args only
    if getattr(exc, "args", None):
        parts.append(f"args={exc.args}")

    return " | ".join(parts)


class ModelAPI:
    """Flexible model API that routes to different providers based on configuration."""
    
    def __init__(self):
        self._providers = {
            "openrouter": OpenRouterChatModel()
        }
    
    def __call__(self, model_id: str = None, prompt: str = None, max_attempts: int = 3, provider: str = None, n: int = 1, num_candidates: int = 1, **kwargs):
        """
        Call the appropriate provider for the given model.
        
        Args:
            model_id: The model identifier
            prompt: The input prompt
            max_attempts: Maximum number of retry attempts
            provider: The provider to use (openrouter)
            n: Number of responses to generate
            num_candidates: Number of candidates to generate
            **kwargs: Additional parameters to pass to the provider
        """
        
        if provider is None:
            raise ValueError("Provider must be specified. Available provider: openrouter")
        
        if provider not in self._providers:
            raise ValueError(f"Provider '{provider}' not supported. Available providers: {list(self._providers.keys())}")
        
        provider_instance = self._providers[provider]
        try:
            return provider_instance(model_id, prompt, num_candidates, max_attempts, **kwargs)
        except Exception as e:
            logger.error(
                "LLM query failed (provider=%s, model_id=%s): %s",
                provider,
                model_id,
                _format_llm_exception(e),
                exc_info=True,
            )
            raise
