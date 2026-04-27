import attrs
import logging
from .openai_llm import OpenAIChatModel
from .openrouter_llm import OpenRouterChatModel
from .gemini_llm import GeminiChatModel
from .vllm_llm import VLLMChatModel
from .sandbox_llm import SandboxChatModel

logger = logging.getLogger(__name__)


def _format_llm_exception(exc: Exception) -> str:
    """Best-effort formatting for provider exceptions (status, body, args)."""
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
            "openai": OpenAIChatModel(),
            "openrouter": OpenRouterChatModel(),
            "gemini": GeminiChatModel(),
            "vllm": VLLMChatModel(),
            "sandbox": SandboxChatModel()
        }
    
    def __call__(self, model_id: str = None, prompt: str = None, max_attempts: int = 3, provider: str = None, n: int = 1, num_candidates: int = 1, **kwargs):
        """
        Call the appropriate provider for the given model.
        
        Args:
            model_id: The model identifier
            prompt: The input prompt
            max_attempts: Maximum number of retry attempts
            provider: The provider to use (openai, openrouter, gemini, vllm)
            n: Number of responses to generate
            num_candidates: Number of candidates to generate
            **kwargs: Additional parameters to pass to the provider
        """
        
        if provider is None:
            raise ValueError("Provider must be specified. Available providers: openai, openrouter, gemini, vllm")
        
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
        
        
        