import attrs
import requests
from typing import Union, Optional
from openai import OpenAI

from .base_llm import ModelAPIProtocol

def get_della_base_url(
    port: Optional[int] = None,
    host: str = "localhost",
    scheme: str = "http",
    base_url: Optional[str] = None,
) -> str:
    """
    Get the base URL for della-inference API (without /v1).
    
    Args:
        port: Port number for della-inference API (required unless base_url is set).
        host: Hostname for della-inference API.
        scheme: URL scheme (http or https).
        base_url: Full base URL override (e.g., "http://compute001:8001").
    
    Returns:
        Base URL string (e.g., "http://localhost:8001")
    """
    if base_url:
        return base_url.rstrip("/")

    if port is None:
        raise ValueError(
            "Port must be specified explicitly when base_url is not set. "
            "Set DellaInferenceModel(port=..., host=...) or DellaInferenceModel(base_url=...)."
        )
    
    return f"{scheme}://{host}:{port}"


def get_della_client(
    port: Optional[int] = None,
    host: str = "localhost",
    scheme: str = "http",
    base_url: Optional[str] = None,
    timeout: float = 60.0,
):
    """
    Initialize OpenAI client for della-inference API.
    
    Args:
        port: Port number for della-inference API (required unless base_url is set).
        host: Hostname for della-inference API.
        scheme: URL scheme (http or https).
        base_url: Full base URL override (e.g., "http://compute001:8001").
        timeout: Timeout in seconds for API requests (default: 60.0)
    
    Returns:
        OpenAI client configured for della-inference
    """
    resolved_base_url = get_della_base_url(
        port=port,
        host=host,
        scheme=scheme,
        base_url=base_url,
    )
    return OpenAI(
        base_url=f"{resolved_base_url}/v1",
        api_key="token-abc123",  # della-inference uses any token
        timeout=timeout,
        max_retries=3  # SDK handles retries with exponential backoff
    )

DellaChatPrompt = list[dict[str, str]]
DellaBasePrompt = Union[str, list[str]]

@attrs.define
class DellaInferenceModel(ModelAPIProtocol):
    """Model API for della-inference served models."""
    
    port: Optional[int] = None  # Port number (required, must be set explicitly)
    host: str = "localhost"
    scheme: str = "http"
    base_url: Optional[str] = None
    timeout: Optional[float] = None  # Timeout in seconds for API requests
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        raise NotImplementedError
    
    def __call__(self, model_ids, prompt, num_candidates, max_attempts=3, **kwargs):
        # Della uses OpenAI SDK which handles retries automatically
        return self._make_api_call(prompt, model_ids, num_candidates, **kwargs)

class DellaInferenceChatModel(DellaInferenceModel):
    """Chat model interface for della-inference."""
    
    def _truncate_messages(self, messages: DellaChatPrompt, max_chars: int = 48000) -> DellaChatPrompt:
        """
        Truncate chat messages to avoid exceeding the model context window.

        Uses a character-based heuristic (~2 chars/token): keeps the first message
        (usually system) and the most recent messages until under max_chars,
        so we stay under typical 32K context when the client also requests 8K output
        (32768 - 8192 = 24576 max input tokens; 24000 * 2 = 48000 chars with safety margin).
        """
        try:
            if not messages:
                return messages
            total = sum(len(str(m.get("content", ""))) for m in messages)
            if total <= max_chars:
                return messages

            # Keep first message (usually system) and then fill from the end
            first = [messages[0]]
            first_len = len(str(messages[0].get("content", "")))
            rest = messages[1:]
            budget = max_chars - first_len
            if budget <= 0:
                return first

            kept: DellaChatPrompt = []
            running = 0
            for m in reversed(rest):
                content = str(m.get("content", ""))
                length = len(content)
                if running + length > budget:
                    if not kept:
                        # Must include at least one message — truncate its content to fit
                        truncated = dict(m)
                        truncated["content"] = content[:budget]
                        kept.append(truncated)
                    break
                kept.append(m)
                running += length
            kept.reverse()
            return first + kept
        except Exception:
            return messages
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        client_kwargs = {
            "port": self.port,
            "host": self.host,
            "scheme": self.scheme,
            "base_url": self.base_url,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        client = get_della_client(**client_kwargs)

        # Ensure num_candidates/n parameter is set
        if num_candidates > 1:
            params['n'] = num_candidates

        # Best-effort truncation to avoid context-length errors on the server
        if isinstance(prompt, list):
            prompt = self._truncate_messages(prompt)

        # Retry with progressively tighter truncation on context-length errors
        for attempt, max_chars in enumerate([None, 40000, 32000, 24000]):
            if attempt > 0 and isinstance(prompt, list):
                prompt = self._truncate_messages(prompt, max_chars=max_chars)
            try:
                api_response = client.chat.completions.create(
                    messages=prompt,
                    model=model_id,
                    **params
                )
                break
            except Exception as e:
                if attempt < 3 and "input_tokens" in str(e) and "context length" in str(e):
                    continue
                raise
        
        # Handle multiple candidates if requested
        if num_candidates > 1:
            return [choice.message.content for choice in api_response.choices]
        else:
            return api_response.choices[0].message.content


class DellaInferenceEmbeddingModel(DellaInferenceModel):
    """Embeddings via /v1/embeddings on pooling-runner servers."""
    
    def _make_api_call(self, inputs, model_id, num_candidates, **params):
        client_kwargs = {
            "port": self.port,
            "host": self.host,
            "scheme": self.scheme,
            "base_url": self.base_url,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        client = get_della_client(**client_kwargs)
        
        resp = client.embeddings.create(
            model=model_id,
            input=inputs,
            **params,
        )
        
        # Return list of embedding vectors
        return [d.embedding for d in resp.data]


class DellaInferencePoolingClient(DellaInferenceModel):
    """Low-level client for vLLM /pooling (token_embed, token_classify, etc.)."""
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        base_url = get_della_base_url(
            port=self.port,
            host=self.host,
            scheme=self.scheme,
            base_url=self.base_url,
        )
        api_url = f"{base_url}/pooling"
        
        payload = {"model": model_id, **prompt, **params}
        
        resp = requests.post(api_url, json=payload, timeout=60)
        resp.raise_for_status()
        
        return resp.json()


class DellaInferenceClassifyModel(DellaInferenceModel):
    """Sequence classification via vLLM /classify."""
    
    def _make_api_call(self, inputs, model_id, num_candidates, **params):
        base_url = get_della_base_url(
            port=self.port,
            host=self.host,
            scheme=self.scheme,
            base_url=self.base_url,
        )
        api_url = f"{base_url}/classify"
        
        payload = {
            "model": model_id,
            "input": inputs,  # string or list[str]
            **params,
        }
        
        resp = requests.post(api_url, json=payload, timeout=60)
        resp.raise_for_status()
        
        return resp.json()


class DellaInferenceScoreModel(DellaInferenceModel):
    """Pairwise scoring via vLLM /score."""
    
    def _make_api_call(self, pairs, model_id, num_candidates, **params):
        base_url = get_della_base_url(
            port=self.port,
            host=self.host,
            scheme=self.scheme,
            base_url=self.base_url,
        )
        api_url = f"{base_url}/score"
        
        # pairs: list[tuple[str, str]]
        inputs = [{"text_1": t1, "text_2": t2} for (t1, t2) in pairs]
        
        payload = {
            "model": model_id,
            "input": inputs,
            **params,
        }
        
        resp = requests.post(api_url, json=payload, timeout=60)
        resp.raise_for_status()
        
        return resp.json()



