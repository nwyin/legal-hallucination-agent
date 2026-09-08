import os
import attrs
from typing import Union
from openai import OpenAI

from .base_llm import ModelAPIProtocol

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