import os
import attrs
from typing import Union
from openai import OpenAI

from .base_llm import ModelAPIProtocol

def get_client():
    """Lazy initialization of OpenAI client."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3  # SDK handles retries with exponential backoff
    )

OAIChatPrompt = list[dict[str, str]]
OAIBasePrompt = Union[str, list[str]]

@attrs.define
class OpenAIModel(ModelAPIProtocol):
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        raise NotImplementedError
    
    def __call__(self, model_ids, prompt, num_candidates, max_attempts=3, **kwargs):
        # OpenAI SDK handles retries automatically via max_retries parameter
        return self._make_api_call(prompt, model_ids, num_candidates, **kwargs)
    
class OpenAIChatModel(OpenAIModel):
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        client = get_client()
        
        # Handle gpt-5 models which have different parameter requirements
        if model_id and model_id.startswith('gpt-5'):
            # gpt-5 models require max_completion_tokens instead of max_tokens
            if 'max_tokens' in params:
                params['max_completion_tokens'] = params.pop('max_tokens')
            # gpt-5 models only support default temperature (1.0), remove temperature if not 1.0
            if 'temperature' in params and params.get('temperature') != 1.0:
                params.pop('temperature')
        
        api_response = client.chat.completions.create(messages=prompt, model=model_id, **params)
        return api_response.choices[0].message.content