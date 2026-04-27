import logging
import os
import time
import attrs
from typing import Union
from openai import AzureOpenAI
from portkey_ai import Portkey

from .base_llm import ModelAPIProtocol

logger = logging.getLogger(__name__)

def get_client():
    # return AzureOpenAI(
    #     api_key=os.environ['AI_SANDBOX_KEY'],
    #     azure_endpoint=os.environ.get("SANDBOX_ENDPOINT", "https://api-ai-sandbox.princeton.edu/"),
    #     api_version=os.environ.get("SANDBOX_API_VERSION", "2024-02-01")
    # )
    return Portkey(api_key=os.environ['AI_SANDBOX_KEY'])

SandboxChatPrompt = list[dict[str, str]]
SandboxBasePrompt = Union[str, list[str]]

@attrs.define
class SandboxModel(ModelAPIProtocol):
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        raise NotImplementedError
    
    def __call__(self, model_ids, prompt, num_candidates, max_attempts=3, **kwargs):
        last_error = None
        for i in range(max_attempts):
            try:
                responses = self._make_api_call(prompt, model_ids, num_candidates, **kwargs)
                if responses is None:
                    raise ValueError("API returned None")
                return responses
            except Exception as e:
                last_error = e
                if i < max_attempts - 1:
                    time.sleep(60)
                else:
                    raise
        if last_error is not None:
            raise last_error
    
class SandboxChatModel(SandboxModel):
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        # gpt-5 via Azure has different parameter requirements
        if model_id and str(model_id).startswith("gpt-5"):
            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            if "temperature" in params and params.get("temperature") != 1.0:
                params.pop("temperature")
        client = get_client()
        api_response = client.chat.completions.create(messages=prompt, model=model_id, **params)
        choice = api_response.choices[0]
        content = choice.message.content
        if not content or not content.strip():
            logger.warning(
                "Empty content from %s. finish_reason=%s, message fields=%s",
                model_id,
                choice.finish_reason,
                {k: v for k, v in vars(choice.message).items() if v is not None},
            )
            return None  # caller/__call__ will treat as failure and retry or raise
        return content.strip()