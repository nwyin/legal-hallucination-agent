import os
import time
import attrs
from typing import Union
from google import genai
from google.genai import types

from .base_llm import ModelAPIProtocol


def get_client():
    """Lazy initialization of Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)

GeminiChatPrompt = list[dict[str, str]]
GeminiBasePrompt = Union[str, list[str]]

@attrs.define
class GeminiModel(ModelAPIProtocol):
    
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        raise NotImplementedError
    
    def __call__(self, model_ids, prompt, num_candidates, max_attempts=3, **kwargs):
        for attempt in range(max_attempts):
            try:
                responses = self._make_api_call(prompt, model_ids, num_candidates, **kwargs)
                return responses
            except Exception as e:
                if attempt >= max_attempts - 1:
                    raise
                # Exponential backoff for Gemini (SDK doesn't handle retries properly)
                time.sleep(2 ** attempt)
    
class GeminiChatModel(GeminiModel):
    def _make_api_call(self, prompt, model_id, num_candidates, **params):
        client = get_client()
        
        # Convert OpenAI-style messages to Gemini format
        system_instruction = None
        contents = []
        
        if isinstance(prompt, list) and len(prompt) > 0:
            # Handle chat format - collect system content and build contents list
            for msg in prompt:
                if msg.get("role") == "user":
                    contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
                elif msg.get("role") == "assistant":
                    contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
                elif msg.get("role") == "system":
                    if system_instruction is None:
                        system_instruction = msg["content"]
                    else:
                        system_instruction += "\n\n" + msg["content"]
        
        # Build GenerateContentConfig with all parameters, including seed (properly supported in new SDK)
        # System instruction is included in the config for the new SDK
        generation_config = None
        config_kwargs = {}
        
        # Add generation parameters
        if 'seed' in params:
            config_kwargs['seed'] = params['seed']
        if 'temperature' in params:
            config_kwargs['temperature'] = params['temperature']
        if 'max_tokens' in params or 'max_output_tokens' in params:
            max_tokens = params.get('max_tokens') or params.get('max_output_tokens')
            config_kwargs['max_output_tokens'] = max_tokens
        if 'top_p' in params:
            config_kwargs['top_p'] = params['top_p']
        if 'top_k' in params:
            config_kwargs['top_k'] = params['top_k']
        
        # Add system instruction if available
        if system_instruction:
            config_kwargs['system_instruction'] = types.Content(role="user", parts=[types.Part(text=system_instruction)])
        
        # Disable Automatic Function Calling (AFC) - we control tools at the agent level
        config_kwargs['automatic_function_calling'] = types.AutomaticFunctionCallingConfig(disable=True)
        
        if config_kwargs or isinstance(prompt, list):
            generation_config = types.GenerateContentConfig(**config_kwargs)
        
        if isinstance(prompt, list) and len(prompt) > 0:
            # Use the chat API
            chat = client.chats.create(
                model=model_id,
                config=generation_config,
                history=contents[:-1] if len(contents) > 1 else None
            )
            
            # Send the last user message (if there are messages)
            if contents:
                last_message = contents[-1]
                if last_message.role == "user":
                    # Extract text from parts
                    message_text = last_message.parts[0].text if last_message.parts else ""
                    response = chat.send_message(message_text, config=generation_config)
                    return response.text
                else:
                    # If last message is assistant, send an empty message to get next response
                    response = chat.send_message("", config=generation_config)
                    return response.text
            return ""
        else:
            # Handle simple text prompt
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=generation_config
            )
            return response.text