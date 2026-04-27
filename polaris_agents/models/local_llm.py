import logging
import os

import sglang as sgl
from transformers import AutoTokenizer

# Set environment variable to help with CUDA memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from sglang.srt.reasoning_parser import ReasoningParser
except ImportError:
    from sglang.srt.parser.reasoning_parser import ReasoningParser

def strip_thinking_token(text: str, model: str = "gpt-oss") -> str:
    """Strip thinking from text.

    Args:
        text: The text to strip thinking from.
        model: The model to use for reasoning parsing. Default is "gpt-oss".
            See https://docs.sglang.ai/advanced_features/separate_reasoning.html for an updated list of parsers.
    
    Returns:
        The text with thinking tokens stripped.
    """
    reasoning_parser = ReasoningParser(model)
    reasoning_text, response = reasoning_parser.parse_non_stream(text)

    return response

def init_local_model(model_path: str, tp_size: int, logger: logging.Logger) -> tuple[sgl.Engine, AutoTokenizer]:
    """Initialize a local model with path model_path and tensor parallel size tp_size.

    Args:
        model_path: Path to the local model directory.
        tp_size: Tensor parallel size.

    Returns:
        Tuple of (llm, tokenizer) where llm is the SGLang engine and tokenizer is the AutoTokenizer.
    """
    logger.info(f"Initializing local model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            trust_remote_code=True,
            disable_cuda_graph=True,
        )
        logger.info("Local model initialized.")
    except Exception as e:
        logger.error(f"Error initializing local model: {e}")
        exit(1)

    return llm, tokenizer


def call_local_model(
    llm: sgl.Engine,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    temperature: float = 0,
    max_new_tokens: int = 4096,
    enable_thinking: bool = True,
    gpt_oss_reasoning_level: str | None = "high",
    strip_thinking_model: str = "gpt-oss",
    logger: logging.Logger = None,
) -> str:
    """Calls a local model using SGLang for evaluation.
    
    Args:
        llm: The SGLang engine to call.
        tokenizer: The tokenizer for the local model.
        messages: List of message dictionaries with 'role' and 'content' keys.
        temperature: The temperature for the local model.
        max_new_tokens: The maximum number of new tokens to generate.
        enable_thinking: Whether to enable thinking for the local model.
        gpt_oss_reasoning_level: The reasoning level for gpt-oss injected into the system prompt.
            Default is "high". If None, will not be injected.
        strip_thinking_model: Model to use for stripping thinking tokens.
    
    Returns:
        The response from the local model.
    """
    LOCAL_MODEL_PARAMS = {
        "temperature": temperature,  # greedy
        "max_new_tokens": max_new_tokens,
        "n": 1,
    }

    if gpt_oss_reasoning_level:
        if messages[0]["role"] == "system":
            messages[0]["content"] += f"\nReasoning: {gpt_oss_reasoning_level}"
        else:
            messages.insert(0, {"role": "system", "content": f"\nReasoning: {gpt_oss_reasoning_level}"})
    try:
        # # First attempt: with system prompt
        # chat_prompt_structure = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt},
        # ]

        # Convert format based on enable_thinking
        try:
            if enable_thinking:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
        except TypeError:
            # Fallback for models that don't support enable_thinking parameter
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )


    except Exception as e:
        logger.warning(
            f"Error with system prompt. Concatenating user prompt to system prompt and trying again. Original error message: {e}"
        )
        unified_prompt = f"{messages[0]['content']}\n{messages[1]['content']}"

        # Second attempt: without system prompt
        unified_messages = [{"role": "user", "content": unified_prompt}]

        # Try with enable_thinking=False for Qwen3 compatibility
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                unified_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback for models that don't support enable_thinking parameter
            formatted_prompt = tokenizer.apply_chat_template(
                unified_messages, tokenize=False, add_generation_prompt=True
            )

    outputs = llm.generate(
        [formatted_prompt],
        sampling_params=LOCAL_MODEL_PARAMS,
        return_logprob=False,
    )
    response = strip_thinking_token(
        outputs[0]["text"], model=strip_thinking_model
    )

    return response