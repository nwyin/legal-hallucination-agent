"""
Common utilities for agent operations.
"""

import json
import logging
import re
from typing import Any, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON string from text, handling code blocks and standalone JSON.
    Uses proper brace matching to handle nested JSON objects.
    
    Args:
        text: Text that may contain JSON (possibly in code blocks or with extra text)
        
    Returns:
        Extracted JSON string, or None if no valid JSON found
    """
    json_str = None
    
    # Try to find JSON in code block first
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?)\s*```', text, flags=re.IGNORECASE | re.DOTALL)
    if code_block_match:
        # Extract JSON with proper brace matching
        brace_start = code_block_match.start(1)
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[brace_start:i + 1]
                    break
    
    # If no code block, try standalone JSON with brace matching
    if not json_str:
        brace_start = text.find('{')
        if brace_start != -1:
            brace_count = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:i + 1]
                        break
    
    return json_str


def parse_json_from_text(text: str, response_name: str = "response") -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from text that may contain code blocks or extra text.
    Uses proper brace matching to handle nested JSON objects.
    
    Args:
        text: Text that may contain JSON (possibly in code blocks or with extra text)
        response_name: Name of the response type for logging
        
    Returns:
        Parsed JSON data as dictionary, or None if no valid JSON found
        
    Raises:
        ValueError: If JSON is found but invalid
    """
    json_str = extract_json_from_text(text)
    
    if not json_str:
        return None
    
    try:
        data = json.loads(json_str)
        logger.debug(f"Successfully extracted and parsed {response_name} from text")
        return data
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues before giving up
        # Fix trailing commas
        json_str_fixed = json_str.replace(',}', '}').replace(',]', ']')
        
        # Try parsing again with fixed trailing commas
        try:
            data = json.loads(json_str_fixed)
            logger.debug(f"Successfully parsed {response_name} after fixing trailing commas")
            return data
        except json.JSONDecodeError:
            # If still failing, raise the original error with context
            raise ValueError(f"Invalid JSON in {response_name} response: {e}")


def parse_json_response(response: str, response_name: str = "response") -> Dict[str, Any]:
    """
    Parse a JSON response with common cleanup and error handling.
    
    Args:
        response: Raw response string from LLM
        response_name: Name of the response type for logging (e.g., "prediction")
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        ValueError: If response is empty or invalid JSON
        KeyError: If required fields are missing
    """
    if not response or response.strip() == "":
        raise ValueError(f"Empty {response_name} response from model")
    
    logger.debug(f"Raw {response_name} response: '{response}'")
    
    # First try the robust extraction method (handles code blocks, extra text, etc.)
    try:
        data = parse_json_from_text(response, response_name)
        if data is not None:
            return data
    except ValueError:
        # If extraction found JSON but it's malformed, try fallback cleanup
        pass
    
    # Fallback to simple cleanup for cases where extraction didn't find JSON or parsing failed
    # Clean up markdown formatting if present
    response_clean = response.strip()
    if response_clean.startswith("```json"):
        response_clean = response_clean[7:]
    if response_clean.endswith("```"):
        response_clean = response_clean[:-3]
    response_clean = response_clean.strip()
    
    # Fix common JSON issues (trailing commas, etc.)
    response_clean = response_clean.replace(',}', '}').replace(',]', ']')
    
    try:
        data = json.loads(response_clean)
        logger.debug(f"Successfully parsed {response_name}")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {response_name} response: {e}")


def parse_prediction_response(response: str) -> Tuple[str, float]:
    """
    Parse a prediction response and return prediction and confidence.
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Tuple of (prediction, confidence)
        
    Raises:
        ValueError: If response is empty or invalid JSON
        KeyError: If required fields are missing
    """
    data = parse_json_response(response, "prediction")
    
    action = data.get('action', {})
    raw_prediction = action.get('response', '')
    # Handle both str and list (e.g. hallucination checker returns JSON array)
    if isinstance(raw_prediction, list):
        prediction = json.dumps(raw_prediction)  # Preserve as JSON string for downstream parsing
    else:
        prediction = str(raw_prediction).strip() if raw_prediction else ''
    confidence = float(data.get('confidence', 0.0))
    
    if not prediction:
        raise ValueError("Empty prediction in response")

    pred_preview = str(prediction)[:200] + "..." if len(str(prediction)) > 200 else str(prediction)
    logger.info(f"Parsed prediction: {pred_preview} (confidence: {confidence:.3f})")
    return prediction, confidence


def extract_action_parameters(action: Any) -> Dict[str, Any]:
    """
    Extract parameters from an action using the action class `inputs` schema.
    
    Args:
        action: Action instance to extract parameters from
        
    Returns:
        Dictionary of parameter names to values (only non-None values)
        
    Rules:
        - Include all fields defined in the `inputs` schema (required and optional)
        - For each field, read the value from the action instance when set (not None)
        - Omit fields that are unset (None) to avoid redundant defaults
    """
    params: Dict[str, Any] = {}
    inputs_schema = getattr(action.__class__, 'inputs', None)
    if not isinstance(inputs_schema, dict):
        return params
        
    for name, _spec in inputs_schema.items():
        if hasattr(action, name):
            value = getattr(action, name)
            if value is not None:
                params[name] = value
                
    return params


def normalize_yes_no_answer(answer: str) -> str:
    """
    Normalize Yes/No answer variations to standard format.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer ("Yes" or "No")
        
    Raises:
        ValueError: If answer is not a recognized Yes/No variation
    """
    answer_upper = answer.upper().strip()
    if answer_upper in ['YES', 'Y']:
        return 'Yes'
    elif answer_upper in ['NO', 'N']:
        return 'No'
    else:
        raise ValueError(f"Invalid answer format: '{answer}'. Expected 'Yes' or 'No'")


def normalize_yes_no_answers(answers: str) -> str:
    """
    Normalize semicolon-delimited Yes/No answers.
    
    Args:
        answers: Semicolon-delimited answers string (e.g., "Yes; No; Yes")
        
    Returns:
        Normalized semicolon-delimited answers
        
    Raises:
        ValueError: If any answer is not a recognized Yes/No variation
    """
    answer_list = [ans.strip() for ans in answers.split(';')]
    normalized_answers = [normalize_yes_no_answer(answer) for answer in answer_list]
    return '; '.join(normalized_answers)
