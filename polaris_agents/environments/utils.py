import re
import string
import random
import numpy as np
import datetime
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def set_random_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: The seed value to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\s+', '', text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer: str, key: str):
    return normalize_answer(answer) == normalize_answer(key)


def get_timestamp() -> str:
    return datetime.datetime.now().isoformat()


def parse_date_to_timestamp(date_string: str) -> Optional[int]:
    """
    Parse a date string in various formats to a Unix timestamp (midnight UTC).
    
    The timestamp represents midnight UTC on the given date. When used as a 
    "before" filter, this excludes documents published on that day or later,
    which prevents data leakage of resolution-day content.
    
    Args:
        date_string: Date string in one of the supported formats
        
    Returns:
        Unix timestamp (int) for midnight UTC on that date, or None if parsing fails
        
    Supported formats:
        - "%Y-%m-%d" (ISO format: 2025-08-14)
        - "%b %d %Y" (Short month: Jun 18 2025)
        - "%B %d, %Y" (Full month: June 18, 2025)
        - "%m/%d/%Y" (US format: 06/18/2025)
    """
    if not date_string:
        return None
    
    date_formats = [
        "%Y-%m-%d",     # ISO format: 2025-08-14
        "%b %d %Y",     # Short month: Jun 18 2025
        "%B %d, %Y",    # Full month: June 18, 2025
        "%m/%d/%Y",     # US format: 06/18/2025
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.datetime.strptime(date_string, fmt)
            # Treat as midnight UTC (not local time) to match document timestamps
            # This ensures consistent filtering regardless of server timezone
            return int(parsed_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        except ValueError:
            continue
    
    return None