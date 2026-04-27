"""Natural language date parsing utilities for search results."""

from datetime import datetime, date, timedelta
from typing import Optional, Union
import re
import logging
from dateutil import parser as dateutil_parser
from dateutil.tz import tzutc

logger = logging.getLogger(__name__)


def parse_date_string(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse various date string formats into a datetime object.
    
    Handles formats from:
    - SerpAPI: "07/22/2025, 06:39 PM, +0000 UTC", "14 hours ago", "2 days ago"
    - MediaStack: "2025-07-23T18:21:49+00:00"
    - General: ISO 8601, RFC formats, relative dates
    
    Args:
        date_str: Raw date string from search API
        
    Returns:
        Parsed datetime object with timezone info, or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    try:
        # Handle relative dates first (SerpAPI organic results)
        relative_match = _parse_relative_date(date_str)
        if relative_match:
            return relative_match
        
        # Handle SerpAPI news format: "07/22/2025, 06:39 PM, +0000 UTC"
        serpapi_match = _parse_serpapi_news_format(date_str)
        if serpapi_match:
            return serpapi_match
        
        # Use dateutil for standard formats (ISO 8601, RFC, etc.)
        # This handles MediaStack format: "2025-07-23T18:21:49+00:00"
        parsed = dateutil_parser.parse(date_str)
        
        # Ensure timezone awareness
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tzutc())
        
        return parsed
        
    except Exception as e:
        logger.warning(f"Failed to parse date string '{date_str}': {e}")
        return None


def _parse_relative_date(date_str: str) -> Optional[datetime]:
    """
    Parse relative date strings like "14 hours ago", "2 days ago", "1 week ago".
    
    Args:
        date_str: Relative date string
        
    Returns:
        Datetime object representing the relative time, or None if not a relative date
    """
    date_str_lower = date_str.lower()
    now = datetime.now(tzutc())
    
    # Pattern: number + time unit + "ago"
    patterns = [
        (r'(\d+)\s+hours?\s+ago', lambda n: now - timedelta(hours=int(n))),
        (r'(\d+)\s+days?\s+ago', lambda n: now - timedelta(days=int(n))),
        (r'(\d+)\s+weeks?\s+ago', lambda n: now - timedelta(weeks=int(n))),
        (r'(\d+)\s+months?\s+ago', lambda n: now - timedelta(days=int(n) * 30)),  # Approximate
        (r'(\d+)\s+years?\s+ago', lambda n: now - timedelta(days=int(n) * 365)),  # Approximate
        (r'(\d+)\s+minutes?\s+ago', lambda n: now - timedelta(minutes=int(n))),
    ]
    
    for pattern, calc_func in patterns:
        match = re.search(pattern, date_str_lower)
        if match:
            return calc_func(match.group(1))
    
    # Handle special cases
    if 'yesterday' in date_str_lower:
        return now - timedelta(days=1)
    elif 'today' in date_str_lower:
        return now
    elif 'just now' in date_str_lower or 'moments ago' in date_str_lower:
        return now
    
    return None


def _parse_serpapi_news_format(date_str: str) -> Optional[datetime]:
    """
    Parse SerpAPI news format: "07/22/2025, 06:39 PM, +0000 UTC"
    
    Args:
        date_str: SerpAPI news date string
        
    Returns:
        Parsed datetime object or None if format doesn't match
    """
    # Pattern for SerpAPI news: MM/DD/YYYY, HH:MM AM/PM, +HHMM UTC
    pattern = r'(\d{2}/\d{2}/\d{4}),\s+(\d{1,2}:\d{2}\s+[AP]M),\s+([+-]\d{4})\s+UTC'
    match = re.match(pattern, date_str)
    
    if not match:
        return None
    
    date_part, time_part, tz_offset = match.groups()
    
    try:
        # Combine date and time parts
        datetime_str = f"{date_part} {time_part}"
        
        # Parse the datetime
        dt = datetime.strptime(datetime_str, "%m/%d/%Y %I:%M %p")
        
        # Apply timezone offset
        # Convert offset like "+0000" to hours
        offset_hours = int(tz_offset[:3])  # +00 or -05
        offset_minutes = int(tz_offset[3:])  # 00
        
        if tz_offset.startswith('-'):
            offset_minutes = -offset_minutes
        
        total_offset = timedelta(hours=offset_hours, minutes=offset_minutes)
        
        # Apply UTC timezone and adjust for offset
        dt_utc = dt.replace(tzinfo=tzutc()) - total_offset
        
        return dt_utc
        
    except ValueError as e:
        logger.warning(f"Failed to parse SerpAPI news format '{date_str}': {e}")
        return None


def format_parsed_date(dt: Optional[datetime], format_type: str = "readable") -> Optional[str]:
    """
    Format a parsed datetime object for display.
    
    Args:
        dt: Datetime object to format
        format_type: Format type - "readable", "iso", "short", or "relative"
        
    Returns:
        Formatted date string or None if input is None
    """
    if not dt:
        return None
    
    if format_type == "iso":
        return dt.isoformat()
    elif format_type == "short":
        return dt.strftime("%Y-%m-%d")
    elif format_type == "relative":
        return _format_relative_time(dt)
    else:  # readable
        return dt.strftime("%B %d, %Y at %I:%M %p UTC")


def _format_relative_time(dt: datetime) -> str:
    """
    Format datetime as relative time (e.g., "2 hours ago").
    
    Args:
        dt: Datetime to format
        
    Returns:
        Relative time string
    """
    now = datetime.now(tzutc())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzutc())
    
    diff = now - dt
    
    if diff.days > 0:
        if diff.days == 1:
            return "1 day ago"
        else:
            return f"{diff.days} days ago"
    
    hours = diff.seconds // 3600
    if hours > 0:
        if hours == 1:
            return "1 hour ago"
        else:
            return f"{hours} hours ago"
    
    minutes = (diff.seconds % 3600) // 60
    if minutes > 0:
        if minutes == 1:
            return "1 minute ago"
        else:
            return f"{minutes} minutes ago"
    
    return "Just now"