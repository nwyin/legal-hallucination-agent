"""Open web search: date parsing, SerpAPI and MediaStack clients, and search entry points."""

import json
import logging
import os
import random
import re
import string
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from dateutil import parser as dateutil_parser
from dateutil.tz import tzutc
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# --- Date parsing for search result timestamps ---


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


# --- SerpAPI client ---


# Load environment variables from .env file



class SerpApiClient:
    """Client for interacting with SerpApi search services using direct requests."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize SerpApi client.
        
        Args:
            api_key: SerpApi API key. If not provided, will try to get from SERPAPI_API_KEY env var
            max_retries: Maximum number of retry attempts for rate-limited requests
            base_delay: Base delay in seconds for exponential backoff
        """
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SerpApi API key is required. Set SERPAPI_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://serpapi.com/search"
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def search(self, query: str, engine: str = "google", cutoff_date: Optional[date] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform a search using SerpApi.
        
        Args:
            query: Search query string
            engine: Search engine to use (default: google)
            cutoff_date: Optional cutoff date for search results
            **kwargs: Additional parameters for the search
            
        Returns:
            Dict containing search results
        """
        # Apply cutoff date by appending to query
        if cutoff_date:
            query = f"{query} before:{cutoff_date.strftime('%Y-%m-%d')}"
        
        params = {
            "q": query,
            "engine": engine,
            "api_key": self.api_key,
            **kwargs
        }
        
        return self._make_request_with_retry(self.base_url, params)
    
    def _make_request_with_retry(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry for rate limits.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Dict containing API response
            
        Raises:
            requests.RequestException: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json()
                
            except requests.HTTPError as e:
                last_exception = e
                
                # Check if it's a rate limit error (429)
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        # Calculate delay with exponential backoff and jitter
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"SerpApi rate limit hit (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"SerpApi rate limit exceeded after {self.max_retries + 1} attempts")
                        raise
                else:
                    # For non-rate-limit errors, don't retry
                    logger.error(f"SerpApi request failed with status {response.status_code}: {e}")
                    raise
                    
            except requests.RequestException as e:
                last_exception = e
                # For connection errors, we might want to retry
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"SerpApi connection error (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"SerpApi request failed after {self.max_retries + 1} attempts: {e}")
                    raise
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise requests.RequestException("Unknown error occurred during SerpApi request")
    
    def google_search(self, query: str, num_results: int = 10, cutoff_date: Optional[date] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform a Google search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            cutoff_date: Optional cutoff date for search results
            **kwargs: Additional Google search parameters
            
        Returns:
            Dict containing Google search results
        """
        return self.search(
            query=query,
            engine="google",
            num=num_results,
            cutoff_date=cutoff_date,
            **kwargs
        )

    def google_scholar_search(self, query, **kwargs) -> Dict[str, Any]:
        """
        Perform a Google Scholar search.

        Args:
            query: Search query
            num_results: Number of results to return
            cutoff_date: Optional cutoff date for search results
            **kwargs: Additional Google search parameters
            
        Returns:
            Dict containing Google search results
        """
        return self.search(
            query=query,
            engine="google_scholar",
            cutoff_date=None,
            **kwargs
        )
    
    def news_search(self, query: str, cutoff_date: Optional[date] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform a Google News search.
        
        Args:
            query: Search query
            cutoff_date: Optional cutoff date for search results
            **kwargs: Additional news search parameters
            
        Returns:
            Dict containing news search results
        """
        return self.search(
            query=query,
            engine="google_news",
            cutoff_date=cutoff_date,
            **kwargs
        )
    
    def extract_organic_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract organic search results from SerpApi response.
        
        Args:
            search_results: Raw SerpApi response
            
        Returns:
            List of organic search results
        """
        return search_results.get("organic_results", [])
    
    def extract_news_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract news results from SerpApi response.
        
        Args:
            search_results: Raw SerpApi response
            
        Returns:
            List of news results
        """
        return search_results.get("news_results", [])


# --- MediaStack client ---


# Load environment variables from .env file



class MediaStackClient:
    """Client for interacting with MediaStack news API."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize MediaStack client.
        
        Args:
            api_key: MediaStack API key. If not provided, will try to get from MEDIASTACK_API_KEY env var
            max_retries: Maximum number of retry attempts for rate-limited requests
            base_delay: Base delay in seconds for exponential backoff
        """
        self.api_key = api_key or os.getenv("MEDIASTACK_API_KEY")
        if not self.api_key:
            raise ValueError("MediaStack API key is required. Set MEDIASTACK_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "http://api.mediastack.com/v1"
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def search_news(self, 
                   keywords: str,
                   sources: Optional[str] = None,
                   categories: Optional[str] = None,
                   countries: Optional[str] = None,
                   languages: Optional[str] = None,
                   date_from: Optional[date] = None,
                   date_to: Optional[date] = None,
                   cutoff_date: Optional[date] = None,
                   sort: str = "published_desc",
                   limit: int = 25,
                   offset: int = 0) -> Dict[str, Any]:
        """
        Search for news articles.
        
        Args:
            keywords: Search keywords
            sources: Comma-separated list of news sources
            categories: Comma-separated list of categories
            countries: Comma-separated list of country codes
            languages: Comma-separated list of language codes
            date_from: Start date for search
            date_to: End date for search
            cutoff_date: Cutoff date for search results (uses 1980-01-01 as lower bound)
            sort: Sort order (published_desc, published_asc, popularity)
            limit: Number of results to return (max 100)
            offset: Offset for pagination
            
        Returns:
            Dict containing news search results
        """
        params = {
            "access_key": self.api_key,
            "keywords": keywords,
            "sort": sort,
            "limit": min(limit, 100),
            "offset": offset
        }
        
        if sources:
            params["sources"] = sources
        if categories:
            params["categories"] = categories
        if countries:
            params["countries"] = countries
        if languages:
            params["languages"] = languages
        
        # Handle date parameters with cutoff_date priority
        if cutoff_date:
            # Use 1980-01-01 as lower bound and cutoff_date as upper bound
            lower_bound = date(1980, 1, 1)
            params["date"] = f"{lower_bound.strftime('%Y-%m-%d')},{cutoff_date.strftime('%Y-%m-%d')}"
        elif date_from and date_to:
            params["date"] = f"{date_from.strftime('%Y-%m-%d')},{date_to.strftime('%Y-%m-%d')}"
        elif date_from:
            params["date"] = date_from.strftime("%Y-%m-%d")
        
        return self._make_request_with_retry(f"{self.base_url}/news", params)
    
    def get_sources(self, 
                   countries: Optional[str] = None,
                   categories: Optional[str] = None,
                   languages: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available news sources.
        
        Args:
            countries: Comma-separated list of country codes
            categories: Comma-separated list of categories
            languages: Comma-separated list of language codes
            
        Returns:
            Dict containing available news sources
        """
        params = {"access_key": self.api_key}
        
        if countries:
            params["countries"] = countries
        if categories:
            params["categories"] = categories
        if languages:
            params["languages"] = languages
        
        return self._make_request_with_retry(f"{self.base_url}/sources", params)
    
    def _make_request_with_retry(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry for rate limits.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Dict containing API response
            
        Raises:
            requests.RequestException: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json()
                
            except requests.HTTPError as e:
                last_exception = e
                
                # Check if it's a rate limit error (429)
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        # Calculate delay with exponential backoff and jitter
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"MediaStack rate limit hit (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"MediaStack rate limit exceeded after {self.max_retries + 1} attempts")
                        raise
                else:
                    # For non-rate-limit errors, don't retry
                    logger.error(f"MediaStack request failed with status {response.status_code}: {e}")
                    raise
                    
            except requests.RequestException as e:
                last_exception = e
                # For connection errors, we might want to retry
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"MediaStack connection error (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"MediaStack request failed after {self.max_retries + 1} attempts: {e}")
                    raise
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise requests.RequestException("Unknown error occurred during MediaStack request")
    
    def extract_articles(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract articles from MediaStack response.
        
        Args:
            search_results: Raw MediaStack response
            
        Returns:
            List of news articles
        """
        return search_results.get("data", [])
    
    def filter_by_date_range(self, 
                           articles: List[Dict[str, Any]], 
                           start_date: date, 
                           end_date: date) -> List[Dict[str, Any]]:
        """
        Filter articles by date range.
        
        Args:
            articles: List of articles
            start_date: Start date
            end_date: End date
            
        Returns:
            Filtered list of articles
        """
        filtered = []
        for article in articles:
            if article.get("published_at"):
                try:
                    pub_date = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00")).date()
                    if start_date <= pub_date <= end_date:
                        filtered.append(article)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid date format in article: {article.get('published_at')}")
                    continue
        return filtered
    
    def get_categories(self) -> List[str]:
        """
        Get list of available categories.
        
        Returns:
            List of category names
        """
        return [
            "general", "business", "entertainment", "health",
            "science", "sports", "technology"
        ]


# --- Search results and search entry points ---


def _filter_by_cutoff_date(results: List['SearchResult'], cutoff_date: Optional[date], exclude_undated: bool = True) -> List['SearchResult']:
    """
    Filter search results to exclude those published after the cutoff date.
    
    Args:
        results: List of SearchResult objects
        cutoff_date: Optional cutoff date for filtering
        exclude_undated: If True, exclude results without publication dates (default: True)
        
    Returns:
        Filtered list of SearchResult objects
    """
    filtered_results = []
    for result in results:
        if result.published_date:
            # If we have a cutoff date, check if the result is within the cutoff
            if cutoff_date:
                # Convert cutoff_date to datetime for comparison
                cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time())
                cutoff_datetime = cutoff_datetime.replace(tzinfo=result.published_date.tzinfo or datetime.now().astimezone().tzinfo)
                
                # Only include results published strictly before the cutoff date
                if result.published_date < cutoff_datetime:
                    filtered_results.append(result)
            else:
                # No cutoff date, include all results with dates
                filtered_results.append(result)
        else:
            # Handle results without dates based on exclude_undated setting
            if not exclude_undated:
                filtered_results.append(result)
    
    return filtered_results


class SearchResult:
    """Standardized search result format."""
    
    def __init__(self, 
                 title: str,
                 url: str,
                 snippet: str,
                 source: str,
                 published_date: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 result_id: Optional[str] = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date_raw = published_date  # Store original string
        self.published_date = parse_date_string(published_date)  # Parse to datetime
        self.metadata = metadata or {}
        self.result_id = result_id or self._generate_result_id()
    
    def _generate_result_id(self) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    
    def __repr__(self) -> str:
        if self.published_date:
            # Format as readable date
            date_str = f" ({format_parsed_date(self.published_date, 'readable')})"
        else:
            date_str = ""
        return f"{self.title}{date_str}\n{self.snippet}\n{self.url}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "published_date_raw": self.published_date_raw,
            "metadata": self.metadata
        }


def search_results_xml(results: List[SearchResult]) -> str:
    """
    Convert search results to XML format for LLM consumption.
    
    Args:
        results: List of SearchResult objects
        
    Returns:
        XML string with search results formatted for LLM
    """
    if not results:
        return "<search_results></search_results>"
    
    xml_parts = ["<search_results>"]
    
    for result in results:
        # Format date for display
        date_str = ""
        if result.published_date:
            date_str = result.published_date.strftime("%Y-%m-%d")
        
        # Escape XML special characters in text content
        title = _escape_xml(result.title)
        description = _escape_xml(result.snippet)
        url = _escape_xml(result.url)
        
        xml_parts.append(f"  <result>")
        xml_parts.append(f"    <result_id>{result.result_id}</result_id>")
        xml_parts.append(f"    <title>{title}</title>")
        xml_parts.append(f"    <date>{date_str}</date>")
        xml_parts.append(f"    <description>{description}</description>")
        xml_parts.append(f"    <url>{url}</url>")
        xml_parts.append(f"  </result>")
    
    xml_parts.append("</search_results>")
    
    return "\n".join(xml_parts)


def _escape_xml(text: str) -> str:
    if not text:
        return ""
    
    # Replace XML special characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    
    return text


def _save_search_history(results: List[SearchResult], query: str, search_type: str) -> None:
    """
    Save search results to data/search_history.json file.
    
    Args:
        results: List of SearchResult objects
        query: The search query used
        search_type: Type of search performed
    """
    history_file = "data/search_history.json"
    
    # Load existing history or create new
    history = {}
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load search history: {e}")
            history = {}
    
    # Add new results to history
    for result in results:
        history[result.result_id] = {
            **result.to_dict(),
            "search_query": query,
            "search_type": search_type,
            "timestamp": datetime.now().isoformat()
        }
    
    # Save updated history
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save search history: {e}")


def search(query: str, 
          search_type: str = "web",
          num_results: int = 10,
          cutoff_date: Optional[date] = None,
          news_source: str = "serpapi",
          exclude_undated: bool = False,
          **kwargs) -> List[SearchResult]:
    """
    Unified search function that can use different search backends.
    
    Args:
        query: Search query string
        search_type: Type of search ("web", "news", "google_scholar")
        num_results: Number of results to return
        cutoff_date: Optional cutoff date for search results
        news_source: News source to use ("serpapi" or "mediastack", default: "serpapi")
        exclude_undated: If True, exclude results without publication dates.
                        Default: False (include all results, filter at environment level if needed)
        **kwargs: Additional parameters passed to specific search clients
        
    Returns:
        List of SearchResult objects
    """
    # Perform the search
    if search_type == "web":
        results = _web_search(query, num_results, cutoff_date=cutoff_date, exclude_undated=exclude_undated, **kwargs)
    elif search_type == "news":
        results = _news_search(query, num_results, cutoff_date=cutoff_date, news_source=news_source, exclude_undated=exclude_undated, **kwargs)
    elif search_type == "google_scholar":
        results = _scholar_search(query, num_results, exclude_undated=exclude_undated, **kwargs)
    else:
        raise ValueError(f"Unsupported search type: {search_type}")
    
    # Save search history
    # if results:
    #     _save_search_history(results, query, search_type)
    
    return results


def _web_search(query: str, num_results: int, cutoff_date: Optional[date] = None, exclude_undated: bool = True, **kwargs) -> List[SearchResult]:
    """
    Perform web search using SerpApi.
    
    Args:
        query: Search query
        num_results: Number of results to return
        cutoff_date: Optional cutoff date for search results
        **kwargs: Additional parameters for SerpApi
        
    Returns:
        List of SearchResult objects
    """
    try:
        client = SerpApiClient()
        results = client.google_search(query, num_results=num_results, cutoff_date=cutoff_date, **kwargs)
        organic_results = client.extract_organic_results(results)
        
        search_results = []
        for result in organic_results:            
            # Check for any date-related fields (though they're typically not present in web results)
            published_date = result.get("date") or result.get("publication_date") or result.get("published_at")
            
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("link", ""),
                snippet=result.get("snippet", ""),
                source="google",
                published_date=published_date,
                metadata={
                    "position": result.get("position"),
                    "displayed_link": result.get("displayed_link"),
                    "favicon": result.get("favicon"),
                    "source_info": result.get("source"),
                    "snippet_highlighted_words": result.get("snippet_highlighted_words", [])
                }
            )
            search_results.append(search_result)
        
        # Filter by cutoff date after parsing dates
        filtered_results = _filter_by_cutoff_date(search_results, cutoff_date, exclude_undated)
        
        # Return requested number of results after filtering
        return filtered_results[:num_results]
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def _news_search(query: str, num_results: int, cutoff_date: Optional[date] = None, news_source: str = "serpapi", exclude_undated: bool = True, **kwargs) -> List[SearchResult]:
    """
    Perform news search using SerpAPI or MediaStack.
    
    Args:
        query: Search query
        num_results: Number of results to return
        cutoff_date: Optional cutoff date for search results
        news_source: News source to use ("serpapi" or "mediastack")
        **kwargs: Additional parameters for the search client
        
    Returns:
        List of SearchResult objects
    """
    if news_source == "serpapi":
        return _serpapi_news_search(query, num_results, cutoff_date, exclude_undated, **kwargs)
    elif news_source == "mediastack":
        return _mediastack_news_search(query, num_results, cutoff_date, exclude_undated, **kwargs)
    else:
        raise ValueError(f"Unsupported news source: {news_source}")


def _serpapi_news_search(query: str, num_results: int, cutoff_date: Optional[date] = None, exclude_undated: bool = True, **kwargs) -> List[SearchResult]:
    """
    Perform news search using SerpAPI.
    
    Args:
        query: Search query
        num_results: Number of results to return
        cutoff_date: Optional cutoff date for search results
        **kwargs: Additional parameters for SerpAPI
        
    Returns:
        List of SearchResult objects
    """
    try:
        client = SerpApiClient()
        results = client.news_search(query, cutoff_date=cutoff_date, **kwargs)
        news_articles = client.extract_news_results(results)
        
        search_results = []
        for article in news_articles:
            # Extract source name from the source dict
            source_info = article.get("source", {})
            source_name = source_info.get("name", "") if isinstance(source_info, dict) else str(source_info)
            
            # Use title as snippet for news results since SerpAPI news doesn't provide separate snippets
            snippet = article.get("snippet", "") or article.get("title", "")
            
            search_result = SearchResult(
                title=article.get("title", ""),
                url=article.get("link", ""),
                snippet=snippet,
                source=source_name,
                published_date=article.get("date"),  # SerpAPI format: "07/22/2025, 01:10 PM, +0000 UTC"
                metadata={
                    "position": article.get("position"),
                    "thumbnail": article.get("thumbnail"),
                    "thumbnail_small": article.get("thumbnail_small"),
                    "authors": source_info.get("authors", []) if isinstance(source_info, dict) else [],
                    "source_icon": source_info.get("icon", "") if isinstance(source_info, dict) else ""
                }
            )
            search_results.append(search_result)
        
        # Filter by cutoff date after parsing dates
        filtered_results = _filter_by_cutoff_date(search_results, cutoff_date, exclude_undated)
        
        # Return requested number of results after filtering
        return filtered_results[:num_results]
        
    except Exception as e:
        logger.error(f"SerpAPI news search failed: {e}")
        return []


def _mediastack_news_search(query: str, num_results: int, cutoff_date: Optional[date] = None, exclude_undated: bool = True, **kwargs) -> List[SearchResult]:
    """
    Perform news search using MediaStack.
    
    Args:
        query: Search query
        num_results: Number of results to return
        cutoff_date: Optional cutoff date for search results
        **kwargs: Additional parameters for MediaStack
        
    Returns:
        List of SearchResult objects
    """
    try:
        client = MediaStackClient()
        results = client.search_news(keywords=query, limit=num_results, cutoff_date=cutoff_date, **kwargs)
        articles = client.extract_articles(results)
        
        search_results = []
        for article in articles:
            search_result = SearchResult(
                title=article.get("title", ""),
                url=article.get("url", ""),
                snippet=article.get("description", ""),
                source=article.get("source", ""),
                published_date=article.get("published_at"),  # MediaStack format: "2025-07-23T17:52:56+00:00"
                metadata={
                    "author": article.get("author"),
                    "category": article.get("category"),
                    "country": article.get("country"),
                    "language": article.get("language"),
                    "image": article.get("image")
                }
            )
            search_results.append(search_result)
        
        # Filter by cutoff date after parsing dates
        filtered_results = _filter_by_cutoff_date(search_results, cutoff_date, exclude_undated)
        
        # Return requested number of results after filtering
        return filtered_results[:num_results]
        
    except Exception as e:
        logger.error(f"MediaStack news search failed: {e}")
        return []


def multi_source_search(query: str, 
                       sources: List[str] = None,
                       num_results_per_source: int = 5,
                       cutoff_date: Optional[date] = None,
                       news_source: str = "serpapi",
                       exclude_undated: bool = True) -> Dict[str, List[SearchResult]]:
    """
    Search across multiple sources and return results grouped by source.
    
    Args:
        query: Search query
        sources: List of sources to search ("web", "news", "google_scholar")
        num_results_per_source: Number of results per source
        cutoff_date: Optional cutoff date for search results
        news_source: News source to use for news searches ("serpapi" or "mediastack")
        exclude_undated: If True, exclude results without publication dates (default: True)
        
    Returns:
        Dict mapping source names to lists of SearchResult objects
    """
    if sources is None:
        sources = ["web", "news", "google_scholar"]
    
    results = {}
    for source in sources:
        try:
            source_results = search(query, search_type=source, num_results=num_results_per_source, cutoff_date=cutoff_date, news_source=news_source, exclude_undated=exclude_undated)
            results[source] = source_results
        except Exception as e:
            logger.error(f"Search failed for source {source}: {e}")
            results[source] = []
    
    return results


def combine_and_deduplicate_results(results_dict: Dict[str, List[SearchResult]]) -> List[SearchResult]:
    """
    Combine results from multiple sources and remove duplicates.
    
    Args:
        results_dict: Dict of source -> results mapping
        
    Returns:
        Combined and deduplicated list of SearchResult objects
    """
    seen_urls = set()
    combined_results = []
    
    for source, results in results_dict.items():
        for result in results:
            if result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                combined_results.append(result)
    
    return combined_results


def _scholar_search(query: str, num_results: int, exclude_undated: bool = True, **kwargs) -> List[SearchResult]:
    """
    Perform google scholar search using SerpApi.
    
    Args:
        query: Search query
        num_results: Number of results to return
        **kwargs: Additional parameters for SerpApi including cutoff_year
    Returns:
        List of SearchResult objects
    """
    client = SerpApiClient()
    if 'cutoff_year' in kwargs and isinstance(kwargs['cutoff_year'], date):
        search_year = str(kwargs['cutoff_year'].year - 1)
    else:
        search_year = str('2099')

    try:
        params = {
            'as_ylo': "1900",
            'as_yhi': search_year,
            'num': num_results,
            'hl': kwargs.get('hl', 'en')  # Language
        }
        results = client.google_scholar_search(query=query, **params)
        formatted_results = _format_google_scholar_serpapi_results(results)
        
        # Convert formatted results to SearchResult objects
        search_results = []
        for article in formatted_results.get('articles', []):
            search_result = SearchResult(
                title=article.get('title', ''),
                url=article.get('link', ''),
                snippet=article.get('snippet', ''),
                source="google_scholar",
                published_date=article.get('published_date'),
                metadata={
                    'authors': article.get('authors', []),
                    'pdf_link': article.get('pdf_link', ''),
                    'cited_by': article.get('cited_by', ''),
                    'position': article.get('position', 0),
                    'journal_venue': article.get('journal/venue', ''),
                    'query': article.get('query', ''),
                    'search_time': article.get('search_time', '')
                }
            )
            search_results.append(search_result)
        
        return search_results

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []
    

def _format_google_scholar_serpapi_results(results_data):
    """
    Format SerpAPI Google Scholar results into structured data
    """
    # Extract search metadata
    search_info = {
        'query': results_data.get('search_parameters', {}).get('q', 'N/A'),
        'year': results_data.get('search_parameters', {}).get('as_yhi', 'N/A'),
        'total_results': results_data.get('search_information', {}).get('total_results', 0),
        'search_time': results_data.get('search_metadata', {}).get('created_at', 'N/A')
    }
    
    formatted_articles = []
    if 'organic_results' in results_data:
        for article in results_data['organic_results']:
            
            # authors
            authors = []
            if 'publication_info' in article and 'authors' in article['publication_info']:
                authors = [author.get('name', 'Unknown') for author in article['publication_info']['authors']]
            
            # PDF link
            pdf_link = None
            if 'resources' in article:
                for resource in article['resources']:
                    if resource.get('file_format') == 'PDF':
                        pdf_link = resource.get('link')
                        break
            
            # Format single article
            formatted_article = {
                'title': article.get('title', 'N/A'),
                'authors': authors,
                'cutoff_year': search_info['year'],
                'query': search_info['query'],
                'journal/venue': article.get('publication_info', {}).get('summary', 'N/A'),
                'link': article.get('link', 'N/A'),
                'pdf_link': pdf_link,
                'snippet': article.get('snippet', 'N/A'),
                'cited_by': article.get('inline_links', {}).get('cited_by', {}).get('total', 0),
                'position': article.get('position', 0),
                'search_time': search_info['search_time']
            }
            
            formatted_articles.append(formatted_article)
    
    return {
        'articles': formatted_articles
    }
