"""MediaStack client for news and media search functionality."""

import os
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import logging
import time
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


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