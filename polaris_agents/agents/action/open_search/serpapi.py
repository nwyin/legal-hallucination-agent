"""SerpApi client for web search functionality using direct requests."""

import os
import requests
from typing import Dict, Any, Optional, List
from datetime import date
import logging
import time
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


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