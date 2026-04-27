"""
CourtListener API search implementation.

This module provides functionality to search the CourtListener API for legal opinions,
cases, dockets, and other legal documents using the unified search API.
"""

import os
import logging
import requests_cache
import requests as requests_original

import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode
import backoff
from datetime import timedelta, datetime
import certifi

class NonRetryableError(Exception):
    """Exception for errors that should not be retried (e.g., malformed queries, auth issues)."""
    pass

requests = requests_cache.CachedSession(
    "courtlistener_cache",
    backend="sqlite",
    expire_after=timedelta(days=10),
)

from ...action import Action
from ....environments.base import Observation

logger = logging.getLogger(__name__)

# CourtListener API base URL
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# API key/token should be set as environment variable (support both names)
def _get_courtlistener_api_key() -> Optional[str]:
    """Fetch the CourtListener API key from environment variables."""
    return os.getenv("COURTLISTENER_API_KEY") or os.getenv("COURTLISTENER_API_TOKEN")

# Rate limiting - CourtListener V4 API allows more requests
RATE_LIMIT_DELAY = 1  # seconds between requests for V4 API

# Search type mappings based on CourtListener API documentation
SEARCH_TYPES = {
    "opinions": "o",      # Case law opinion clusters with nested Opinion documents
    "cases": "r",         # List of Federal cases (dockets) with up to three nested documents
    "dockets": "d",       # Federal cases (dockets) from PACER
    "filings": "rd",      # Federal filing documents from PACER
    "judges": "p",        # Judges
    "oral_arguments": "oa" # Oral argument audio files
}


@backoff.on_exception(
    backoff.expo,
    requests_original.exceptions.RequestException,
    max_time=300,
)
def make_courtlistener_request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a request to the CourtListener V4 API with rate limiting and retry logic.
    
    Args:
        endpoint: API endpoint (e.g., "search/", "opinions/", "dockets/")
        params: Query parameters for the request
        
    Returns:
        JSON response from the API
    """
    url = f"{COURTLISTENER_BASE_URL}/{endpoint}"
    
    # Add API key if available (V4 API uses Authorization header)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    
    api_key = _get_courtlistener_api_key()
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    
    logger.debug(f"Making request to {url} with params: {params}")
    
    # Rate limiting
    time.sleep(RATE_LIMIT_DELAY)
    
    response = requests.get(
        url, 
        params=params, 
        headers=headers,
        verify=certifi.where()
    )
    
    if not response.ok:
        logger.error(f"CourtListener V4 API request failed: {response.status_code} - {response.text}")
        
        # Check for semantic errors that shouldn't be retried
        if response.status_code in [400, 401, 403, 404]:
            # These are client errors (malformed queries, auth issues, etc.) - don't retry
            raise NonRetryableError(f"Semantic error: {response.status_code} - {response.text}")
        
        # For other errors (500, 502, 503, etc.), let the backoff decorator handle retries
        response.raise_for_status()
    
    return response.json()


def convert_date_filter_to_api_params(date_filter: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert structured date_filter to CourtListener API parameters.
    
    Args:
        date_filter: Dict with 'field' and 'before'/'after' keys (dates in YYYY-MM-DD format)
        
    Returns:
        Dictionary with date_filed__gt and/or date_filed__lt parameters
    """
    if not date_filter or 'field' not in date_filter:
        return {}
    
    # Only support date_filed field for now
    if date_filter['field'] != 'date_filed':
        return {}
    
    params = {}
    
    if 'before' in date_filter:
        before_date = date_filter['before']
        # Validate date format (basic check)
        if isinstance(before_date, str) and len(before_date) == 10 and before_date.count('-') == 2:
            params['date_filed__lt'] = before_date
    
    if 'after' in date_filter:
        after_date = date_filter['after']
        # Validate date format (basic check)
        if isinstance(after_date, str) and len(after_date) == 10 and after_date.count('-') == 2:
            params['date_filed__gt'] = after_date
    
    return params


def fetch_opinion(opinion_id: str) -> Dict[str, Any]:
    """
    Fetch a single opinion from CourtListener by ID.
    
    Args:
        opinion_id: The CourtListener opinion ID
        
    Returns:
        Dictionary containing the full opinion data
    """
    endpoint = f"opinions/{opinion_id}/"
    return make_courtlistener_request(endpoint, {})


def search_courtlistener(query: str, search_type: str = "opinions", **kwargs) -> Dict[str, Any]:
    """
    Search CourtListener using the unified search API.
    
    Args:
        query: Search query string
        search_type: Type of search ("opinions", "cases", "dockets", "filings", "judges", "oral_arguments")
        **kwargs: Additional search parameters (court, etc.)
        
    Returns:
        Dictionary containing search results
    """
    # Map search_type to API type parameter
    api_type = SEARCH_TYPES.get(search_type, "o")  # Default to opinions
    
    params = {
        "q": query,
        "type": api_type,
        "order_by": "score desc",
        "available_only": "on",
        **kwargs
    }
    
    return make_courtlistener_request("search/", params)


def create_search_summary(results: Dict[str, Any], search_type: str, max_snippets: int = 5) -> Dict[str, Any]:
    """
    Create a summary of search results with snippets instead of full content.
    
    Args:
        results: Full search results from CourtListener
        search_type: Type of search performed
        max_snippets: Maximum number of snippets to include
        
    Returns:
        Dictionary with summary information and snippets
    """
    if not results or 'results' not in results:
        return results
    
    summary_results = []
    
    for i, result in enumerate(results['results'][:max_snippets]):
        # Extract common fields
        case_name = result.get('caseName', f'Case {i}')
        court = result.get('court', 'Unknown Court')
        date_filed = result.get('dateFiled', 'Unknown Date')
        url = result.get('absolute_url', '')
        # Handle different result types
        if search_type == "opinions":
            # For opinions, we have opinion data directly
            snippet = result.get('snippet', '')
            if not snippet and 'opinions' in result and result['opinions']:
                # Fallback to first opinion snippet
                snippet = result['opinions'][0].get('snippet', '')
            
            # Get citation information
            citations = result.get('citation', [])
            citation_text = '; '.join(citations) if citations else ''
            
            summary_result = {
                'id': result.get('cluster_id'),
                'case_name': case_name,
                'case_name_full': result.get('caseNameFull', case_name),
                'court': court,
                'date_filed': date_filed,
                'url': url,
                'snippet': snippet,
                'citations': citation_text,
                'docket_number': result.get('docketNumber', ''),
                'status': result.get('status', ''),
                'metadata': result
            }
            
        elif search_type in ["cases", "dockets"]:
            # For cases/dockets, extract relevant information
            summary_result = {
                'id': result.get('docket_id'),
                'case_name': case_name,
                'court': court,
                'date_filed': date_filed,
                'url': url,
                'docket_number': result.get('docketNumber', ''),
                'more_docs': result.get('more_docs', False),
                'metadata': result
            }
            
        elif search_type == "filings":
            # For filings/documents
            summary_result = {
                'id': result.get('id'),
                'case_name': case_name,
                'court': court,
                'date_filed': date_filed,
                'url': url,
                'document_type': result.get('document_type', ''),
                'metadata': result
            }
            
        elif search_type == "judges":
            # For judges
            summary_result = {
                'id': result.get('id'),
                'name': result.get('name', ''),
                'court': court,
                'position': result.get('position', ''),
                'url': url,
                'metadata': result
            }
            
        elif search_type == "oral_arguments":
            # For oral arguments
            summary_result = {
                'id': result.get('id'),
                'case_name': case_name,
                'court': court,
                'date_argued': result.get('dateArgued', ''),
                'url': url,
                'duration': result.get('duration', ''),
                'metadata': result
            }
            
        else:
            # Generic fallback
            summary_result = {
                'id': result.get('id'),
                'case_name': case_name,
                'court': court,
                'date_filed': date_filed,
                'url': url,
                'metadata': result
            }
        
        summary_results.append(summary_result)
    
    # Create summary response
    summary = {
        'count': results.get('count', 0),
        'next': results.get('next'),
        'previous': results.get('previous'),
        'results': summary_results,
        'total_results': len(results.get('results', [])),
        'snippets_shown': len(summary_results),
        'search_type': search_type
    }
    
    return summary


# Legacy functions for backward compatibility
def search_opinions(query: str, **kwargs) -> Dict[str, Any]:
    """
    Search for legal opinions using the CourtListener search API.
    
    Args:
        query: Search query string
        **kwargs: Additional search parameters (court, etc.)
        
    Returns:
        Dictionary containing search results
    """
    return search_courtlistener(query, search_type="opinions", **kwargs)


def search_cases(query: str, **kwargs) -> Dict[str, Any]:
    """
    Search for cases in CourtListener using the search API.
    
    Args:
        query: Search query string
        **kwargs: Additional search parameters
        
    Returns:
        Search results from CourtListener API
    """
    return search_courtlistener(query, search_type="cases", **kwargs)


def search_dockets(query: str, **kwargs) -> Dict[str, Any]:
    """
    Search for dockets in CourtListener using the search API.
    
    Args:
        query: Search query string
        **kwargs: Additional search parameters
        
    Returns:
        Search results from CourtListener API
    """
    return search_courtlistener(query, search_type="dockets", **kwargs)


def search_filings(query: str, **kwargs) -> Dict[str, Any]:
    """
    Search for federal filing documents in CourtListener using the search API.
    
    Args:
        query: Search query string
        **kwargs: Additional search parameters
        
    Returns:
        Search results from CourtListener API
    """
    return search_courtlistener(query, search_type="filings", **kwargs)


def search_judges(query: str, **kwargs) -> Dict[str, Any]:
    """
    Search for judges in CourtListener using the search API.
    
    Args:
        query: Search query string
        **kwargs: Additional search parameters
        
    Returns:
        Search results from CourtListener API
    """
    return search_courtlistener(query, search_type="judges", **kwargs)


def search_oral_arguments(query: str, **kwargs) -> Dict[str, Any]:
    """
    Search for oral argument audio files in CourtListener using the search API.
    
    Args:
        query: Search query string
        **kwargs: Additional search parameters
        
    Returns:
        Search results from CourtListener API
    """
    return search_courtlistener(query, search_type="oral_arguments", **kwargs)


def execute_courtlistener_opinion_access(opinion_id: str) -> Observation:
    """
    Fetch a single CourtListener opinion by ID.
    
    Args:
        opinion_id: The CourtListener opinion ID (e.g., from search results)
        
    Returns:
        Observation containing the full opinion data
    """
    if not opinion_id:
        raise ValueError("opinion_id is required for CourtListener opinion access")

    try:
        opinion_data = fetch_opinion(opinion_id)
        
        observation_result = {
            "action_type": "ACCESS_COURTLISTENER_OPINION",
            "opinion_id": opinion_id,
            "opinion": opinion_data,
        }
        
        return Observation(
            result=observation_result,
            metadata={
                "action_type": "ACCESS_COURTLISTENER_OPINION",
                "opinion_id": opinion_id,
                "opinion": opinion_data,
            }
        )
        
    except NonRetryableError as e:
        logger.error(f"CourtListener opinion access failed (non-retryable): {str(e)}")
        return Observation(
            result={
                "error": str(e),
                "opinion_id": opinion_id,
                "opinion": None,
            },
            metadata={
                "action_type": "ACCESS_COURTLISTENER_OPINION",
                "opinion_id": opinion_id,
                "error": True,
                "non_retryable": True,
            }
        )
    except Exception as e:
        logger.error(f"CourtListener opinion access failed: {str(e)}")
        return Observation(
            result={
                "error": str(e),
                "opinion_id": opinion_id,
                "opinion": None,
            },
            metadata={
                "action_type": "ACCESS_COURTLISTENER_OPINION",
                "opinion_id": opinion_id,
                "error": True,
            }
        )


CITATION_LOOKUP_URL = f"{COURTLISTENER_BASE_URL}/citation-lookup/"


def _sleep_until_iso(iso_timestamp: str) -> None:
    """Sleep until the given ISO timestamp (e.g. from CourtListener rate limit wait_until)."""
    try:
        from dateutil import parser as date_parser
        until = date_parser.isoparse(iso_timestamp)
    except Exception:
        until = None
    if until is not None:
        delta = (until - datetime.now(until.tzinfo)).total_seconds() if until.tzinfo else (until - datetime.utcnow()).total_seconds()
        if delta > 0:
            time.sleep(min(delta, 60))


def execute_courtlistener_citation_lookup(cite: str) -> Observation:
    """
    Look up a reporter citation (e.g. '934 F.3d 53', '143 S. Ct. 1196') on CourtListener.
    Uses the citation-lookup API (POST). Returns Observation with citations list and full results.
    """
    cite = (cite or "").strip()
    if not cite:
        return Observation(
            result={
                "action_type": "COURTLISTENER_CITATION_LOOKUP",
                "cite": cite,
                "citations": [],
                "results": [],
                "error": "cite is required",
            },
            metadata={"action_type": "COURTLISTENER_CITATION_LOOKUP", "cite": cite, "error": "cite is required"},
        )

    api_key = _get_courtlistener_api_key()
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; courtlistener-citation-lookup/1.0)",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    max_attempts = 3
    hit: Optional[List[Dict[str, Any]]] = None

    for attempt in range(max_attempts):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.post(
                CITATION_LOOKUP_URL,
                headers=headers,
                json={"text": cite},
                timeout=30,
                verify=certifi.where(),
            )

            if resp.status_code == 429:
                j = resp.json() if resp.text else {}
                wait_until = j.get("wait_until")
                if wait_until:
                    _sleep_until_iso(wait_until)
                    continue
                time.sleep(5)
                continue

            resp.raise_for_status()
            raw = resp.json()
            hit = raw if isinstance(raw, list) else []
            break
        except requests_original.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (400, 401, 403, 404):
                raise NonRetryableError(str(e))
            if attempt == max_attempts - 1:
                raise
            time.sleep(5)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            logger.warning(f"Citation lookup attempt {attempt + 1} failed: {e}")
            time.sleep(5)

    if hit is None:
        hit = []

    citations = [item.get("citation") for item in hit if isinstance(item, dict) and item.get("citation")]

    result = {
        "action_type": "COURTLISTENER_CITATION_LOOKUP",
        "cite": cite,
        "citations": citations,
        "results": hit,
    }
    metadata = {
        "action_type": "COURTLISTENER_CITATION_LOOKUP",
        "cite": cite,
    }
    return Observation(result=result, metadata=metadata)


def execute_courtlistener_search(query: str, 
                                search_type: str = "opinions",
                                **search_kwargs) -> Observation:
    """
    Execute a CourtListener search action using the unified search API.
    
    Args:
        action: CourtListenerSearchAction with search parameters
        
    Returns:
        Observation containing search results summary and full results for storage
    """
    if not query:
        raise ValueError("Query parameter is required for CourtListener search")
    
    # Validate search type
    if search_type not in SEARCH_TYPES:
        raise ValueError(f"Unsupported search type: {search_type}. Supported types: {list(SEARCH_TYPES.keys())}")

    try:
        # Use the unified search API
        full_results = search_courtlistener(query, search_type, **search_kwargs)
        
        # Create summary with snippets for immediate display
        summary_results = create_search_summary(full_results, search_type, max_snippets=5)
        
        # Create metadata with search context and full results for storage
        metadata = {
            "search_type": search_type,
            "api_type": SEARCH_TYPES[search_type],
            "query": query,
            "search_params": search_kwargs,
            "total_results": full_results.get("count", 0),
            "api_endpoint": f"{COURTLISTENER_BASE_URL}/search/",
            "full_results": full_results  # Store full results in metadata for document manager
        }
        
        return Observation(result=summary_results, metadata=metadata)
        
    except NonRetryableError as e:
        logger.error(f"CourtListener search failed (non-retryable): {str(e)}")
        return Observation(
            result={
                "error": f"Search query error: {str(e)}. Please fix the query and try again.",
                "results": [],
                "query_error": True
            },
            metadata={
                "search_type": search_type,
                "query": query,
                "error": True,
                "non_retryable": True
            }
        )
    except Exception as e:
        logger.error(f"CourtListener search failed: {str(e)}")
        return Observation(
            result={"error": str(e), "results": []},
            metadata={
                "search_type": search_type,
                "query": query,
                "error": True
            }
        ) 