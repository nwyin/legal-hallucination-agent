"""Unified search interface for open search functionality."""

from typing import Dict, Any, List, Optional, Union
from datetime import date, datetime
import logging
import json
import os
import random
import string

from .serpapi import SerpApiClient
from .mediastack import MediaStackClient
from .date_parser import parse_date_string, format_parsed_date

logger = logging.getLogger(__name__)


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

