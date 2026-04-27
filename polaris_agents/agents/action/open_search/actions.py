"""Action functions for open search operations."""

from typing import Optional, List, Dict, Any
from datetime import date
import json
import os
import requests
import logging

from ..action_types import ActionType, Action
from .search import search, search_results_xml, _escape_xml

logger = logging.getLogger(__name__)


def search_and_return_xml(
    query: str,
    search_type: str = "web",
    cutoff_date: Optional[date] = None,
    num_results: int = 10,
    news_source: str = "mediastack",
    exclude_undated: bool = True
) -> str:
    """
    Perform a search and return the XML string representation of the results.
    
    This is a convenience function that combines search execution with XML formatting,
    making it easy to get LLM-ready search results in a single call.
    
    Args:
        query: Search query string
        search_type: Type of search ("web", "news", "google_scholar")
        cutoff_date: Optional cutoff date for search results (YYYY-MM-DD)
        num_results: Number of results to return (default: 10)
        news_source: News source to use for news searches ("serpapi" or "mediastack", default: "serpapi")
        exclude_undated: If True, exclude results without publication dates (default: True)
        
    Returns:
        XML string representation of search results formatted for LLM consumption
        
    Example:
        >>> xml_results = search_and_return_xml(
        ...     query="Supreme Court decision",
        ...     search_type="news",
        ...     num_results=5
        ... )
        >>> print(xml_results)
        <search_results>
          <result>
            <result_id>AbC12</result_id>
            <title>Supreme Court Rules on Important Case</title>
            <date>2025-07-22</date>
            <description>The Supreme Court issued a landmark ruling today...</description>
            <url>https://example.com/news/supreme-court-ruling</url>
          </result>
        </search_results>
    """
    # Perform the search
    results = search(
        query=query,
        search_type=search_type,
        num_results=num_results,
        cutoff_date=cutoff_date,
        news_source=news_source,
        exclude_undated=exclude_undated
    )
    
    # Convert results to XML format
    xml_output = search_results_xml(results)
    
    return xml_output


def convert_search_results_to_markdown(
    result_ids: List[str],
    history_file: str = "data/search_history.json",
    use_readability: bool = True,
    markdown_format: str = "github",
    num_words: Optional[int] = 1000
) -> str:
    """
    Retrieve URLs from search history by IDs and convert them to Markdown.
    
    Uses Brett Terpstra's Markdownifier utility (heckyesmarkdown.com) to convert
    web pages to clean Markdown format and returns results as XML.
    
    Args:
        result_ids: List of search result IDs to convert
        history_file: Path to search history JSON file (default: "data/search_history.json")
        use_readability: Whether to use Readability to clean content (default: True)
        markdown_format: Markdown format to use - "github", "multimarkdown", etc. (default: "github")
        num_words: Maximum number of words to include per result (default: None for no limit)
        
    Returns:
        XML string containing conversion results:
        <markdown_results>
          <result>
            <result_id>AbC12</result_id>
            <title>Supreme Court Rules on Important Case</title>
            <date>2025-07-22</date>
            <url>https://example.com/article</url>
            <markdown>
              # Supreme Court Rules on Important Case
              
              The Supreme Court issued a landmark ruling today...
            </markdown>
            <success>true</success>
          </result>
        </markdown_results>
        
    Example:
        >>> xml_results = convert_search_results_to_markdown(["AbC12", "XyZ34"])
        >>> print(xml_results)
        >>> # Limit to 100 words per result
        >>> xml_results = convert_search_results_to_markdown(["AbC12"], num_words=100)
        >>> print(xml_results)
    """
    # Load search history
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load search history from {history_file}: {e}")
        return f"<markdown_results><error>Failed to load history: {_escape_xml(str(e))}</error></markdown_results>"
    
    xml_parts = ["<markdown_results>"]
    
    for result_id in result_ids:
        xml_parts.append("  <result>")
        xml_parts.append(f"    <result_id>{result_id}</result_id>")
        
        if result_id not in history:
            xml_parts.append(f"    <success>false</success>")
            xml_parts.append(f"    <error>Result ID '{result_id}' not found in search history</error>")
            xml_parts.append("  </result>")
            continue
        
        entry = history[result_id]
        url = entry.get("url", "")
        title = entry.get("title", "")
        published_date = entry.get("published_date", "")
        
        # Add basic metadata
        xml_parts.append(f"    <title>{_escape_xml(title)}</title>")
        xml_parts.append(f"    <date>{published_date[:10] if published_date else ''}</date>")  # Extract date part only
        xml_parts.append(f"    <url>{_escape_xml(url)}</url>")
        
        if not url:
            xml_parts.append(f"    <success>false</success>")
            xml_parts.append(f"    <error>No URL found in search history entry</error>")
            xml_parts.append("  </result>")
            continue
        
        # Convert URL to Markdown using Markdownifier
        try:
            markdown_content = _convert_url_to_markdown(url, use_readability, markdown_format)
            
            # Truncate to specified number of words if requested
            if num_words is not None and num_words > 0:
                markdown_content = _truncate_markdown_words(markdown_content, num_words)
            
            xml_parts.append(f"    <success>true</success>")
            xml_parts.append(f"    <markdown>{_escape_xml(markdown_content)}</markdown>")
            logger.info(f"Successfully converted {url} to Markdown for result ID {result_id}")
            
        except Exception as e:
            xml_parts.append(f"    <success>false</success>")
            xml_parts.append(f"    <error>Failed to convert URL to Markdown: {_escape_xml(str(e))}</error>")
            logger.error(f"Failed to convert {url} to Markdown for result ID {result_id}: {e}")
        
        xml_parts.append("  </result>")
    
    xml_parts.append("</markdown_results>")
    
    return "\n".join(xml_parts)


def _truncate_markdown_words(markdown_content: str, num_words: int) -> str:
    """
    Truncate markdown content to a specified number of words while preserving structure.
    
    This function attempts to preserve markdown formatting by truncating at word boundaries
    and adding an ellipsis to indicate truncation.
    
    Args:
        markdown_content: The markdown content to truncate
        num_words: Maximum number of words to keep
        
    Returns:
        Truncated markdown content
    """
    if not markdown_content or num_words <= 0:
        return ""
    
    # Split content into words while preserving whitespace structure
    words = markdown_content.split()
    
    if len(words) <= num_words:
        return markdown_content
    
    # Take the first num_words words
    truncated_words = words[:num_words]
    truncated_content = " ".join(truncated_words)
    
    # Add ellipsis to indicate truncation
    truncated_content += "..."
    
    return truncated_content


def _convert_url_to_markdown(url: str, use_readability: bool = True, markdown_format: str = "github") -> str:
    """
    Convert a single URL to Markdown using Brett Terpstra's Markdownifier.
    
    Args:
        url: URL to convert
        use_readability: Whether to use Readability to clean content
        markdown_format: Markdown format to use
        
    Returns:
        Markdown content as string
        
    Raises:
        requests.RequestException: If the HTTP request fails
        ValueError: If the conversion fails or returns empty content
    """
    markdownifier_url = "http://heckyesmarkdown.com/go/"
    
    # Prepare parameters for the Markdownifier API
    params = {
        "u": url,
        "read": "1" if use_readability else "0",
        "md": markdown_format
    }
    
    try:
        # Make request to Markdownifier
        response = requests.get(markdownifier_url, params=params, timeout=30)
        response.raise_for_status()
        
        markdown_content = response.text.strip()
        
        # Check if conversion was successful
        if not markdown_content:
            raise ValueError("Markdownifier returned empty content")
        
        # Check for common error indicators
        if "Sorry, Marky couldn't process that URL" in markdown_content:
            raise ValueError("Markdownifier could not process the URL")
        
        if "404" in markdown_content and len(markdown_content) < 100:
            raise ValueError("URL appears to return 404 or is not accessible")
        
        return markdown_content
        
    except requests.Timeout:
        raise requests.RequestException(f"Timeout while converting URL: {url}")
    except requests.RequestException as e:
        raise requests.RequestException(f"HTTP error while converting URL {url}: {e}")


def get_search_result_urls(result_ids: List[str], history_file: str = "data/search_history.json") -> Dict[str, str]:
    """
    Retrieve URLs from search history for given result IDs.
    
    Args:
        result_ids: List of search result IDs
        history_file: Path to search history JSON file
        
    Returns:
        Dict mapping result_id to URL
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load search history: {e}")
        return {}
    
    urls = {}
    for result_id in result_ids:
        if result_id in history:
            urls[result_id] = history[result_id].get("url", "")
        else:
            logger.warning(f"Result ID '{result_id}' not found in search history")
    
    return urls


def render_markdown_results_for_jupyter(markdown_xml: str) -> None:
    """
    Render markdown results XML in Jupyter notebook with proper formatting.
    
    This function parses the XML output from convert_search_results_to_markdown()
    and displays each result with rendered Markdown in Jupyter notebooks using
    IPython.display.
    
    Args:
        markdown_xml: XML string from convert_search_results_to_markdown()
        
    Example:
        >>> xml_results = convert_search_results_to_markdown(["AbC12"])
        >>> render_markdown_results_for_jupyter(xml_results)
        # Displays formatted results in Jupyter
    """
    try:
        from IPython.display import display, Markdown, HTML
        import xml.etree.ElementTree as ET
    except ImportError:
        print("Error: IPython is required for Jupyter rendering. Install with: pip install ipython")
        return
    
    try:
        # Parse the XML
        root = ET.fromstring(markdown_xml)
        
        # Check for error at root level
        error_elem = root.find('error')
        if error_elem is not None:
            display(HTML(f"<div style='color: red; font-weight: bold;'>Error: {error_elem.text}</div>"))
            return
        
        # Process each result
        for result in root.findall('result'):
            result_id = result.find('result_id').text if result.find('result_id') is not None else "Unknown"
            title = result.find('title').text if result.find('title') is not None else "No Title"
            date = result.find('date').text if result.find('date') is not None else ""
            url = result.find('url').text if result.find('url') is not None else ""
            success = result.find('success').text if result.find('success') is not None else "false"
            
            # Display header with metadata
            header_html = f"""
            <div style='border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background-color: #f9f9f9;'>
                <h3 style='margin-top: 0; color: #333;'>{_escape_xml(title)}</h3>
                <p style='margin: 5px 0; color: #666; font-size: 0.9em;'>
                    <strong>ID:</strong> {result_id} | 
                    <strong>Date:</strong> {date if date else "N/A"} | 
                    <strong>URL:</strong> <a href='{_escape_xml(url)}' target='_blank'>{_escape_xml(url[:60])}{'...' if len(url) > 60 else ''}</a>
                </p>
            """
            
            if success.lower() == "true":
                markdown_elem = result.find('markdown')
                if markdown_elem is not None and markdown_elem.text:
                    markdown_content = markdown_elem.text
                    
                    # Debug: Check if we have content
                    if len(markdown_content.strip()) == 0:
                        header_html += "<p style='color: orange;'>Markdown content is empty</p></div>"
                        display(HTML(header_html))
                    else:
                        # Close header div and display
                        header_html += f"<p style='color: green; font-size: 0.8em;'>✓ Markdown content: {len(markdown_content)} characters, {markdown_content.count(chr(10))} lines</p></div>"
                        display(HTML(header_html))
                        
                        # Display the rendered Markdown content
                        try:
                            display(Markdown(markdown_content))
                        except Exception as md_error:
                            display(HTML(f"<div style='color: red;'>Markdown rendering error: {md_error}</div>"))
                            # Fallback: display as preformatted text
                            display(HTML(f"<pre style='background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto;'>{_escape_xml(markdown_content[:1000])}{'...' if len(markdown_content) > 1000 else ''}</pre>"))
                else:
                    header_html += "<p style='color: orange;'>No markdown content available (element is None or text is None)</p></div>"
                    display(HTML(header_html))
            else:
                error_elem = result.find('error')
                error_text = error_elem.text if error_elem is not None else "Unknown error"
                header_html += f"<p style='color: red;'><strong>Error:</strong> {_escape_xml(error_text)}</p></div>"
                display(HTML(header_html))
            
            # Add separator
            display(HTML("<hr style='margin: 20px 0; border: none; border-top: 1px solid #eee;'>"))
    
    except ET.ParseError as e:
        display(HTML(f"<div style='color: red;'>XML Parse Error: {e}</div>"))
    except Exception as e:
        display(HTML(f"<div style='color: red;'>Rendering Error: {e}</div>"))
        import traceback
        display(HTML(f"<pre>{traceback.format_exc()}</pre>"))


class OpenWebSearch(Action):
    """
    Search action for web search operations.
    
    This action allows the agent to perform web searches to find relevant information
    from the internet using various search engines and news sources.
    """
    
    action_type = ActionType.OPEN_WEB_SEARCH
    description = "Perform Google search on the open internet (web, news, and Google Scholar). Use this to find current information, news, scholarly articles, and web content that may not be in the closed document index."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant web information",
            "required": True
        },
        "search_type": {
            "type": "string",
            "description": "The type of search to perform. Options: 'web', 'news', 'google_scholar' (default: 'web')",
            "required": False
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (default: 10)",
            "required": False
        },
        "news_source": {
            "type": "string",
            "description": "News source to use for news searches. Options: 'serpapi', 'mediastack' (default: 'serpapi')",
            "required": False
        },
        "cutoff_date": {
            "type": "string",
            "description": "Cutoff date for search results (YYYY-MM-DD format). Only results published on or before this date will be returned. Optional.",
            "required": False
        },
        "exclude_undated": {
            "type": "boolean",
            "description": "If true, exclude results without publication dates (default: true)",
            "required": False
        }
    }

    def __init__(self, query: str, search_type: str = "web", num_results: int = 10, 
                 news_source: str = "serpapi", cutoff_date: str = None, exclude_undated: bool = True, **kwargs):
        super().__init__(query=query, search_type=search_type, num_results=num_results, 
                        news_source=news_source, cutoff_date=cutoff_date, exclude_undated=exclude_undated, **kwargs)
        self.query = query
        self.search_type = search_type
        self.num_results = num_results
        self.news_source = news_source
        self.cutoff_date = cutoff_date
        self.exclude_undated = exclude_undated

    def __repr__(self):
        return f"<OpenWebSearch query='{self.query}' search_type='{self.search_type}' num_results={self.num_results}>"