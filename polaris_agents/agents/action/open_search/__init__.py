"""Open search module for strategic exploration."""

from .search import search, search_results_xml, SearchResult
from .serpapi import SerpApiClient
from .mediastack import MediaStackClient
from .actions import search_and_return_xml, convert_search_results_to_markdown, get_search_result_urls, render_markdown_results_for_jupyter

__all__ = ["search", "search_results_xml", "SearchResult", "SerpApiClient", "MediaStackClient", 
           "search_and_return_xml", "convert_search_results_to_markdown", "get_search_result_urls", 
           "render_markdown_results_for_jupyter"]