"""Open search module for strategic exploration."""

from .search import search, search_results_xml, SearchResult
from .serpapi import SerpApiClient
from .mediastack import MediaStackClient

__all__ = ["search", "search_results_xml", "SearchResult", "SerpApiClient", "MediaStackClient"]
