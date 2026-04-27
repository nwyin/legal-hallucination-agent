"""
CourtListener Search Action Module.

This module provides functionality to search the CourtListener API for legal opinions, cases, dockets, and other legal documents using the unified search API.
"""

from .main import (
    search_courtlistener,
    search_opinions,
    search_cases,
    search_dockets,
    search_filings,
    search_judges,
    search_oral_arguments,
    fetch_opinion,
    execute_courtlistener_search,
    execute_courtlistener_opinion_access,
    execute_courtlistener_citation_lookup,
    make_courtlistener_request,
)

__all__ = [
    "search_courtlistener",
    "search_opinions",
    "search_cases",
    "search_dockets",
    "search_filings",
    "search_judges",
    "search_oral_arguments",
    "fetch_opinion",
    "execute_courtlistener_search",
    "execute_courtlistener_opinion_access",
    "execute_courtlistener_citation_lookup",
    "make_courtlistener_request",
] 