"""
CourtListener search actions for the agent system.

This module implements search actions for the CourtListener database,
allowing agents to search for legal cases and documents.
"""

from typing import Any

from polaris_agents.agents.action.action_types import Action, ActionType
from polaris_agents.agents.action.courtlistener_search.main import SEARCH_TYPES


class AccessCourtListenerOpinion(Action):
    """
    Access a single CourtListener opinion by ID.
    
    Use this after OPEN_COURTLISTENER_SEARCH to fetch the full text and metadata
    of a specific opinion (e.g., to verify a citation or quotation).
    The full opinion is stored locally; the agent sees a short snippet. Use
    SEARCH_LOCAL_OPINION to search within a previously fetched opinion.
    """
    
    action_type = ActionType.ACCESS_COURTLISTENER_OPINION
    description = "Fetch the full opinion from CourtListener by opinion ID. Use after searching to retrieve the complete opinion text for citation verification. The full opinion is stored locally; use SEARCH_LOCAL_OPINION to search within it."
    inputs = {
        "opinion_id": {
            "type": "string",
            "description": "The CourtListener opinion ID (obtained from OPEN_COURTLISTENER_SEARCH results)",
            "required": True
        }
    }

    def __init__(self, opinion_id: str):
        super().__init__(opinion_id=opinion_id)
        self.opinion_id = opinion_id


class SearchLocalOpinion(Action):
    """
    Search for a string within a previously fetched opinion (by opinion_id).
    
    Use after ACCESS_COURTLISTENER_OPINION to search within that opinion's full text.
    Returns a snippet around the match, or None if not found.
    """
    
    action_type = ActionType.SEARCH_LOCAL_OPINION
    description = "Search for a string within an opinion already fetched with ACCESS_COURTLISTENER_OPINION. Returns a snippet around the match, or None if not found."
    inputs = {
        "opinion_id": {
            "type": "string",
            "description": "The CourtListener opinion ID (from a previous ACCESS_COURTLISTENER_OPINION call)",
            "required": True
        },
        "search_string": {
            "type": "string",
            "description": "The string to search for in the opinion text (e.g., a quoted phrase or citation)",
            "required": True
        }
    }

    def __init__(self, opinion_id: str, search_string: str):
        super().__init__(opinion_id=opinion_id, search_string=search_string)
        self.opinion_id = opinion_id
        self.search_string = search_string


class CourtListenerCitationLookup(Action):
    """
    Look up a reporter citation on CourtListener (e.g. '934 F.3d 53', '143 S. Ct. 1196').
    Returns matching opinion(s) if found.
    """
    action_type = ActionType.COURTLISTENER_CITATION_LOOKUP
    description = "Look up a legal citation (e.g. '934 F.3d 53', '143 S. Ct. 1196') on CourtListener. Returns matching opinion(s) if found."
    inputs = {
        "cite": {
            "type": "string",
            "description": "The reporter citation to look up (e.g. '934 F.3d 53', '143 S. Ct. 1196')",
            "required": True
        }
    }

    def __init__(self, cite: str):
        super().__init__(cite=cite)
        self.cite = cite


class OpenCourtListenerSearch(Action):
    """
    Search action for CourtListener database.
    
    This action allows the agent to search the CourtListener database
    for legal cases, opinions, and other legal documents.
    """
    
    action_type = ActionType.OPEN_COURTLISTENER_SEARCH
    description = "Search the CourtListener database for legal cases, opinions, and documents. Use this to find relevant legal information and precedents."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant legal cases and documents",
            "required": True
        },
        "search_type": {
            "type": "string",
            "description": "The type of search to perform. Options: " + ", ".join([f"'{k}'" for k in SEARCH_TYPES.keys()]) + " (default: 'opinions')",
            "required": False
        },
        "court": {
            "type": "string",
            "description": "The court to search (e.g., 'scotus', 'ca1', 'ca2') (default: 'scotus')",
            "required": False
        },
        "date_filter": {
            "type": "object",
            "description": "Date filter with 'field' and 'before'/'after' keys (e.g., {'field': 'date_filed', 'before': '2022-01-01', 'after': '2021-01-01'}). Dates in YYYY-MM-DD format. Only 'date_filed' field is supported.",
            "required": False
        }
    }

    def __init__(self, query: str, search_type: str="opinions", court: str="scotus", 
                 date_filter: dict=None):
        super().__init__(query=query, search_type=search_type)
        self.query = query
        self.search_type = search_type
        self.court = court
        self.date_filter = date_filter