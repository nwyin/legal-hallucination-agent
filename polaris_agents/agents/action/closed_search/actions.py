"""
Closed World Search Action.
Implements both dense and lexical (BM25) search over documents with filtering capabilities.
"""

from ..action_types import ActionType, Action
from typing import Optional, Dict, List, Any


class ClosedSearch(Action):
    """
    Action that performs search over documents to find relevant information.
    Supports both dense and lexical (BM25) search with various filtering options.
    """
    
    action_type = ActionType.CLOSED_SEARCH
    description = "Semantic search through a fixed set of indexed documents (closed world) using natural language queries. Queries are matched by meaning, so use descriptive phrases about what you want to find. If you get no results or poor results, it could mean (1) the information doesn't exist in the indexed documents, or (2) your query needs to be rephrased with different terminology or more specific details."
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language query describing the information you're looking for. Be specific about the type of document, topic, or content you need.",
            "required": True
        },
        "search_type": {
            "type": "string",
            "description": "Which index to search. See Search Capabilities for available options and what each contains.",
            "required": True,
            "enum": ["docket_file_documents", "legal_research_documents"]
        },
        "k": {
            "type": "integer",
            "description": "Number of results to return",
            "required": False,
            "default": 10
        }
        # Note: field_filters, date_filter, must_have_fields are handled automatically
        # by the environment
    }
    
    def __init__(self, query: str, search_type: str, k: int = 3, **kwargs):
        super().__init__(query=query, search_type=search_type, k=k, **kwargs)
        self.query = query
        self.search_type = search_type
        self.k = k
        # These are set by the environment automatically
        self.field_filters = {}
        self.date_filter = None
        self.must_have_fields = None
        self.include_null_dates = None
    
    
    def __repr__(self):
        return f"<ClosedSearch query='{self.query}' k={self.k}>"
