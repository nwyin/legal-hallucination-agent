"""Action functions for open search operations."""

from ..action_types import ActionType, Action


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
