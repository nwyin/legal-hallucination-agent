"""
Filesystem actions for document reading and scratchpad editing.
"""

from polaris_agents.agents.action.action_types import Action, ActionType


class ReadDocument(Action):
    """
    Action for reading a portion of a document.
    
    This action allows the agent to read specific portions of documents
    that have been retrieved from search results or other sources.
    """
    
    action_type = ActionType.READ_DOCUMENT
    description = "Read a portion of a document from search results or available documents. Use this to examine specific parts of legal cases, opinions, or other documents."
    inputs = {
        "opinion_id": {
            "type": "string",
            "description": "The opinion to read: use opinion_<id> (e.g. opinion_9001448) or just the numeric id (e.g. 9001448) from a previous ACCESS_COURTLISTENER_OPINION.",
            "required": True
        },
        "start_line": {
            "type": "integer",
            "description": "The starting line number to read from (0-indexed). Use 0 for the first line.",
            "required": True
        },
        "num_lines": {
            "type": "integer",
            "description": "The number of lines to read from the document starting from start_line. Must be a positive integer.",
            "required": True
        }
    }

    def __init__(self, opinion_id: str, start_line: int, num_lines: int):
        super().__init__(opinion_id=opinion_id, start_line=start_line, num_lines=num_lines)
        self.opinion_id = opinion_id
        self.start_line = start_line
        self.num_lines = num_lines


class EditScratchpad(Action):
    """
    Action for editing the agent's scratchpad.
    
    This action allows the agent to take notes, organize thoughts,
    and maintain a working memory during reasoning.
    """
    
    action_type = ActionType.EDIT_SCRATCHPAD
    description = "Edit the agent's scratchpad to take notes, organize thoughts, or maintain working memory. Use this to keep track of important information during reasoning."
    inputs = {
        "operation": {
            "type": "string",
            "description": "The operation to perform: 'append' (add to end), 'insert' (insert at position), 'replace' (replace at position), or 'clear' (clear all content)",
            "required": True
        },
        "content": {
            "type": "string",
            "description": "The content to add, insert, or replace in the scratchpad",
            "required": True
        },
        "position": {
            "type": "integer",
            "description": "The position for insert/replace operations (0-indexed). Required for 'insert' and 'replace' operations. Ignored for 'append' and 'clear' operations. Optional.",
            "required": False
        }
    }

    def __init__(self, operation: str, content: str, position: int = None):
        super().__init__(operation=operation, content=content, position=position)
        self.operation = operation
        self.content = content
        self.position = position
