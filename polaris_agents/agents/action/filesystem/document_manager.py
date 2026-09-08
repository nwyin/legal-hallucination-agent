"""
Document manager for storing opinions and editing the scratchpad.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Manages documents and provides windowed reading capabilities.
    """
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.scratchpad: OrderedDict[int, str] = OrderedDict()
        self.scratchpad_counter = 0

    def register_opinion(self, opinion_id: str, content: str, case_name: str = "", **metadata: Any) -> str:
        """
        Register a fetched opinion so it can be read via READ_DOCUMENT.

        Args:
            opinion_id: CourtListener opinion ID (e.g. "9001448")
            content: Full plain text of the opinion
            case_name: Optional case name for metadata
            **metadata: Optional extra fields (e.g. court, date_filed, url)

        Returns:
            Document ID to use with READ_DOCUMENT (e.g. "opinion_9001448")
        """
        doc_id = f"opinion_{opinion_id}"
        lines = content.split("\n") if content else []
        self.documents[doc_id] = {
            "type": "opinion",
            "case_name": case_name or "",
            "court": metadata.get("court", ""),
            "date_filed": metadata.get("date_filed", ""),
            "url": metadata.get("url", ""),
            "content": content,
            "lines": lines,
            "metadata": {"opinion_id": opinion_id, **metadata},
        }
        logger.info(f"Registered opinion for READ_DOCUMENT: {doc_id}")
        return doc_id

    def resolve_document_id(self, document_id: str) -> Optional[str]:
        """
        Resolve document_id to the key used in self.documents.
        Accepts "opinion_9001448" or "9001448" for opinions.
        """
        if document_id in self.documents:
            return document_id
        if document_id.isdigit() and f"opinion_{document_id}" in self.documents:
            return f"opinion_{document_id}"
        return None

    def edit_scratchpad(self, operation: str, content: str, position: Optional[int] = None) -> bool:
        """
        Edit the scratchpad.
        
        Args:
            operation: Type of edit (append, insert, replace, clear)
            content: Content to add or replace
            position: Position for insert/replace operations
            
        Returns:
            True if edit was successful
        """
        if operation == 'append':
            self.scratchpad[self.scratchpad_counter] = content
            self.scratchpad_counter += 1
            
        elif operation == 'insert':
            if position is None:
                logger.error("Position required for insert operation")
                return False
                
            # Shift existing entries
            new_scratchpad = OrderedDict()
            for i in range(self.scratchpad_counter):
                if i < position:
                    new_scratchpad[i] = self.scratchpad[i]
                elif i == position:
                    new_scratchpad[i] = content
                    new_scratchpad[i + 1] = self.scratchpad[i]
                else:
                    new_scratchpad[i + 1] = self.scratchpad[i]
            
            self.scratchpad = new_scratchpad
            self.scratchpad_counter += 1
            
        elif operation == 'replace':
            if position is None or position not in self.scratchpad:
                logger.error(f"Invalid position {position} for replace operation")
                return False
                
            self.scratchpad[position] = content
            
        elif operation == 'clear':
            self.scratchpad.clear()
            self.scratchpad_counter = 0
            
        else:
            logger.error(f"Unknown scratchpad operation: {operation}")
            return False
            
        logger.info(f"Scratchpad {operation}: {content[:50]}...")
        return True
    
    def get_scratchpad_content(self) -> str:
        """
        Get the current scratchpad content as a string.
        
        Returns:
            Scratchpad content
        """
        return '\n'.join(self.scratchpad.values())
    
    def get_document_content(self, document_id: str) -> Optional[str]:
        """
        Get the full content of a document directly.
        document_id can be "opinion_9001448" or "9001448" for opinions.

        Args:
            document_id: ID of the document

        Returns:
            Document content as string, or None if not found
        """
        key = self.resolve_document_id(document_id)
        if key is None:
            logger.error(f"Document {document_id} not found")
            return None
        doc = self.documents[key]
        return doc.get("content", "")


def read_document_content(doc_content: str, document_id: str, start_line: int, num_lines: int) -> Tuple[str, int, int, int]:
    """
    Read a portion of a document from the document manager.
    """
    # Split content into lines
    lines = doc_content.split('\n')
    total_lines = len(lines)
    
    # Calculate the range to read
    end_line = min(start_line + num_lines, total_lines)
    actual_start = max(0, start_line)
    actual_end = min(end_line, total_lines)
    
    # Extract the requested lines
    selected_lines = lines[actual_start:actual_end]
    content = '\n'.join(selected_lines)

    return content, actual_start, actual_end, total_lines