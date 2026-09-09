"""Document manager: fetched opinion text for READ_DOCUMENT and the agent's scratchpad."""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DocumentManager:
    """Holds fetched opinions by document id ("opinion_<id>") and the scratchpad entries."""

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.scratchpad: List[str] = []

    def reset(self) -> None:
        self.documents.clear()
        self.scratchpad.clear()

    def register_opinion(self, opinion_id: str, content: str) -> str:
        """Store an opinion's plain text; returns the document id used with READ_DOCUMENT."""
        doc_id = f"opinion_{opinion_id}"
        self.documents[doc_id] = content
        logger.info(f"Registered opinion for READ_DOCUMENT: {doc_id}")
        return doc_id

    def resolve_document_id(self, document_id: str) -> Optional[str]:
        """Accept "opinion_9001448" or "9001448"; return the stored key or None."""
        if document_id in self.documents:
            return document_id
        if document_id.isdigit() and f"opinion_{document_id}" in self.documents:
            return f"opinion_{document_id}"
        return None

    def edit_scratchpad(self, operation: str, content: str, position: Optional[int] = None) -> bool:
        """Apply append / insert / replace / clear; returns False on an invalid request."""
        if operation == "append":
            self.scratchpad.append(content)
        elif operation == "insert":
            if position is None:
                logger.error("Position required for insert operation")
                return False
            self.scratchpad.insert(position, content)
        elif operation == "replace":
            if position is None or not 0 <= position < len(self.scratchpad):
                logger.error(f"Invalid position {position} for replace operation")
                return False
            self.scratchpad[position] = content
        elif operation == "clear":
            self.scratchpad.clear()
        else:
            logger.error(f"Unknown scratchpad operation: {operation}")
            return False
        logger.info(f"Scratchpad {operation}: {content[:50]}...")
        return True

    def get_scratchpad_content(self) -> str:
        return "\n".join(self.scratchpad)


def read_document_content(doc_content: str, start_line: int, num_lines: int) -> Tuple[str, int, int, int]:
    """Return (text, start, end, total_lines) for the line window [start_line, start_line + num_lines)."""
    lines = doc_content.split("\n")
    start = max(0, start_line)
    end = min(start_line + num_lines, len(lines))
    return "\n".join(lines[start:end]), start, end, len(lines)
