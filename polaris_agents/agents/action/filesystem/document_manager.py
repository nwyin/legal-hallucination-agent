"""
Document manager for handling search results and document storage.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
from polaris_agents.agents.action.action_types import Action
from polaris_agents.environments.base import Observation

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

    def store_search_results(self, search_results: Dict[str, Any]) -> List[str]:
        """
        Store search results as individual documents.
        
        Args:
            search_results: Results from CourtListener search (full results, not snippets)
            
        Returns:
            List of document IDs created
        """
        document_ids = []
        
        if 'results' in search_results:
            for i, result in enumerate(search_results['results']):
                doc_id = f"search_result_{i}"
                
                # Extract key information from processed results
                # Handle new unified API response format
                case_name = result.get('case_name', result.get('caseName', f'Case {i}'))
                court = result.get('court', 'Unknown Court')
                date_filed = result.get('date_filed', result.get('dateFiled', 'Unknown Date'))
                url = result.get('url', result.get('absolute_url', ''))
                
                # Get content - try different possible field names
                content = ''
                # Prefer full text first, then plain text, then snippet, then nested opinions
                if 'full_text' in result:
                    content = result['full_text']
                elif 'plain_text' in result:
                    content = result['plain_text']
                elif 'snippet' in result:
                    content = result['snippet']
                elif 'opinions' in result and result['opinions']:
                    # For opinions, try to get content from nested opinions
                    for opinion in result['opinions']:
                        if 'full_text' in opinion:
                            content = opinion['full_text']
                            break
                        elif 'plain_text' in opinion:
                            content = opinion['plain_text']
                            break
                        elif 'snippet' in opinion:
                            content = opinion['snippet']
                            break
                
                # If no content found, create a summary from available fields
                if not content:
                    content_parts = []
                    if case_name:
                        content_parts.append(f"Case: {case_name}")
                    if court:
                        content_parts.append(f"Court: {court}")
                    if date_filed:
                        content_parts.append(f"Date: {date_filed}")
                    if 'citations' in result:
                        content_parts.append(f"Citations: {result['citations']}")
                    if 'docket_number' in result:
                        content_parts.append(f"Docket: {result['docket_number']}")
                    if 'status' in result:
                        content_parts.append(f"Status: {result['status']}")
                    content = '\n'.join(content_parts)
                
                # Store document
                self.documents[doc_id] = {
                    'type': 'search_result',
                    'case_name': case_name,
                    'court': court,
                    'date_filed': date_filed,
                    'url': url,
                    'content': content,
                    'lines': content.split('\n') if content else [],
                    'metadata': result
                }
                
                document_ids.append(doc_id)
                logger.info(f"Stored document {doc_id}: {case_name}")
        
        return document_ids
    
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
    
    def get_available_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all available documents.
        
        Returns:
            List of document summaries
        """
        summaries = []
        for doc_id, doc in self.documents.items():
            summaries.append({
                'document_id': doc_id,
                'case_name': doc['case_name'],
                'court': doc['court'],
                'date_filed': doc['date_filed']
            })
        return summaries
    
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
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document metadata dictionary, or None if not found
        """
        if document_id not in self.documents:
            logger.error(f"Document {document_id} not found")
            return None
            
        doc = self.documents[document_id]
        return {
            'case_name': doc.get('case_name', ''),
            'court': doc.get('court', ''),
            'date_filed': doc.get('date_filed', ''),
            'url': doc.get('url', ''),
            'type': doc.get('type', ''),
            'metadata': doc.get('metadata', {})
        }


def summarize_search_results(results: List[Dict[str, Any]], search_type: str, total_count: int) -> List[str]:
    """
    Summarize search results from an observation.
    
    Args:
        results: List of search result dictionaries
        search_type: Type of search performed
        total_count: Total number of results found
        
    Returns:
        List of strings representing the summary parts
    """
    # Build summary of search results
    summary_parts = []
    summary_parts.append(f"Found {total_count} relevant {search_type}:")
    
    for i, result in enumerate(results):
        case_name = result.get('case_name', result.get('caseName', f'Result {i+1}'))
        court = result.get('court', 'Unknown Court')
        date_filed = result.get('date_filed', result.get('dateFiled', 'Unknown Date'))
        
        # Handle different result types
        if search_type == "opinions":
            snippet = result.get('snippet', '')[:200] + '...' if len(result.get('snippet', '')) > 200 else result.get('snippet', '')
            citations = result.get('citation', [])
            citation_text = '; '.join(citations) if citations else ''
            summary_parts.append(f"\n{i+1}. {case_name}")
            summary_parts.append(f"   Court: {court}")
            summary_parts.append(f"   Date: {date_filed}")
            if citation_text:
                summary_parts.append(f"   Citations: {citation_text}")
            if snippet:
                summary_parts.append(f"   Snippet: {snippet}")
            summary_parts.append(f"   Document ID: search_result_{i}")
            
        elif search_type in ["cases", "dockets"]:
            docket_number = result.get('docketNumber', '')
            more_docs = result.get('more_docs', False)
            summary_parts.append(f"\n{i+1}. {case_name}")
            summary_parts.append(f"   Court: {court}")
            summary_parts.append(f"   Date: {date_filed}")
            if docket_number:
                summary_parts.append(f"   Docket: {docket_number}")
            if more_docs:
                summary_parts.append(f"   Additional documents available")
            summary_parts.append(f"   Document ID: search_result_{i}")
            
        elif search_type == "filings":
            document_type = result.get('document_type', '')
            summary_parts.append(f"\n{i+1}. {case_name}")
            summary_parts.append(f"   Court: {court}")
            summary_parts.append(f"   Date: {date_filed}")
            if document_type:
                summary_parts.append(f"   Type: {document_type}")
            summary_parts.append(f"   Document ID: search_result_{i}")
            
        elif search_type == "judges":
            name = result.get('name', case_name)
            position = result.get('position', '')
            summary_parts.append(f"\n{i+1}. {name}")
            summary_parts.append(f"   Court: {court}")
            if position:
                summary_parts.append(f"   Position: {position}")
            summary_parts.append(f"   Document ID: search_result_{i}")
            
        elif search_type == "oral_arguments":
            date_argued = result.get('dateArgued', '')
            duration = result.get('duration', '')
            summary_parts.append(f"\n{i+1}. {case_name}")
            summary_parts.append(f"   Court: {court}")
            if date_argued:
                summary_parts.append(f"   Date Argued: {date_argued}")
            if duration:
                summary_parts.append(f"   Duration: {duration}")
            summary_parts.append(f"   Document ID: search_result_{i}")
            
        else:
            # Generic fallback
            snippet = result.get('snippet', '')[:200] + '...' if len(result.get('snippet', '')) > 200 else result.get('snippet', '')
            summary_parts.append(f"\n{i+1}. {case_name}")
            summary_parts.append(f"   Court: {court}")
            summary_parts.append(f"   Date: {date_filed}")
            if snippet:
                summary_parts.append(f"   Snippet: {snippet}")
            summary_parts.append(f"   Document ID: search_result_{i}")
    
    summary_parts.append(f"\nAll {len(results)} documents are now available to read using the READ_DOCUMENT action.")
    summary_parts.append("Use document IDs like 'search_result_0', 'search_result_1', etc. to access specific results.")
    
    return summary_parts 


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