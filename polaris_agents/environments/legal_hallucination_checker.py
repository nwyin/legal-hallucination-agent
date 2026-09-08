"""
Hallucination Checker Environment.
The environment process actions

"""

import json
import os
import re
import logging
import random
import string
import datetime
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple

from .base import Environment, Observation
from .utils import get_timestamp
from ..actions import ActionType, Action
from ..prompts.environments.legal_hallucination_checker import (
    get_response_requirements,
    get_search_capabilities_open_search,
)

from ..agents.action.open_search.search import search as search_web
from ..agents.action.courtlistener_search.main import (
    execute_courtlistener_search,
    execute_courtlistener_opinion_access,
    execute_courtlistener_citation_lookup,
)
from ..agents.action.filesystem.document_manager import (
    DocumentManager,
    read_document_content as read_document_content_fn,
)
logger = logging.getLogger(__name__)


class HallucinationCheckerEnvironment(Environment):
    """
    Legal Hallucination Checker environment that supports:
    - PROVIDE_FINAL_RESPONSE: Classify if citation is hallucinated or not
    - THINK: Internal reasoning (no environment interaction)
    - OPEN_WEB_SEARCH: Search the web for legal information
    - OPEN_COURTLISTENER_SEARCH: Search CourtListener
    - ACCESS_COURTLISTENER_OPINION: Fetch full opinion by ID (stored locally; agent sees snippet)
    - SEARCH_LOCAL_OPINION: Search for a string within a previously fetched opinion
    """
    
    # Snippet length returned to agent after ACCESS_COURTLISTENER_OPINION (full opinion saved to disk)
    OPINION_SNIPPET_LENGTH = 400
    # Context chars before/after a match for SEARCH_LOCAL_OPINION
    SEARCH_SNIPPET_CONTEXT = 200
    # Max number of match snippets to return for SEARCH_LOCAL_OPINION (first N hits)
    SEARCH_LOCAL_OPINION_MAX_SNIPPETS = 3
    # Fuzzy match: min ratio (0–1) to accept a sliding-window match; exact match is always used when present
    SEARCH_LOCAL_OPINION_FUZZY_MIN_RATIO = 0.6
    # Fuzzy match: step size (chars) when sliding over the opinion to limit cost
    SEARCH_LOCAL_OPINION_FUZZY_STEP = 30
    
    def __init__(self, 
                 brief_info: Dict[str, Any],
                 brief_text: str,
                 max_steps: int = 10, 
                 search_top_k: int = 3,
                 opinion_cache_dir: Optional[str] = None):
        """
        
        Args:
            brief_info: Dictionary containing brief information
            max_steps: Maximum number of steps allowed
            opinion_cache_dir: Directory to store full opinions (by opinion_id); default outputs/opinion_cache
        """
        # Initialize with required base class parameters
        key = (
            brief_info.get("list_hallucinations")
            or brief_info.get("listed_hallucinations")
            or brief_info.get("list_hallucinationss")
            or []
        )
        
        action_space = [
            ActionType.PROVIDE_FINAL_RESPONSE,
            ActionType.THINK,
            ActionType.OPEN_WEB_SEARCH,
            ActionType.OPEN_COURTLISTENER_SEARCH,
            ActionType.ACCESS_COURTLISTENER_OPINION,
            ActionType.COURTLISTENER_CITATION_LOOKUP,
            ActionType.SEARCH_LOCAL_OPINION,
            ActionType.READ_DOCUMENT,
            ActionType.EDIT_SCRATCHPAD,
        ]
        
        super().__init__(question="", key=key, action_space=action_space, max_steps=max_steps)
        
        # Potentially can pass more info here but not going to use it for now
        self.brief_info = brief_info
        self.brief_text = brief_text
        
        
        # Action space is already set by parent class
        
        # Track agent performance
        self.search_history = []

        # Number of results to return for search actions (from config search.top_k).
        # (Some Action models expose `num_results`; older code used `k`.)
        self.search_top_k = search_top_k
        
        # Directory to store full opinions by opinion_id (ACCESS_COURTLISTENER_OPINION); agent sees snippet only
        self.opinion_cache_dir = opinion_cache_dir or "outputs/opinion_cache"
        os.makedirs(self.opinion_cache_dir, exist_ok=True)

        # Document manager: registers fetched opinions for READ_DOCUMENT and holds scratchpad for EDIT_SCRATCHPAD
        self.document_manager = DocumentManager()
        
        self.initial_observation = Observation(
            result="Initial State.",
            metadata={
                "max_steps": self.max_steps,
                "available_actions": [action_type.value for action_type in self.action_space]
            }
        )
        
        logger.debug(f"Initialized Legal Hallucination Checker environment (max_steps={self.max_steps})")

    def step(self, action: Action) -> Observation:
        # Handle parsing failures by skipping the round
        if action is None:
            logger.warning("Agent returned None (parsing failed), skipping this round")
            return Observation(
                result="Round skipped due to parsing failure",
                metadata={"skipped": True, "reason": "parsing_failure"}
            )
        
        # Increment step counter (same as parent class)
        self.current_step += 1
        
        if action.action_type == ActionType.PROVIDE_FINAL_RESPONSE:
            # Handle legal prediction
            return self._handle_final_response_action(action)
            
        elif action.action_type == ActionType.THINK:
            # Handle internal reasoning - no environment interaction
            return self._handle_think_action(action)
            
        elif action.action_type == ActionType.OPEN_WEB_SEARCH:
            # Handle web search
            return self._handle_web_search_action(action)
        
        elif action.action_type == ActionType.OPEN_COURTLISTENER_SEARCH:
            return self._handle_courtlistener_search_action(action)
        
        elif action.action_type == ActionType.ACCESS_COURTLISTENER_OPINION:
            return self._handle_courtlistener_opinion_action(action)
        
        elif action.action_type == ActionType.COURTLISTENER_CITATION_LOOKUP:
            return self._handle_courtlistener_citation_lookup_action(action)
        
        elif action.action_type == ActionType.SEARCH_LOCAL_OPINION:
            return self._handle_search_local_opinion_action(action)
        elif action.action_type == ActionType.READ_DOCUMENT:
            return self._handle_read_document_action(action)
        elif action.action_type == ActionType.EDIT_SCRATCHPAD:
            return self._handle_edit_scratchpad_action(action)
        else:
            raise NotImplementedError(f"Action type {action.action_type} not implemented in HallucinationChecker environment.")
    
    def _handle_final_response_action(self, action: Action) -> Observation:
        response = getattr(action, "response", "")
        
        # Set the answer for the base class's is_correct() method
        self.answer = response
        
        # Terminate the episode after final response
        self.terminated = True
        
        # Use the custom is_correct method for ordered questions/answers
        # is_correct = self.is_correct()
        is_correct = 0
        observation = Observation(
            result=f"Legal prediction submitted: {response}",
            metadata={
                "action_type": "PROVIDE_FINAL_RESPONSE",
                "response": response,
                "is_correct": is_correct,
            }
        )
        return observation
    
    def _handle_think_action(self, action: Action) -> Observation:
        thought = getattr(action, "thought", "")
        
        observation = Observation(
            result=f"Internal reasoning: {thought}",
            metadata={
                "action_type": "THINK",
                "thought": thought
            }
        )
        return observation
    
    def _handle_web_search_action(self, action: Action) -> Observation:
        query = getattr(action, "query", "")
        search_type = getattr(action, "search_type", "web")  # Default to "web" if not specified
        k = (
            getattr(action, "num_results", None)
            or getattr(action, "k", None)
            or getattr(self, "search_top_k", 10)
        )
        
        try:
            # Get automatic date cutoff to prevent data leakage
            # cutoff_date = self._get_resolution_cutoff_date()
            
            # Perform web search with automatic date cutoff
            search_results = search_web(
                query=query, 
                search_type=search_type,  # Respect action's search_type parameter
                num_results=k,
                # cutoff_date=cutoff_date,  # Automatic date cutoff to prevent data leakage
                exclude_undated=True,  # Conservative: exclude undated results
            )
            
            # Create structured search results
            structured_results = []
            for i, result in enumerate(search_results):
                # Get snippet as contents (most relevant part from Google)
                contents = result.snippet if hasattr(result, 'snippet') else ""
                
                # Use position from search results (lower is better)
                # Check for web_position, google_scholar_position, or position in metadata
                position = None
                if hasattr(result, 'metadata'):
                    position = (result.metadata.get('web_position') or 
                              result.metadata.get('google_scholar_position') or 
                              result.metadata.get('position') or 
                              i+1)
                else:
                    position = i+1
                
                # Create structured result dict
                result_dict = result.to_dict()
                # Add position to metadata if not already there
                if 'metadata' not in result_dict:
                    result_dict['metadata'] = {}
                result_dict['metadata']['position'] = position
                structured_results.append(result_dict)
            
            # Create structured observation result (matching observation_metadata format)
            observation_result = {
                "action_type": "OPEN_WEB_SEARCH",
                "query": query,
                "search_type": search_type,
                "num_results": len(search_results),
                "search_results": structured_results
            }
            
            # Store search in history
            search_record = {
                "step": self.current_step,
                "query": query,
                "search_type": search_type,  # Use actual search_type from action
                "num_results": len(search_results),
                "timestamp": get_timestamp()
            }
            self.search_history.append(search_record)
            
            observation = Observation(
                result=observation_result,
                metadata={
                    "action_type": "OPEN_WEB_SEARCH",
                    "query": query,
                    "search_type": search_type,
                    "num_results": len(search_results),
                    "search_results": structured_results  # Use structured dicts for JSON serialization
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Web search failed: {error_msg}")
            observation = Observation(
                result=f"Web search failed: {error_msg}",
                metadata={
                    "action_type": "OPEN_WEB_SEARCH",
                    "query": query,
                    "error": error_msg,
                    "web_search_results": []
                }
            )
        
        return observation
    
    def _handle_courtlistener_search_action(self, action: Action) -> Observation:
        query = (getattr(action, "query", "") or "").strip()
        search_type = getattr(action, "search_type", "opinions")  # Default to "opinions" if not specified
        k = (
            getattr(action, "num_results", None)
            or getattr(action, "k", None)
            or getattr(self, "search_top_k", 10)
        )

        if not query:
            error_msg = "CourtListener search failed: missing query"
            logger.error(error_msg)
            return Observation(
                result=f"CourtListener search failed: {error_msg}",
                metadata={
                    "action_type": "OPEN_COURTLISTENER_SEARCH",
                    "query": query,
                    "error": "missing_query"
                }
            )

        try:
            search_observation = execute_courtlistener_search(query, search_type=search_type)
        except Exception as exc:
            error_msg = str(exc)
            logger.error(f"CourtListener search execution failed: {error_msg}")
            return Observation(
                result=f"CourtListener search failed: {error_msg}",
                metadata={
                    "action_type": "OPEN_COURTLISTENER_SEARCH",
                    "query": query,
                    "search_type": search_type,
                    "error": error_msg
                }
            )

        summary_payload = search_observation.result or {}
        if isinstance(summary_payload, dict) and summary_payload.get("error"):
            error_msg = summary_payload["error"]
            logger.error(f"CourtListener search failed: {error_msg}")
            return Observation(
                result=f"CourtListener search failed: {error_msg}",
                metadata={
                    "action_type": "OPEN_COURTLISTENER_SEARCH",
                    "query": query,
                    "search_type": search_type,
                    "error": error_msg,
                    "raw_metadata": search_observation.metadata or {}
                }
            )

        raw_results = []
        if isinstance(summary_payload, dict):
            raw_results = summary_payload.get("results", [])
        elif isinstance(summary_payload, list):
            raw_results = summary_payload

        structured_results = []
        for idx, result in enumerate(raw_results[:k]):
            if not isinstance(result, dict):
                continue
            metadata = dict(result.get("metadata") or {})
            metadata["position"] = idx + 1
            structured_results.append({
                "result_id": result.get("id") or result.get("case_name") or result.get("name") or result.get("docket_number"),
                "title": result.get("case_name_full") or result.get("case_name") or result.get("name") or result.get("document_type") or "Result",
                "url": result.get("url") or metadata.get("absolute_url"),
                "snippet": result.get("snippet"),
                "court": result.get("court"),
                "date_filed": result.get("date_filed") or result.get("date_argued"),
                "docket_number": result.get("docket_number"),
                "metadata": metadata
            })

        observation_result = {
            "action_type": "OPEN_COURTLISTENER_SEARCH",
            "query": query,
            "search_type": search_type,
            "num_results": len(structured_results),
            "search_results": structured_results
        }

        self.search_history.append({
            "step": self.current_step,
            "query": query,
            "search_type": "courtlistener",
            "num_results": len(structured_results),
            "timestamp": get_timestamp()
        })
        return Observation(
            result=observation_result,
            metadata={
                "action_type": "OPEN_COURTLISTENER_SEARCH",
                "query": query,
                "search_type": search_type,
                "num_results": len(structured_results),
                "search_results": structured_results,
                "raw_metadata": search_observation.metadata or {}
            }
        )

    def _handle_courtlistener_opinion_action(self, action: Action) -> Observation:
        opinion_id = (getattr(action, "opinion_id", "") or "").strip()
        if not opinion_id:
            error_msg = "ACCESS_COURTLISTENER_OPINION failed: missing opinion_id"
            logger.error(error_msg)
            return Observation(
                result={"error": error_msg, "opinion_id": None, "opinion": None},
                metadata={
                    "action_type": "ACCESS_COURTLISTENER_OPINION",
                    "opinion_id": None,
                    "error": "missing_opinion_id"
                }
            )
        try:
            obs = execute_courtlistener_opinion_access(opinion_id)
            result = obs.result if isinstance(obs.result, dict) else {}
            if result.get("error") or not result.get("opinion"):
                return obs
            opinion = result["opinion"]
            # Store full opinion to disk (filename = opinion_id)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(opinion_id))
            cache_path = os.path.join(self.opinion_cache_dir, f"{safe_id}.json")
            try:
                with open(cache_path, "w") as f:
                    json.dump(opinion, f, indent=2)
                logger.debug(f"Stored full opinion to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to write opinion cache {cache_path}: {e}")
            # Register opinion for READ_DOCUMENT (line-windowed reading)
            plain = self._get_searchable_opinion_text(opinion)
            if not plain:
                plain = opinion.get("plain_text", "") if isinstance(opinion, dict) else ""
            if not isinstance(plain, str):
                plain = str(opinion)
            case_name = opinion.get("case_name", "") if isinstance(opinion, dict) else ""
            self.document_manager.register_opinion(opinion_id, plain, case_name=case_name)
            # Return observation with snippet only (not full opinion)
            snippet = (plain[: self.OPINION_SNIPPET_LENGTH] + "...") if len(plain) > self.OPINION_SNIPPET_LENGTH else plain
            snippet_result = {
                "action_type": "ACCESS_COURTLISTENER_OPINION",
                "opinion_id": opinion_id,
                "case_name": case_name,
                "snippet": snippet,
                "stored": True,
                "message": (
                    f"Full opinion stored. Use SEARCH_LOCAL_OPINION with opinion_id={opinion_id} to search within it. "
                    f"Use READ_DOCUMENT with opinion_id=opinion_{opinion_id} (or {opinion_id}) to read the full text in sections (start_line, num_lines)."
                ),
            }
            return Observation(
                result=snippet_result,
                metadata={
                    "action_type": "ACCESS_COURTLISTENER_OPINION",
                    "opinion_id": opinion_id,
                    "stored_path": cache_path,
                }
            )
        except Exception as exc:
            error_msg = str(exc)
            logger.error(f"ACCESS_COURTLISTENER_OPINION failed: {error_msg}")
            return Observation(
                result={"error": error_msg, "opinion_id": opinion_id, "opinion": None},
                metadata={
                    "action_type": "ACCESS_COURTLISTENER_OPINION",
                    "opinion_id": opinion_id,
                    "error": error_msg
                }
            )

    def _handle_courtlistener_citation_lookup_action(self, action: Action) -> Observation:
        cite = (getattr(action, "cite", "") or "").strip()
        try:
            return execute_courtlistener_citation_lookup(cite)
        except Exception as exc:
            logger.error(f"COURTLISTENER_CITATION_LOOKUP failed: {exc}")
            return Observation(
                result={
                    "action_type": "COURTLISTENER_CITATION_LOOKUP",
                    "cite": cite,
                    "citations": [],
                    "results": None,
                    "error": str(exc),
                },
                metadata={
                    "action_type": "COURTLISTENER_CITATION_LOOKUP",
                    "cite": cite,
                    "error": str(exc),
                },
            )
            
    def _get_searchable_opinion_text(self, opinion: Any) -> str:
        if not isinstance(opinion, dict):
            return str(opinion)
        
        # Fall back to HTML/XML fields; strip tags for searchable text
        for key in ("html_lawbox", "html", "html_with_citations", "xml_harvard"):
            raw = opinion.get(key) or ""
            if not isinstance(raw, str) or len(raw.strip()) < 100:
                continue
            # Strip tags and collapse whitespace
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 100:
                return text
        return ""

    def _normalize_quotes_for_search(self, s: str) -> str:
        if not s:
            return s
        s = s.replace("\u2019", "'").replace("\u2018", "'")  # curly apostrophes
        s = s.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
        return s

    def _handle_search_local_opinion_action(self, action: Action) -> Observation:
        opinion_id = (getattr(action, "opinion_id", "") or "").strip()
        search_string = (getattr(action, "search_string", "") or "").strip()
        if not opinion_id:
            return Observation(
                result={"action_type": "SEARCH_LOCAL_OPINION", "opinion_id": None, "snippet": None, "error": "missing opinion_id"},
                metadata={"action_type": "SEARCH_LOCAL_OPINION", "error": "missing_opinion_id"},
            )
        if not search_string:
            return Observation(
                result={"action_type": "SEARCH_LOCAL_OPINION", "opinion_id": opinion_id, "snippet": None, "error": "missing search_string"},
                metadata={"action_type": "SEARCH_LOCAL_OPINION", "error": "missing_search_string"},
            )
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(opinion_id))
        cache_path = os.path.join(self.opinion_cache_dir, f"{safe_id}.json")
        if not os.path.isfile(cache_path):
            return Observation(
                result={"action_type": "SEARCH_LOCAL_OPINION", "opinion_id": opinion_id, "snippet": None, "error": "opinion not in cache (fetch with ACCESS_COURTLISTENER_OPINION first)"},
                metadata={"action_type": "SEARCH_LOCAL_OPINION", "error": "not_cached"},
            )
        try:
            with open(cache_path) as f:
                opinion = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read opinion cache {cache_path}: {e}")
            return Observation(
                result={"action_type": "SEARCH_LOCAL_OPINION", "opinion_id": opinion_id, "snippet": None, "error": str(e)},
                metadata={"action_type": "SEARCH_LOCAL_OPINION", "error": "read_error"},
            )
        plain = self._get_searchable_opinion_text(opinion)
        # Normalize curly quotes/apostrophes to straight so brief and opinion match (U+2019 vs U+0027, etc.)
        plain = self._normalize_quotes_for_search(plain)
        search_string = self._normalize_quotes_for_search(search_string)
        search_lower = search_string.lower()
        # 1) Try exact (case-insensitive) matches first; take up to MAX_SNIPPETS
        indices = []
        start_idx = 0
        while len(indices) < self.SEARCH_LOCAL_OPINION_MAX_SNIPPETS:
            idx = plain.lower().find(search_lower, start_idx)
            if idx == -1:
                break
            indices.append(idx)
            start_idx = idx + 1
        # 2) If no exact match, use fuzzy matching (sliding window)
        if not indices:
            indices = self._fuzzy_find_in_text(
                plain, search_string, self.SEARCH_LOCAL_OPINION_MAX_SNIPPETS,
                self.SEARCH_SNIPPET_CONTEXT, self.SEARCH_LOCAL_OPINION_FUZZY_MIN_RATIO,
                self.SEARCH_LOCAL_OPINION_FUZZY_STEP,
            )
        if not indices:
            return Observation(
                result={"action_type": "SEARCH_LOCAL_OPINION", "opinion_id": opinion_id, "search_string": search_string, "snippet": None, "found": False},
                metadata={"action_type": "SEARCH_LOCAL_OPINION", "found": False},
            )
        snippets = []
        for i, idx in enumerate(indices):
            start = max(0, idx - self.SEARCH_SNIPPET_CONTEXT)
            end = min(len(plain), idx + len(search_string) + self.SEARCH_SNIPPET_CONTEXT)
            part = (("..." if start > 0 else "") + plain[start:end] + ("..." if end < len(plain) else ""))
            if len(indices) > 1:
                part = f"--- Match {i + 1} ---\n{part}"
            snippets.append(part)
        snippet = "\n\n".join(snippets)
        return Observation(
            result={"action_type": "SEARCH_LOCAL_OPINION", "opinion_id": opinion_id, "search_string": search_string, "snippet": snippet, "found": True, "match_count": len(indices)},
            metadata={"action_type": "SEARCH_LOCAL_OPINION", "found": True, "match_count": len(indices)},
        )

    def _fuzzy_find_in_text(
        self,
        plain: str,
        search_string: str,
        max_snippets: int,
        context_chars: int,
        min_ratio: float,
        step: int,
    ) -> List[int]:
        """Return up to max_snippets start indices of best fuzzy matches of search_string in plain (sliding window)."""
        if not search_string or not plain:
            return []
        q = search_string.strip().lower()
        if len(q) < 3:
            return []
        # Window length: query length + context on both sides so each window is snippet-sized
        window_len = min(len(q) + 2 * context_chars, len(plain))
        if window_len > len(plain):
            window_len = len(plain)
        step = max(1, min(step, window_len // 2))
        candidates: List[Tuple[int, float]] = []
        for start in range(0, len(plain) - window_len + 1, step):
            window = plain[start : start + window_len].lower()
            ratio = SequenceMatcher(None, q, window).ratio()
            if ratio >= min_ratio:
                candidates.append((start, ratio))
        if not candidates:
            return []
        # Sort by ratio descending, then take top max_snippets; optionally merge overlapping
        candidates.sort(key=lambda x: -x[1])
        chosen: List[int] = []
        for pos, _ in candidates:
            if len(chosen) >= max_snippets:
                break
            # Skip if too close to an already chosen position (avoid duplicate snippets)
            if any(abs(pos - c) < window_len // 2 for c in chosen):
                continue
            chosen.append(pos)
        chosen.sort()
        return chosen

    def _handle_read_document_action(self, action: Action) -> Observation:
        opinion_id = (getattr(action, "opinion_id", None) or getattr(action, "document_id", "") or "").strip()
        start_line = getattr(action, "start_line", 0)
        num_lines = getattr(action, "num_lines", 50)
        if not opinion_id:
            logger.info("READ_DOCUMENT result: error=missing opinion_id, content=None")
            return Observation(
                result={"action_type": "READ_DOCUMENT", "error": "missing opinion_id", "content": None},
                metadata={"action_type": "READ_DOCUMENT", "error": "missing_opinion_id"},
            )
        key = self.document_manager.resolve_document_id(opinion_id)
        if key is None:
            available = [k for k in self.document_manager.documents if k.startswith("opinion_")]
            logger.info(
                "READ_DOCUMENT result: opinion_id=%s error=document not found, available=%s, content=None",
                opinion_id,
                available,
            )
            return Observation(
                result={
                    "action_type": "READ_DOCUMENT",
                    "opinion_id": opinion_id,
                    "error": f"document not found (available: {available})",
                    "content": None,
                },
                metadata={"action_type": "READ_DOCUMENT", "error": "not_found"},
            )
        content = self.document_manager.get_document_content(key)
        if content is None:
            logger.info("READ_DOCUMENT result: opinion_id=%s error=empty document, content=None", key)
            return Observation(
                result={"action_type": "READ_DOCUMENT", "opinion_id": opinion_id, "error": "empty document", "content": None},
                metadata={"action_type": "READ_DOCUMENT", "error": "empty"},
            )
        try:
            start_line = max(0, int(start_line))
            num_lines = max(1, min(500, int(num_lines)))
        except (TypeError, ValueError):
            start_line, num_lines = 0, 50
        text, actual_start, actual_end, total_lines = read_document_content_fn(
            content, opinion_id, start_line, num_lines
        )
        # Log what is being returned for READ_DOCUMENT
        preview_len = 400
        content_preview = (text[:preview_len] + "...") if len(text) > preview_len else text
        logger.info(
            "READ_DOCUMENT result: opinion_id=%s lines %s-%s of %s, content_length=%d. Preview: %s",
            key,
            actual_start,
            actual_end,
            total_lines,
            len(text),
            repr(content_preview),
        )
        return Observation(
            result={
                "action_type": "READ_DOCUMENT",
                "opinion_id": key,
                "content": text,
                "start_line": actual_start,
                "end_line": actual_end,
                "total_lines": total_lines,
            },
            metadata={"action_type": "READ_DOCUMENT", "found": True},
        )

    def _handle_edit_scratchpad_action(self, action: Action) -> Observation:
        operation = (getattr(action, "operation", "") or "").strip().lower()
        content = getattr(action, "content", "") or ""
        position = getattr(action, "position", None)
        if not operation:
            return Observation(
                result={"action_type": "EDIT_SCRATCHPAD", "error": "missing operation", "scratchpad": self.document_manager.get_scratchpad_content()},
                metadata={"action_type": "EDIT_SCRATCHPAD", "error": "missing_operation"},
            )
        ok = self.document_manager.edit_scratchpad(operation, content, position)
        scratchpad_now = self.document_manager.get_scratchpad_content()
        if not ok:
            return Observation(
                result={"action_type": "EDIT_SCRATCHPAD", "error": "edit failed", "scratchpad": scratchpad_now},
                metadata={"action_type": "EDIT_SCRATCHPAD", "success": False},
            )
        return Observation(
            result={
                "action_type": "EDIT_SCRATCHPAD",
                "operation": operation,
                "success": True,
                "scratchpad": scratchpad_now,
            },
            metadata={"action_type": "EDIT_SCRATCHPAD", "success": True},
        )

    def get_environment_description(self) -> str:
        description =f"""
You are expected to extract all CASE CITATIONS (no other types of citations like regulations, statutes, or other legal sources) and to:
1) Verify Citation Existence: Determine whether the reporter citations correspond to real, verifiable legal cases.
2) Verify Citation Consistency: Determine whether the case names before the reporter citations and the reporter citations themselves refer to the same legal case.
3) Verify Quotation: Determine whether the quoted language appears verbatim in the cited opinion.
4) Verify Pincite: Determine whether the pincite (page number) accurately reflects the location of the quoted language or proposition within the cited opinion.
5) Verify Contextual Accuracy: Assess whether the quoted language or proposition is presented in a manner consistent with its original context within the cited opinions.

BRIEF TEXT: {self.brief_text}\n\n"""
                
        return description

    def get_action_selection_environment_description(self) -> str:
        return self.get_environment_description()


    def get_response_requirements(self) -> str:
        return get_response_requirements()
    
    def get_search_capabilities(self) -> str:
        return get_search_capabilities_open_search()
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        return self.search_history.copy()
    
    def _parse_response_to_list(self, response: str) -> List[str]:
        if not response or not str(response).strip():
            return []
        s = str(response).strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x is not None]
            except (json.JSONDecodeError, TypeError):
                pass
        return [a.strip() for a in s.split(";") if a.strip()]

    def is_correct(self) -> float:
        """Check accuracy of the current answer using ordered questions/answers logic.
        
        Returns:
            float: Accuracy as fraction of correct answers (0.0 to 1.0)
        """
        if not self.key or not self.answer:
            return 0.0
        
        # Parse the response to extract predicted answers
        # Prefer JSON array (e.g. '["cite1", "cite2"]'); fallback to semicolon-separated for backward compat
        predicted_answers_raw = self._parse_response_to_list(self.answer)
        
        if not predicted_answers_raw:
            return 0.0
        
        # Normalize predicted answers (handle variations like yes, Yes, YES, Y)
        def normalize_answer(a: str) -> str:
            a_upper = a.upper()
            if a_upper in ['YES', 'Y', 'TRUE']:
                return 'TRUE'
            elif a_upper in ['NO', 'N', 'FALSE']:
                return 'FALSE'
            return a
        
        predicted_answer = normalize_answer(predicted_answers_raw[0])
        return float(predicted_answer == self.key)
    
    def is_done(self) -> bool:
        if self.is_truncated():
            return True
        
        # Check if a final prediction was made (base class sets terminated=True)
        if self.is_terminated():
            return True
        
        return False
    def get_initial_observation(self) -> Observation:
        return self.initial_observation
    
    def reset(self):
        super().reset()
        self.current_step = 0
        self.search_history = []
        self.document_manager.documents.clear()
        self.document_manager.scratchpad.clear()
        self.document_manager.scratchpad_counter = 0

        self.initial_observation = Observation(
            result="Initial State.",
            metadata={
                "max_steps": self.max_steps,
                "available_actions": [action_type.value for action_type in self.action_space]
            }
        )
        logger.info("Environment reset")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "has_answer": bool(self.answer),
            "num_searches": len(self.search_history),
            "is_done": self.is_done()
        }
