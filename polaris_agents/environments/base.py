import logging
from typing import Any, Optional, Dict, List
from ..agents.action import Action, ActionType
from .utils import EM

logger = logging.getLogger(__name__)


class Observation:
    def __init__(self, result: Any, metadata: Optional[Dict[str, Any]] = None):
        self.result = result
        self.metadata = metadata or {}

    def __str__(self):
        """String representation for prompts - formats result nicely for LLM consumption."""
        if isinstance(self.result, dict):
            # Format structured results (like search results) in a readable way
            if 'action_type' in self.result and 'search_results' in self.result:
                # Format search results
                lines = []
                lines.append(f"Action: {self.result.get('action_type', 'UNKNOWN')}")
                if 'query' in self.result:
                    lines.append(f"Query: {self.result['query']}")
                if 'search_type' in self.result:
                    lines.append(f"Search Type: {self.result['search_type']}")
                lines.append(f"Found {self.result.get('num_results', 0)} results:\n")
                
                for i, result_item in enumerate(self.result.get('search_results', []), 1):
                    lines.append(f"\nResult {i}:")
                    if 'result_id' in result_item:
                        lines.append(f"  ID: {result_item['result_id']}")
                    if 'title' in result_item:
                        lines.append(f"  Title: {result_item['title']}")
                    if 'url' in result_item:
                        lines.append(f"  URL: {result_item['url']}")
                    if 'snippet' in result_item:
                        snippet = result_item.get('snippet') or ''
                        lines.append(f"  Snippet: {snippet[:200]}..." if len(snippet) > 200 else f"  Snippet: {snippet}")
                    if 'published_date_raw' in result_item:
                        lines.append(f"  Published: {result_item['published_date_raw']}")
                    if 'metadata' in result_item and 'contents' in result_item['metadata']:
                        contents = result_item['metadata']['contents'] or ''
                        # Contents are already truncated by environment, but show indication if very long
                        lines.append(f"  Contents: {contents[:500]}..." if len(contents) > 500 else f"  Contents: {contents}")
                    if 'metadata' in result_item and 'score' in result_item['metadata']:
                        lines.append(f"  Score: {result_item['metadata']['score']:.3f}")
                    if 'metadata' in result_item:
                        metadata = result_item['metadata']
                        if 'subdomain' in metadata:
                            lines.append(f"  Topic: {metadata['subdomain']}")
                        if 'position' in metadata:
                            lines.append(f"  Position: {metadata['position']}")
                
                return "\n".join(lines)
            else:
                # Generic dict formatting
                import json
                return json.dumps(self.result, indent=2)
        else:
            # For strings or other types, return as string
            return str(self.result)
    
    def __repr__(self):
        """Full representation for debugging - includes metadata."""
        return f"Observation(result={self.result}, metadata={self.metadata})"


class Environment():
    def __init__(self, question: str, key: str, action_space: List[ActionType], max_steps: int = 6):
        self.question = question
        self.key = key
        self.max_steps = max_steps
        self.action_space = action_space
        # Initialize basic state attributes directly
        self.answer = ""
        self.current_step = 0
        self.terminated = False
        
    def reset(self):
        self.answer = ""
        self.current_step = 0
        self.terminated = False
        
    def is_correct(self):
        return EM(self.answer, self.key)
    
    def is_terminated(self):
        return self.terminated
    
    def is_truncated(self):
        return self.current_step >= self.max_steps
    
    def step(self, action: Action) -> Observation:
        """Execute an action and return the resulting observation."""
        # Handle parsing failures by skipping the round
        if action is None:
            logger.warning("Agent returned None (parsing failed), skipping this round")
            return Observation(
                result="Round skipped due to parsing failure",
                metadata={"skipped": True, "reason": "parsing_failure"}
            )
        
        self.current_step += 1
        
        if action.action_type == ActionType.PROVIDE_FINAL_RESPONSE:
            # Capture the final answer
            self.answer = getattr(action, "answer", "")
            observation = Observation(
                result=f"Agent's response: {self.answer}",
                metadata={"answer": self.answer, "key": self.key, "correct": self.is_correct()}
            )
            self.terminated = True
            return observation
            
        elif action.action_type == ActionType.THINK:
            # Allow internal reasoning
            prompt = getattr(action, "prompt", "")
            observation = Observation(
                result=f"Agent considers: {prompt}",
                metadata={"prompt": prompt}
            )
            return observation

        else:
            raise NotImplementedError(f"Action type {action.action_type} not implemented in base Environment.")

