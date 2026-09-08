"""
Action Classification Module for Evaluating Task vs Design Focus

This module provides LLM-based evaluation of actions to determine whether they
are more focused on design-level or task-level information gathering.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from polaris_agents.models.llm import ModelAPI
from polaris_agents.prompts.base import DomainKnowledgeProvider

logger = logging.getLogger(__name__)

@dataclass
class ActionClassificationResult:
    """Result of action classification evaluation"""
    step: int
    action_type: str
    action_content: str
    task_focus_score: float  # 0.0 = pure design focus, 1.0 = pure task focus
    reasoning: str
    confidence: float  # 0.0 to 1.0, how confident the LLM is in this classification
    label: str = "UNKNOWN"  # TASK, DESIGN, or MIXED

class ActionClassifier:
    """
    LLM-based classifier for evaluating whether actions focus on task-level 
    or design-level effectiveness information gathering.
    """
    
    def __init__(
        self, 
        model_api: ModelAPI, 
        model_name: str = "openai/gpt-4o-mini",
        provider: str = "openrouter",
        max_tokens: int = 4000,  # Output tokens only - responses are short JSON with scores and reasoning
        domain_knowledge: Optional[DomainKnowledgeProvider] = None
    ):
        """
        Initialize the action classifier.
        
        Args:
            model_api: Model API for LLM calls
            model_name: Model to use for classification (default: gpt-4o-mini)
            provider: Model provider to use (default: openrouter)
            max_tokens: Maximum tokens for LLM responses (default: 10000)
            domain_knowledge: Optional domain knowledge provider for task-specific θ/D definitions
        """
        self.model_api = model_api
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.domain_knowledge = domain_knowledge
        self.classification_history: List[ActionClassificationResult] = []
    
    @staticmethod
    def generate_context_from_environment(environment, task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate context for action classification from the environment.
        
        Args:
            environment: The environment instance
            task_name: Optional task name (e.g., "multi_armed_bandit", "scotus_judgment_prediction")
            
        Returns:
            Dictionary with environment_info (task_name only if provided)
        """
        # Get environment description (already includes task instance info)
        env_description = ""
        if hasattr(environment, 'get_environment_description'):
            env_description = environment.get_environment_description()
        
        # Build result dictionary
        result = {
            "environment_info": env_description
        }
        
        # Only add task_name if explicitly provided
        if task_name:
            result["task_name"] = task_name
        
        return result
        
    def classify_action(
        self, 
        step: int, 
        action, 
        context: Optional[Dict[str, Any]] = None,
        environment: Optional[Any] = None,
        task_name: Optional[str] = None
    ) -> ActionClassificationResult:
        """
        Classify an action's focus using LLM-as-a-judge.
        
        Args:
            step: Current step number
            action: The action object to classify
            context: Optional context about the task/environment
            environment: Optional environment instance to automatically generate context from
            task_name: Optional task name (e.g., "multi_armed_bandit", "scotus_judgment_prediction")
            
        Returns:
            ActionClassificationResult with task focus score and reasoning
        """
        # Generate context from environment if provided and no context given
        if environment is not None and context is None:
            context = self.generate_context_from_environment(environment, task_name)
        
        # Extract action information
        action_type = action.action_type.value if hasattr(action, 'action_type') else str(type(action))
        action_content = self._extract_action_content(action)
        
        # Create classification prompt
        prompt = self._create_classification_prompt(
            action_type, action_content, context, step, self.domain_knowledge
        )
        
        try:
            # Get LLM classification
            logger.info(f"Calling ModelAPI with prompt: {prompt[:200]}...")
            response = self.model_api(
                model_id=self.model_name,
                prompt=[{"role": "user", "content": prompt}],  # Format as chat messages
                max_attempts=3,
                provider=self.provider,
                max_tokens=self.max_tokens,
                temperature=0.1  # Low temperature for consistent classification
            )
            logger.info(f"ModelAPI response: {response}")
            
            # Parse response
            result = self._parse_classification_response(
                response, step, action_type, action_content
            )
            
            # Store in history
            self.classification_history.append(result)
            
            logger.info(f"Step {step}: Action classified as {result.task_focus_score:.3f} task focus "
                       f"(confidence: {result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to classify action at step {step}: {e}")
            # Return default result on failure
            return ActionClassificationResult(
                step=step,
                action_type=action_type,
                action_content=action_content,
                task_focus_score=0.5,  # Neutral score
                reasoning=f"Classification failed: {str(e)}",
                confidence=0.0,
                label="UNKNOWN"
            )
    
    def _extract_action_content(self, action) -> str:
        content_parts = []
        
        # Extract action type
        if hasattr(action, 'action_type'):
            content_parts.append(f"Action Type: {action.action_type.value}")
        
        # Extract action-specific content
        if hasattr(action, 'query') and action.query:
            content_parts.append(f"Query: {action.query}")
        if hasattr(action, 'thought') and action.thought:
            content_parts.append(f"Thought: {action.thought}")
        if hasattr(action, 'response') and action.response:
            content_parts.append(f"Response: {action.response}")
        if hasattr(action, 'search_type') and action.search_type:
            content_parts.append(f"Search Type: {action.search_type}")
        
        return "\n".join(content_parts) if content_parts else str(action)
    
    def _create_classification_prompt(
        self, 
        action_type: str, 
        action_content: str, 
        context: Optional[Dict[str, Any]], 
        step: int,
        domain_knowledge: Optional[DomainKnowledgeProvider] = None
    ) -> str:
        """Prompt for classifying whether an action targets design-level (D) or task-level (θ) information."""

        context_str = ""
        if context:
            task_name = context.get("task_name", "unknown task")
            environment_info = context.get("environment_info", "")
            current_beliefs = context.get("current_beliefs")
            
            beliefs_str = ""
            if current_beliefs:
                task_beliefs = current_beliefs.get('task_beliefs', 'No task-level beliefs yet.')
                design_beliefs = current_beliefs.get('design_beliefs', 'No design-level beliefs yet.')
                beliefs_str = f"""
CURRENT BELIEFS (Agent's internal state before this action)
- Task Beliefs (θ): {task_beliefs}
- Design Beliefs (D): {design_beliefs}
"""
            
            context_str = f"""CONTEXT
- Task: {task_name}
- Environment: {environment_info}
- Step: {step}
{beliefs_str}
"""

        # Use canonical definitions
        from polaris_agents.prompts.utils import get_canonical_theta_definition, get_canonical_design_definition
        
        theta_def = get_canonical_theta_definition()
        design_def = get_canonical_design_definition()
        
        definitions = f"""INFORMATION SPACE
For a given task instance, we split the information space into two components:
- **Task parameters (θ)**: {theta_def}
- **Design parameters (D)**: {design_def}

Your task is to classify which information space an action primarily targets."""

        prompt = f"""You are classifying an agent action by its primary information target.

{context_str if context_str else ""}

ACTION
- Type: {action_type}
- Content: {action_content}

{definitions}

CORE CLASSIFICATION PRINCIPLE

Classify based on what the action DOES, not what it mentions:
- Actions that **gather task-specific facts/evidence** → TASK-focused (higher score)
- Actions that **plan, strategize, or learn general methods** → DESIGN-focused (lower score)
"""
        
        # Add domain-specific classification rubric (the detailed task-specific guidance)
        domain_guidance = ""
        if domain_knowledge:
            # Add classification-specific guidance (contains the task-specific rubric)
            if hasattr(domain_knowledge, 'get_classification_guidance'):
                classification_guidance = domain_knowledge.get_classification_guidance()
                if classification_guidance:
                    domain_guidance = classification_guidance
        
        prompt = prompt + domain_guidance + """

## OUTPUT INSTRUCTIONS

Provide a **task_focus_score** from 0.0 to 1.0:
- 0.0 = pure DESIGN focus (planning, strategy, methodology)
- 1.0 = pure TASK focus (gathering/analyzing task-specific facts)

Assign a **label** based on the score:
- DESIGN: task_focus_score < 0.4
- MIXED: task_focus_score 0.4-0.6
- TASK: task_focus_score > 0.6

Note: design_focus_score = 1 - task_focus_score (you only need to provide task_focus_score).

Also provide:
- **reasoning**: Brief explanation of why this score was assigned
- **confidence**: How confident you are in this classification (0.0-1.0)
"""
        
        prompt = prompt + """

RESPONSE FORMAT (JSON only)
{{
  "task_focus_score": <float between 0.0 and 1.0>,
  "label": "TASK" | "DESIGN" | "MIXED",
  "reasoning": "<≤1 sentence>",
  "confidence": <float between 0.0 and 1.0>
}}"""
        return prompt
    
    def _parse_classification_response(
        self, 
        response: str, 
        step: int, 
        action_type: str, 
        action_content: str
    ) -> ActionClassificationResult:
        """Parse the LLM response into an ActionClassificationResult."""
        
        try:
            # Handle None response
            if response is None:
                raise ValueError("ModelAPI returned None response")
            
            # Try to extract JSON from response
            response_clean = str(response).strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            data = json.loads(response_clean)
            
            # Validate and extract values
            task_focus_score = float(data.get('task_focus_score', 0.5))
            task_focus_score = max(0.0, min(1.0, task_focus_score))  # Clamp to [0,1]
            
            # Handle both old and new response formats
            reasoning = str(data.get('rationale', data.get('reasoning', 'No reasoning provided')))
            label = str(data.get('label', 'UNKNOWN'))
            
            confidence = float(data.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            return ActionClassificationResult(
                step=step,
                action_type=action_type,
                action_content=action_content,
                task_focus_score=task_focus_score,
                reasoning=reasoning,
                confidence=confidence,
                label=label
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse classification response at step {step}: {e}")
            # Fallback parsing - try to extract numbers from text
            try:
                import re
                score_match = re.search(r'task_focus_score["\']?\s*:\s*([0-9.]+)', response)
                confidence_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', response)
                
                task_focus_score = float(score_match.group(1)) if score_match else 0.5
                confidence = float(confidence_match.group(1)) if confidence_match else 0.3
                
                return ActionClassificationResult(
                    step=step,
                    action_type=action_type,
                    action_content=action_content,
                    task_focus_score=max(0.0, min(1.0, task_focus_score)),
                    reasoning=f"Parsed from text (JSON parse failed): {response[:200]}",
                    confidence=max(0.0, min(1.0, confidence)),
                    label="UNKNOWN"
                )
            except Exception:
                # Ultimate fallback
                return ActionClassificationResult(
                    step=step,
                    action_type=action_type,
                    action_content=action_content,
                    task_focus_score=0.5,
                    reasoning=f"Failed to parse response: {response[:200]}",
                    confidence=0.0,
                    label="UNKNOWN"
                )
    
    def get_classification_summary(self) -> Dict[str, Any]:
        if not self.classification_history:
            return {"total_classifications": 0}
        
        scores = [c.task_focus_score for c in self.classification_history]
        confidences = [c.confidence for c in self.classification_history]
        
        return {
            "total_classifications": len(self.classification_history),
            "average_task_focus": sum(scores) / len(scores),
            "average_confidence": sum(confidences) / len(confidences),
            "task_focus_std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            "classifications": [
                {
                    "step": c.step,
                    "action_type": c.action_type,
                    "task_focus_score": c.task_focus_score,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                    "label": c.label
                }
                for c in self.classification_history
            ]
        }
    
    def save_classifications(self, filepath: str) -> None:
        summary = self.get_classification_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Action classifications saved to {filepath}")
    
    def load_classifications(self, filepath: str) -> None:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.classification_history = [
                ActionClassificationResult(
                    step=c["step"],
                    action_type=c["action_type"],
                    action_content="",  # Not stored in summary
                    task_focus_score=c["task_focus_score"],
                    reasoning=c["reasoning"],
                    confidence=c["confidence"],
                    label=c.get("label", "UNKNOWN")  # Handle old files without label
                )
                for c in data.get("classifications", [])
            ]
            logger.info(f"Loaded {len(self.classification_history)} classifications from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load classifications from {filepath}: {e}")
