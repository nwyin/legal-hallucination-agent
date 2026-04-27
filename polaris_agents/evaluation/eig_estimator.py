"""
True Information Gain (IG) Estimation Module (Post-Hoc Evaluation)

This module provides LLM-as-a-judge estimation of TRUE information gain for actions,
evaluated post-hoc after the observation is received. This estimates actual information
gain about task space (θ) and design space (D), not expected information gain.
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from polaris_agents.models.llm import ModelAPI
from polaris_agents.prompts.base import DomainKnowledgeProvider

logger = logging.getLogger(__name__)

@dataclass
class EIGEstimationResult:
    """Result of true information gain estimation for an action"""
    step: int
    timestamp: str
    action_type: str
    action_content: str
    task_eig: float
    design_eig: float
    joint_eig: float
    reasoning: str
    confidence: float

class EIGEstimator:
    """
    Estimates TRUE information gain (post-hoc) for actions using an external LLM.
    This evaluates actual information gain after the observation is received,
    not expected information gain before the action.
    """
    
    def __init__(
        self,
        model_api: ModelAPI,
        model_name: str = "gpt-4o-mini",
        provider: str = "openai",
        max_tokens: int = 4000,  # Output tokens only - responses are short JSON with scores and reasoning
        domain_knowledge: Optional[DomainKnowledgeProvider] = None
    ):
        """
        Initialize the EIG estimator.
        
        Args:
            model_api: Model API for LLM calls
            model_name: Model to use for estimation (default: gpt-4o-mini)
            provider: Model provider to use (default: openai)
            max_tokens: Maximum tokens for LLM responses (default: 20000)
            domain_knowledge: Optional domain knowledge provider for task-specific θ/D definitions
        """
        self.model_api = model_api
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.domain_knowledge = domain_knowledge
        self.estimations = []
    
    def estimate_eig_at_step(
        self, 
        step: int, 
        agent: Any,
        action: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> EIGEstimationResult:
        """
        Estimate TRUE information gain (post-hoc) for a given action using external LLM.
        
        Args:
            step: Current step number
            agent: The agent instance (not used, kept for compatibility)
            action: The action to estimate information gain for
            context: Context including task_name, environment_info, current_beliefs, observation_result, observation_metadata
            
        Returns:
            EIGEstimationResult with task, design, and joint information gain estimates
        """
        try:
            # Extract action information
            action_type = action.action_type.value if hasattr(action, 'action_type') else str(type(action))
            action_content = self._extract_action_content(action)
            
            # Extract context information
            task_name = context.get("task_name", "unknown task") if context else "unknown task"
            environment_info = context.get("environment_info", "") if context else ""
            current_beliefs = context.get("current_beliefs") if context else None
            observation_result = context.get("observation_result") if context else None
            observation_metadata = context.get("observation_metadata") if context else None
            
            # Create estimation prompt
            prompt = self._create_estimation_prompt(
                action_type=action_type,
                action_content=action_content,
                task_name=task_name,
                environment_info=environment_info,
                current_beliefs=current_beliefs,
                observation_result=observation_result,
                observation_metadata=observation_metadata,
                step=step
            )
            
            # Get LLM estimation
            logger.info(f"Calling ModelAPI for true information gain estimation at step {step}")
            response = self.model_api(
                model_id=self.model_name,
                prompt=[{"role": "user", "content": prompt}],
                max_attempts=3,
                provider=self.provider,
                max_tokens=self.max_tokens,
                temperature=0.1  # Low temperature for consistent estimation
            )
            logger.info(f"ModelAPI response: {response}")
            
            # Parse response
            result = self._parse_estimation_response(
                response, step, action_type, action_content
            )
            
            # Store in history
            self.estimations.append(result)
            
            logger.info(f"Step {step}: True IG estimated - task={result.task_eig:.3f}, design={result.design_eig:.3f}, joint={result.joint_eig:.3f}")
            
            return result
            
        except Exception as e:
            logger.exception(f"True information gain estimation failed at step {step}: {e}")
            return EIGEstimationResult(
                step=step,
                timestamp=datetime.now().isoformat(),
                action_type=str(type(action)),
                action_content="",
                task_eig=0.0,
                design_eig=0.0,
                joint_eig=0.0,
                confidence=0.0,
                reasoning=f"Estimation failed: {str(e)}"
            )
    
    def _extract_action_content(self, action) -> str:
        """Extract relevant content from action."""
        content_parts = []
        
        if hasattr(action, 'query') and action.query:
            content_parts.append(f"Query: {action.query}")
        if hasattr(action, 'thought') and action.thought:
            content_parts.append(f"Thought: {action.thought}")
        if hasattr(action, 'response') and action.response:
            content_parts.append(f"Response: {action.response}")
        if hasattr(action, 'search_type') and action.search_type:
            content_parts.append(f"Search Type: {action.search_type}")
        
        return "\n".join(content_parts) if content_parts else str(action)
    
    def _create_estimation_prompt(
        self,
        action_type: str,
        action_content: str,
        task_name: str,
        environment_info: str,
        current_beliefs: Optional[Dict[str, Any]],
        observation_result: Optional[Any],
        observation_metadata: Optional[Dict[str, Any]],
        step: int
    ) -> str:
        """Create prompt for estimating true information gain (post-hoc)."""
        
        # Build context section
        beliefs_str = ""
        if current_beliefs:
            task_beliefs = current_beliefs.get('task_beliefs', 'No task-level beliefs yet.')
            design_beliefs = current_beliefs.get('design_beliefs', 'No design-level beliefs yet.')
            beliefs_str = f"""
BELIEFS BEFORE ACTION (Agent's state before taking this action)
- Task Beliefs (θ): {task_beliefs}
- Design Beliefs (D): {design_beliefs}
"""
        
        observation_str = ""
        if observation_result is not None:
            # Format observation to include both result and metadata
            # The environment now creates structured observation.result (dict format)
            # with truncated contents (26,500-42,000 chars per doc) to avoid context overflow
            # We format the full observation (result + metadata) to match what the agent sees
            # This ensures the EIG estimator has the same information as the agent for accurate estimation
            
            # Format the observation result
            if isinstance(observation_result, dict):
                # Structured format (new format) - format nicely with JSON
                obs_result_str = json.dumps(observation_result, indent=2)
            elif isinstance(observation_result, str):
                # Legacy string format - pass through as-is
                obs_result_str = observation_result
            else:
                obs_result_str = str(observation_result)
            
            observation_str = f"""
OBSERVATION RECEIVED (What the agent actually observed after taking this action)

Observation Result:
{obs_result_str}
"""
            
            # Include metadata - this contains additional structured information
            # Note: For structured observation.result, metadata may have some overlap,
            # but it's useful for backward compatibility and additional context
            if observation_metadata:
                # Format metadata nicely - use JSON for dicts, string for strings
                if isinstance(observation_metadata, dict):
                    obs_meta_str = json.dumps(observation_metadata, indent=2)
                else:
                    obs_meta_str = str(observation_metadata)
                observation_str += f"""
Observation Metadata:
{obs_meta_str}
"""
        
        context_str = f"""CONTEXT
- Task: {task_name}
- Environment: {environment_info}
- Step: {step}
{beliefs_str}
{observation_str}
"""
        
        # Use canonical definitions
        from polaris_agents.prompts.utils import get_canonical_theta_definition, get_canonical_design_definition
        
        theta_def = get_canonical_theta_definition()
        design_def = get_canonical_design_definition()
        
        definitions = f"""INFORMATION SPACE
For a given task instance, we split the information space into two components:
- **Task parameters (θ)**: {theta_def}
- **Design parameters (D)**: {design_def}

Your task is to estimate the TRUE information gain from this action, evaluated post-hoc after seeing the observation."""
        
        prompt = f"""You are estimating the TRUE information gain from an agent action, evaluated post-hoc after the observation is received.

{context_str}

ACTION TAKEN
- Type: {action_type}
- Content: {action_content}

{definitions}

INFORMATION GAIN DEFINITION
**True Information Gain (IG)**: The actual amount of information gained about task parameters (θ) or design parameters (D) after observing the result of this action. This is evaluated post-hoc, meaning you can see both what action was taken AND what was observed.

**Task IG**: How much did this action actually improve understanding of task-specific information (θ) needed for this instance? Consider what new facts, evidence, or insights about this specific case were revealed.

**Design IG**: How much did this action actually improve understanding of general strategies and methods (D)? Consider what new generalizable knowledge about effective approaches, patterns, or principles was revealed.

**Joint IG**: Overall information value combining both task-level and design-level insights (can range from 0.0 to 2.0).

## Task

Rate the TRUE information gain of this action on three dimensions:

1. **Task IG** (0.0-1.0): How much did this action actually improve understanding of task-specific information (θ) for this instance?
2. **Design IG** (0.0-1.0): How much did this action actually improve understanding of general strategies (D)?
3. **Joint IG** (0.0-2.0): Overall information value combining both types of insights

## Guidelines

- Evaluate based on what was ACTUALLY learned from the observation, not what was expected
- Consider the agent's beliefs before the action: did the observation significantly update them?
- For actions that returned no useful information, scores should be low
- For actions that revealed critical new facts or strategies, scores should be high
- Use the full scale when appropriate; justify scores succinctly
- Base scores on actual posterior changes given the observation received

## Response Format

Provide your information gain analysis as a JSON object with the following structure:

RESPONSE FORMAT (JSON only)
{{
    "task_eig": <float between 0.0 and 1.0>,
    "design_eig": <float between 0.0 and 1.0>,
    "joint_eig": <float between 0.0 and 2.0>,
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the scores based on what was actually observed>"
}}"""
        
        # Add domain-specific guidance if available
        if self.domain_knowledge:
            theta_desc = self.domain_knowledge.get_theta_description()
            design_desc = self.domain_knowledge.get_design_description()
            domain_guidance = f"""
{theta_desc}

{design_desc}
"""
            prompt = prompt + domain_guidance
        
        return prompt
    
    def _parse_estimation_response(
        self, 
        response: str, 
        step: int, 
        action_type: str, 
        action_content: str
    ) -> EIGEstimationResult:
        """Parse the LLM response into an EIGEstimationResult."""
        
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
            task_eig = float(data.get('task_eig', 0.0))
            task_eig = max(0.0, min(1.0, task_eig))  # Clamp to [0,1]
            
            design_eig = float(data.get('design_eig', 0.0))
            design_eig = max(0.0, min(1.0, design_eig))  # Clamp to [0,1]
            
            joint_eig = float(data.get('joint_eig', 0.0))
            joint_eig = max(0.0, min(2.0, joint_eig))  # Clamp to [0,2]
            
            reasoning = str(data.get('reasoning', 'No reasoning provided'))
            confidence = float(data.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            return EIGEstimationResult(
                step=step,
                timestamp=datetime.now().isoformat(),
                action_type=action_type,
                action_content=action_content,
                task_eig=task_eig,
                design_eig=design_eig,
                joint_eig=joint_eig,
                reasoning=reasoning,
                confidence=confidence
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse estimation response at step {step}: {e}")
            # Fallback parsing - try to extract numbers from text
            try:
                import re
                task_match = re.search(r'task_eig["\']?\s*:\s*([0-9.]+)', response)
                design_match = re.search(r'design_eig["\']?\s*:\s*([0-9.]+)', response)
                joint_match = re.search(r'joint_eig["\']?\s*:\s*([0-9.]+)', response)
                confidence_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', response)
                
                task_eig = float(task_match.group(1)) if task_match else 0.0
                design_eig = float(design_match.group(1)) if design_match else 0.0
                joint_eig = float(joint_match.group(1)) if joint_match else 0.0
                confidence = float(confidence_match.group(1)) if confidence_match else 0.3
                
                return EIGEstimationResult(
                    step=step,
                    timestamp=datetime.now().isoformat(),
                    action_type=action_type,
                    action_content=action_content,
                    task_eig=max(0.0, min(1.0, task_eig)),
                    design_eig=max(0.0, min(1.0, design_eig)),
                    joint_eig=max(0.0, min(2.0, joint_eig)),
                    reasoning=f"Parsed from text (JSON parse failed): {response[:200]}",
                    confidence=max(0.0, min(1.0, confidence))
                )
            except Exception:
                # Ultimate fallback
                return EIGEstimationResult(
                    step=step,
                    timestamp=datetime.now().isoformat(),
                    action_type=action_type,
                    action_content=action_content,
                    task_eig=0.0,
                    design_eig=0.0,
                    joint_eig=0.0,
                    reasoning=f"Failed to parse response: {response[:200]}",
                    confidence=0.0
                )
    
    def get_eig_summary(self) -> Dict[str, Any]:
        """Get summary statistics of EIG estimations."""
        if not self.estimations:
            return {"total_estimations": 0}
        
        task_eigs = [e.task_eig for e in self.estimations if e.task_eig is not None]
        design_eigs = [e.design_eig for e in self.estimations if e.design_eig is not None]
        joint_eigs = [e.joint_eig for e in self.estimations if e.joint_eig is not None]
        confidences = [e.confidence for e in self.estimations if e.confidence is not None]
        
        summary = {
            "total_estimations": len(self.estimations)
        }
        
        if task_eigs:
            summary.update({
                "average_task_eig": sum(task_eigs) / len(task_eigs),
                "max_task_eig": max(task_eigs),
                "min_task_eig": min(task_eigs),
                "task_eig_trend": task_eigs
            })
        
        if design_eigs:
            summary.update({
                "average_design_eig": sum(design_eigs) / len(design_eigs),
                "max_design_eig": max(design_eigs),
                "min_design_eig": min(design_eigs),
                "design_eig_trend": design_eigs
            })
        
        if joint_eigs:
            summary.update({
                "average_joint_eig": sum(joint_eigs) / len(joint_eigs),
                "max_joint_eig": max(joint_eigs),
                "min_joint_eig": min(joint_eigs),
                "joint_eig_trend": joint_eigs
            })
        
        if confidences:
            summary["average_confidence"] = sum(confidences) / len(confidences)
            summary["confidence_trend"] = confidences
        
        return summary
