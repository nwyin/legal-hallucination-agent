"""
Belief Evolution Tracker for monitoring how agent beliefs change over time.

This module provides functionality to track:
1. Bandit-specific: Belief accuracy vs ground truth (for tasks with known true parameters)
2. General: Belief complexity and sophistication (using LLM-as-a-judge)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import re

from polaris_agents.models.llm import ModelAPI

logger = logging.getLogger(__name__)

@dataclass
class BeliefEvolutionResult:
    """Result of belief evolution analysis for a single step."""
    step: int
    
    # Bandit-specific metrics (if applicable)
    task_belief_accuracy: Optional[float] = None  # Baseline-relative accuracy (0-1, higher is better)
    task_belief_mean_kl_bits: Optional[float] = None  # Raw mean KL divergence in bits
    task_belief_baseline_kl_bits: Optional[float] = None  # Baseline KL divergence in bits (uniform q=0.5, weighted by pull counts)
    task_belief_uncertainty: Optional[float] = None  # Entropy or uncertainty measure
    inferred_arm_probs: Optional[List[float]] = None  # Extracted arm probabilities
    
    # General complexity metrics
    task_belief_complexity: Optional[float] = None  # 0.0-1.0 complexity score
    design_belief_complexity: Optional[float] = None  # 0.0-1.0 complexity score
    belief_coherence: Optional[float] = None  # 0.0-1.0 coherence score
    information_density: Optional[float] = None  # 0.0-1.0 information density
    
    # LLM evaluation metadata
    complexity_confidence: Optional[float] = None
    complexity_reasoning: Optional[str] = None

class BeliefEvolutionTracker:
    """Tracks evolution of agent beliefs over time."""
    
    def __init__(self, model_api: ModelAPI, model_name: str = "openai/gpt-4o-mini", provider: str = "openrouter", max_tokens: int = 10000):
        """
        Initialize the belief evolution tracker.
        
        Args:
            model_api: ModelAPI instance for LLM-based evaluation
            model_name: Name of the model to use for evaluation
            provider: Model provider to use
            max_tokens: Maximum tokens for LLM responses (default: 10000)
        """
        self.model_api = model_api
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.belief_history: List[BeliefEvolutionResult] = []
        self.ground_truth = None
        self.task_type = None
        
    def set_ground_truth(self, ground_truth: Any, task_type: str) -> None:
        """
        Set ground truth for bandit-specific belief accuracy tracking.
        
        Args:
            ground_truth: Ground truth values (e.g., true arm probabilities for bandits)
            task_type: Type of task (e.g., "multi_armed_bandit")
        """
        self.ground_truth = ground_truth
        self.task_type = task_type
        logger.info(f"Set ground truth for belief tracking: {ground_truth} (task: {task_type})")
    
    def track_belief_evolution(self, step: int, task_beliefs: str, design_beliefs: str, agent: Any = None) -> BeliefEvolutionResult:
        """
        Track belief evolution for a single step.
        
        Args:
            step: Current step number
            task_beliefs: Agent's task-level beliefs (natural language)
            design_beliefs: Agent's design-level beliefs (natural language)
            
        Returns:
            BeliefEvolutionResult with analysis of beliefs at this step
        """
        result = BeliefEvolutionResult(step=step)
        
        # For bandit tasks, extract arm probabilities and compare to ground truth
        if self.task_type == "multi_armed_bandit" and self.ground_truth is not None and agent is not None:
            result = self._analyze_bandit_beliefs(result, agent)
        
        # For all tasks, analyze belief complexity using LLM-as-a-judge
        result = self._analyze_belief_complexity(result, task_beliefs, design_beliefs)
        
        self.belief_history.append(result)
        return result
    
    def _analyze_bandit_beliefs(self, result: "BeliefEvolutionResult", agent: Any) -> "BeliefEvolutionResult":
        """
        Evaluate beliefs vs ground truth for Bernoulli bandit arms.
        - Accuracy: 1 - min(mean KL (bits), 1.0)  (linear, interpretable)
        - Uncertainty: mean Bernoulli entropy (bits) in [0,1]
        - Aggregation weighted by arm pull counts.
        """
        try:
            # agent always returns (probs, pull_counts)
            inferred_probs, inferred_pull_counts = agent.get_current_arm_probabilities()

            if inferred_probs is None or len(inferred_probs) != len(self.ground_truth):
                logger.warning("Could not extract valid arm probabilities from agent")
                return result

            if inferred_pull_counts is None or len(inferred_pull_counts) != len(self.ground_truth):
                logger.warning("Could not extract valid arm pull counts from agent")
                return result

            # store probs
            result.inferred_arm_probs = inferred_probs

            # 1) Raw mean KL in bits
            mean_kl_bits = self._mean_bernoulli_kl_bits(
                self.ground_truth, inferred_probs, weights=inferred_pull_counts
            )
            result.task_belief_mean_kl_bits = mean_kl_bits

            # 2) Baseline KL in bits (uniform q=0.5 per arm, weighted by pull counts)
            kl_base_bits = self._mean_bernoulli_kl_bits(
                self.ground_truth,
                [0.5] * len(self.ground_truth),
                weights=inferred_pull_counts
            )
            result.task_belief_baseline_kl_bits = kl_base_bits

            # 3) Baseline-relative accuracy
            rel_accuracy = 1.0 - mean_kl_bits / max(kl_base_bits, 1e-12)
            result.task_belief_accuracy = rel_accuracy

            # mean entropy in bits, weighted by pull counts
            mean_H_bits = self._mean_entropy_bits(inferred_probs, weights=inferred_pull_counts)
            result.task_belief_uncertainty = mean_H_bits

            logger.info(
                f"Bandit belief analysis: mean_KL_bits={mean_kl_bits:.4f}, "
                f"baseline_KL_bits={kl_base_bits:.4f}, "
                f"relative_accuracy={result.task_belief_accuracy:.4f}, "
                f"mean_entropy_bits={mean_H_bits:.4f}, "
                f"weights={inferred_pull_counts}"
            )

        except Exception as e:
            logger.warning(f"Bandit belief analysis failed: {e}")

        return result
    
    
    def _bernoulli_kl_bits(self, p: float, q: float) -> float:
        """
        Compute KL divergence between two Bernoulli distributions in bits.
        
        Args:
            p: True probability
            q: Inferred probability
            
        Returns:
            KL divergence in bits
        """
        # Clamp probabilities to valid range
        p = max(1e-10, min(1 - 1e-10, p))
        q = max(1e-10, min(1 - 1e-10, q))
        
        # KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
        kl_bits = p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))
        return max(0, kl_bits)
    
    def _bernoulli_entropy_bits(self, p: float) -> float:
        """
        Compute entropy of a Bernoulli distribution in bits.
        
        Args:
            p: Probability
            
        Returns:
            Entropy in bits (max 1 bit)
        """
        # Clamp probability to valid range
        p = max(1e-10, min(1 - 1e-10, p))
        
        # H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy_bits = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return entropy_bits
    
    def _mean_bernoulli_kl_bits(self, true_probs: List[float], inferred_probs: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Compute mean KL divergence across Bernoulli arms, optionally weighted.
        
        Args:
            true_probs: True arm probabilities
            inferred_probs: Inferred arm probabilities
            weights: Optional weights (e.g., pull counts)
            
        Returns:
            Mean KL divergence in bits
        """
        if weights is None:
            weights = [1.0] * len(true_probs)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_kl = sum(
            weight * self._bernoulli_kl_bits(true_p, inferred_p)
            for true_p, inferred_p, weight in zip(true_probs, inferred_probs, weights)
        )
        
        return weighted_kl / total_weight
    
    def _mean_entropy_bits(self, probs: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Compute mean entropy across Bernoulli arms, optionally weighted.
        
        Args:
            probs: Arm probabilities
            weights: Optional weights (e.g., pull counts)
            
        Returns:
            Mean entropy in bits
        """
        if weights is None:
            weights = [1.0] * len(probs)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_entropy = sum(
            weight * self._bernoulli_entropy_bits(p)
            for p, weight in zip(probs, weights)
        )
        
        return weighted_entropy / total_weight
    
    def _analyze_belief_complexity(self, result: BeliefEvolutionResult, task_beliefs: str, design_beliefs: str) -> BeliefEvolutionResult:
        """
        Analyze belief complexity using LLM-as-a-judge.
        
        Args:
            result: BeliefEvolutionResult to update
            task_beliefs: Agent's task-level beliefs
            design_beliefs: Agent's design-level beliefs
            
        Returns:
            Updated BeliefEvolutionResult
        """
        try:
            prompt = self._create_complexity_prompt(task_beliefs, design_beliefs)
            response = self.model_api(
                model_id=self.model_name,
                prompt=[{"role": "user", "content": prompt}],
                max_attempts=3,
                provider=self.provider,
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            if response:
                complexity_data = self._parse_complexity_response(response)
                
                if complexity_data:
                    result.task_belief_complexity = complexity_data.get("task_complexity")
                    result.design_belief_complexity = complexity_data.get("design_complexity")
                    result.belief_coherence = complexity_data.get("coherence")
                    result.information_density = complexity_data.get("information_density")
                    result.complexity_confidence = complexity_data.get("confidence")
                    result.complexity_reasoning = complexity_data.get("reasoning")
                    
                    logger.info(f"Belief complexity analysis: task={result.task_belief_complexity:.3f}, "
                                 f"design={result.design_belief_complexity:.3f}, coherence={result.belief_coherence:.3f}")
                else:
                    logger.warning(f"Failed to parse complexity analysis response. Raw response: {response[:200]}...")
                    
        except Exception as e:
            logger.warning(f"Belief complexity analysis failed: {e}")
            
        return result
    
    def _create_complexity_prompt(self, task_beliefs: str, design_beliefs: str) -> str:
        """
        Create prompt for LLM-based belief complexity analysis.
        
        Args:
            task_beliefs: Agent's task-level beliefs
            design_beliefs: Agent's design-level beliefs
            
        Returns:
            Formatted prompt for complexity analysis
        """
        return f"""You are an expert in analyzing cognitive complexity and belief evolution in AI agents. You will evaluate the complexity and sophistication of agent beliefs.

## Agent's Current Beliefs

### Task Beliefs (θ):
{task_beliefs}

### Design Beliefs (D):
{design_beliefs}

## Task

Rate the complexity and sophistication of these beliefs on four dimensions:

1. **Task Belief Complexity** (0.0-1.0): How nuanced, detailed, and sophisticated are the task-specific beliefs?
   - 0.0-0.2: Very simple, basic observations
   - 0.3-0.5: Some detail, basic patterns recognized
   - 0.6-0.8: Detailed understanding, multiple factors considered
   - 0.9-1.0: Highly sophisticated, complex relationships understood

2. **Design Belief Complexity** (0.0-1.0): How sophisticated are the meta-level strategic beliefs?
   - 0.0-0.2: Basic strategies, simple heuristics
   - 0.3-0.5: Some strategic thinking, multiple approaches considered
   - 0.6-0.8: Sophisticated strategy selection, adaptive approaches
   - 0.9-1.0: Highly meta-cognitive, complex strategic reasoning

3. **Belief Coherence** (0.0-1.0): How consistent and well-integrated are the beliefs?
   - 0.0-0.2: Contradictory or fragmented beliefs
   - 0.3-0.5: Some inconsistencies, basic integration
   - 0.6-0.8: Mostly coherent, well-integrated understanding
   - 0.9-1.0: Highly coherent, seamless integration

4. **Information Density** (0.0-1.0): How information-rich and detailed are the beliefs?
   - 0.0-0.2: Very sparse, minimal information
   - 0.3-0.5: Moderate detail, some information
   - 0.6-0.8: Rich detail, substantial information
   - 0.9-1.0: Extremely information-dense, comprehensive

## Response Format

Provide your analysis as a JSON object with the following structure:

RESPONSE FORMAT (JSON only)
{{
  "task_complexity": <float between 0.0 and 1.0>,
  "design_complexity": <float between 0.0 and 1.0>,
  "coherence": <float between 0.0 and 1.0>,
  "information_density": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of the scores>"
}}"""
    
    def _parse_complexity_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response for belief complexity analysis.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed complexity data or None if parsing fails
        """
        try:
            # Clean up response
            response_clean = response.strip()
            # logger.info(f"DEBUG: Parsing response (first 200 chars): {response_clean[:200]}")
            
            # Extract JSON from response - use simple approach like action classifier
            # Try to extract JSON from response
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            # Handle truncated JSON responses by attempting to fix common issues
            response_clean = response_clean.strip()
            
            # Check if the response looks truncated (missing closing braces)
            open_braces = response_clean.count('{')
            close_braces = response_clean.count('}')
            if open_braces > close_braces:
                # Try to fix by adding missing closing braces
                missing_braces = open_braces - close_braces
                response_clean += '}' * missing_braces
                logger.info(f"Attempted to fix truncated JSON by adding {missing_braces} closing braces")
            
            # Check for unterminated strings (common issue with truncated responses)
            if '"reasoning":' in response_clean:
                # Find the start of the reasoning field and ensure it's properly terminated
                reasoning_start = response_clean.find('"reasoning":')
                if reasoning_start != -1:
                    # Find the opening quote after the colon
                    quote_start = response_clean.find('"', reasoning_start + 12)
                    if quote_start != -1:
                        # Check if the string is properly terminated
                        remaining = response_clean[quote_start + 1:]
                        if not remaining.endswith('"') and '"' not in remaining:
                            # Likely truncated - try to close the string
                            response_clean += '"'
                            logger.info("Attempted to fix unterminated reasoning string")
            
            # logger.info(f"DEBUG: Cleaned response for JSON parsing: {response_clean[:200]}...")
            
            import json
            try:
                data = json.loads(response_clean.strip())
                
                # Validate required fields
                required_fields = ["task_complexity", "design_complexity", "coherence", "information_density"]
                if all(field in data for field in required_fields):
                    # Ensure all values are floats between 0 and 1
                    for field in required_fields:
                        value = data[field]
                        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                            logger.warning(f"Invalid {field} value: {value}")
                            return None
                    
                    return data
                else:
                    logger.warning(f"Missing required fields. Found: {list(data.keys())}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                logger.warning(f"Raw response: {response_clean[:500]}...")
            except Exception as e:
                logger.warning(f"Unexpected parsing error: {e}")
                logger.warning(f"Raw response: {response_clean[:500]}...")
                    
        except Exception as e:
            logger.info(f"Complexity response parsing failed: {e}")
            
        return None
    
    def get_belief_evolution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for belief evolution over time.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.belief_history:
            return {}
        
        summary = {
            "total_steps": len(self.belief_history),
            "task_type": self.task_type
        }
        
        # Bandit-specific metrics
        if self.task_type == "multi_armed_bandit":
            accuracies = [r.task_belief_accuracy for r in self.belief_history if r.task_belief_accuracy is not None]
            mean_kl_bits = [r.task_belief_mean_kl_bits for r in self.belief_history if r.task_belief_mean_kl_bits is not None]
            baseline_kl_bits = [r.task_belief_baseline_kl_bits for r in self.belief_history if r.task_belief_baseline_kl_bits is not None]
            uncertainties = [r.task_belief_uncertainty for r in self.belief_history if r.task_belief_uncertainty is not None]
            
            if accuracies:
                summary.update({
                    "average_task_belief_accuracy": np.mean(accuracies),
                    "final_task_belief_accuracy": accuracies[-1] if accuracies else None,
                    "task_belief_accuracy_trend": accuracies
                })
            
            if mean_kl_bits:
                summary.update({
                    "average_task_belief_mean_kl_bits": np.mean(mean_kl_bits),
                    "final_task_belief_mean_kl_bits": mean_kl_bits[-1] if mean_kl_bits else None,
                    "task_belief_mean_kl_bits_trend": mean_kl_bits
                })
            
            if baseline_kl_bits:
                summary.update({
                    "average_task_belief_baseline_kl_bits": np.mean(baseline_kl_bits),
                    "final_task_belief_baseline_kl_bits": baseline_kl_bits[-1] if baseline_kl_bits else None,
                    "task_belief_baseline_kl_bits_trend": baseline_kl_bits
                })
            
            if uncertainties:
                summary.update({
                    "average_task_belief_uncertainty": np.mean(uncertainties),
                    "final_task_belief_uncertainty": uncertainties[-1] if uncertainties else None,
                    "task_belief_uncertainty_trend": uncertainties
                })
        
        # General complexity metrics
        task_complexities = [r.task_belief_complexity for r in self.belief_history if r.task_belief_complexity is not None]
        design_complexities = [r.design_belief_complexity for r in self.belief_history if r.design_belief_complexity is not None]
        coherences = [r.belief_coherence for r in self.belief_history if r.belief_coherence is not None]
        densities = [r.information_density for r in self.belief_history if r.information_density is not None]
        
        if task_complexities:
            summary.update({
                "average_task_belief_complexity": np.mean(task_complexities),
                "final_task_belief_complexity": task_complexities[-1],
                "task_belief_complexity_trend": task_complexities
            })
        
        if design_complexities:
            summary.update({
                "average_design_belief_complexity": np.mean(design_complexities),
                "final_design_belief_complexity": design_complexities[-1],
                "design_belief_complexity_trend": design_complexities
            })
        
        if coherences:
            summary.update({
                "average_belief_coherence": np.mean(coherences),
                "final_belief_coherence": coherences[-1],
                "belief_coherence_trend": coherences
            })
        
        if densities:
            summary.update({
                "average_information_density": np.mean(densities),
                "final_information_density": densities[-1],
                "information_density_trend": densities
            })
        
        return summary
    
    def save_belief_evolution(self, filepath: str) -> None:
        """
        Save belief evolution data to a JSON file.
        
        Args:
            filepath: Path to save the data
        """
        data = {
            "belief_history": [
                {
                    "step": r.step,
                    "task_belief_accuracy": r.task_belief_accuracy,
                    "task_belief_mean_kl_bits": r.task_belief_mean_kl_bits,
                    "task_belief_baseline_kl_bits": r.task_belief_baseline_kl_bits,
                    "task_belief_uncertainty": r.task_belief_uncertainty,
                    "inferred_arm_probs": r.inferred_arm_probs,
                    "task_belief_complexity": r.task_belief_complexity,
                    "design_belief_complexity": r.design_belief_complexity,
                    "belief_coherence": r.belief_coherence,
                    "information_density": r.information_density,
                    "complexity_confidence": r.complexity_confidence,
                    "complexity_reasoning": r.complexity_reasoning
                }
                for r in self.belief_history
            ],
            "summary": self.get_belief_evolution_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Belief evolution data saved to: {filepath}")
    
    @classmethod
    def load_belief_evolution(cls, filepath: str) -> 'BeliefEvolutionTracker':
        """
        Load belief evolution data from a JSON file.
        
        Args:
            filepath: Path to the saved data
            
        Returns:
            BeliefEvolutionTracker instance with loaded data
        """
        tracker = cls(None)  # ModelAPI will be set later if needed
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct belief history
        for item in data.get("belief_history", []):
            result = BeliefEvolutionResult(
                step=item["step"],
                task_belief_accuracy=item.get("task_belief_accuracy"),
                task_belief_mean_kl_bits=item.get("task_belief_mean_kl_bits"),
                task_belief_baseline_kl_bits=item.get("task_belief_baseline_kl_bits"),
                task_belief_uncertainty=item.get("task_belief_uncertainty"),
                inferred_arm_probs=item.get("inferred_arm_probs"),
                task_belief_complexity=item.get("task_belief_complexity"),
                design_belief_complexity=item.get("design_belief_complexity"),
                belief_coherence=item.get("belief_coherence"),
                information_density=item.get("information_density"),
                complexity_confidence=item.get("complexity_confidence"),
                complexity_reasoning=item.get("complexity_reasoning")
            )
            tracker.belief_history.append(result)
        
        logger.info(f"Belief evolution data loaded from: {filepath}")
        return tracker
