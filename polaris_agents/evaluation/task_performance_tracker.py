"""
Task performance tracking for evaluating agent prediction quality over time.

This module provides agent-centric prediction tracking, using the agent's own model
to get predictions and confidence scores at each step. This is separate from external 
LLM-as-a-judge metrics - this captures what the agent thinks about the final answer.
"""

import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result of an agent prediction evaluation."""
    step: int
    timestamp: str
    prediction: str
    confidence: float
    is_correct: Optional[bool] = None
    accuracy: Optional[float] = None

class TaskPerformanceTracker:
    """
    Tracks task performance over time by evaluating agent predictions using the agent's own model.
    This captures the agent's internal predictions and confidence, not external judge assessments.
    This runs in parallel to agent execution without interfering with it.
    
    Uses the environment's is_correct() method for evaluation to ensure consistency.
    """
    
    def __init__(self):
        self.predictions: List[PredictionResult] = []
        self.ground_truth = None
        self.task_type = None
        self.environment = None  # Store reference to environment for is_correct() calls
        
    def set_ground_truth(self, ground_truth: Any, task_type: str, environment: Any = None):
        """
        Set the ground truth for evaluation.
        
        Args:
            ground_truth: The correct answer (e.g., "A" for bandits, ["No"] for SCOTUS)
            task_type: Type of task (e.g., "multi_armed_bandit", "scotus_judgment")
            environment: Optional environment instance to use for is_correct() evaluation
        """
        self.ground_truth = ground_truth
        self.task_type = task_type
        self.environment = environment
        logger.info(f"Set ground truth for {task_type}: {ground_truth}")
        if environment:
            logger.info(f"Using environment's is_correct() method for evaluation")
    
    def evaluate_prediction_at_step(
        self, 
        step: int, 
        agent: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Get the agent's prediction and confidence using the agent's own model.
        
        Args:
            step: Current step number
            agent: The agent instance to get predictions from
            context: Additional context about the task/environment
            
        Returns:
            PredictionResult with agent's prediction, confidence, and accuracy evaluation
        """
        try:
            # Get prediction directly from the agent's own model
            if hasattr(agent, 'get_current_prediction'):
                prediction, confidence = agent.get_current_prediction()
            elif hasattr(agent, 'get_best_prediction'):
                prediction, confidence = agent.get_best_prediction()
            else:
                logger.warning(f"Agent {type(agent).__name__} doesn't have get_current_prediction or get_best_prediction method")
                return PredictionResult(
                    step=step,
                    timestamp=datetime.now().isoformat(),
                    prediction="NO_METHOD",
                    confidence=0.0,
                    is_correct=None,
                    accuracy=None
                )
            
            if prediction is None or confidence is None:
                logger.warning(f"Prediction returned None/None from agent")
                return PredictionResult(
                    step=step,
                    timestamp=datetime.now().isoformat(),
                    prediction="NO_PREDICTION",
                    confidence=0.0,
                    is_correct=None,
                    accuracy=None
                )
            
            # Evaluate accuracy if ground truth is available
            is_correct = None
            accuracy = None
            if self.ground_truth is not None:
                accuracy = self._evaluate_correctness(prediction, self.ground_truth)
                is_correct = accuracy == 1.0  # Perfect score means completely correct
            
            result = PredictionResult(
                step=step,
                timestamp=datetime.now().isoformat(),
                prediction=prediction,
                confidence=confidence,
                is_correct=is_correct,
                accuracy=accuracy
            )
            
            self.predictions.append(result)
            # Log both accuracy (ratio) and is_correct (fully correct boolean)
            accuracy_str = f"{accuracy:.3f}" if accuracy is not None else "N/A"
            logger.info(f"Step {step}: Prediction='{prediction}', Confidence={confidence:.3f}, "
                        f"Accuracy={accuracy_str}, FullyCorrect={is_correct}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate prediction at step {step}: {e}")
            return PredictionResult(
                step=step,
                timestamp=datetime.now().isoformat(),
                prediction="ERROR",
                confidence=0.0,
                is_correct=None,
                accuracy=None
            )

    def evaluate_final_prediction(
        self,
        agent: Any,
        environment: Any = None,
        step: int = 0,
    ) -> Optional[PredictionResult]:
        """
        Evaluate the agent's final submitted prediction (once per episode).
        Uses the already-submitted answer from the agent/environment; does not call the LLM.

        Args:
            agent: The agent instance (for last_final_response_list etc.)
            environment: Optional environment (for environment.answer)
            step: Step number to record (e.g. final step)

        Returns:
            PredictionResult, or None if no final prediction could be obtained.
        """
        prediction = None
        confidence = 1.0
        # Prefer the submitted answer: list tasks (e.g. hallucination checker) store list on agent
        if hasattr(agent, "last_final_response_list") and agent.last_final_response_list is not None:
            prediction = agent.last_final_response_list
        if prediction is None and environment is not None and getattr(environment, "answer", None):
            prediction = environment.answer
        if prediction is None:
            logger.warning("No final prediction (last_final_response_list or environment.answer) available")
            return None
        try:
            accuracy = None
            if self.ground_truth is not None:
                accuracy = self._evaluate_correctness(prediction, self.ground_truth)
            is_correct = (accuracy == 1.0) if accuracy is not None else None
            result = PredictionResult(
                step=step,
                timestamp=datetime.now().isoformat(),
                prediction=prediction if isinstance(prediction, str) else str(prediction),
                confidence=confidence,
                is_correct=is_correct,
                accuracy=accuracy,
            )
            self.predictions.append(result)
            logger.info(
                f"Final prediction: step={step}, accuracy={accuracy}, is_correct={is_correct}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate final prediction: {e}")
            return None

    def _evaluate_correctness(self, prediction: Any, ground_truth: Any) -> float:
        """Evaluate accuracy of the prediction.

        For legal_hallucination_checker: uses precision/recall/F1 (returns F1 as accuracy).
        For other tasks: uses the environment's is_correct() method.

        Returns:
            float: Accuracy (0.0 to 1.0). For legal_hallucination_checker, this is F1.
        """
        # Legal hallucination checker: use precision/recall/F1 evaluation
        if self.task_type == "legal_hallucination_checker" and isinstance(ground_truth, list):
            from .hallucination_checker_evaluator import evaluate_entry, compute_metrics
            gt_found, gt_total, correct_pred, pred_total = evaluate_entry(ground_truth, prediction)
            metrics = compute_metrics(gt_found, gt_total, correct_pred, pred_total)
            return float(metrics["f1"])

        # Default: use environment's is_correct()
        if self.environment is None:
            raise ValueError("Environment not set. Call set_ground_truth() with environment parameter.")
        if not hasattr(self.environment, 'is_correct'):
            raise ValueError(f"Environment {type(self.environment).__name__} must implement is_correct() method")

        original_answer = getattr(self.environment, 'answer', None)
        try:
            self.environment.answer = prediction
            accuracy = self.environment.is_correct()
            if isinstance(accuracy, bool):
                accuracy = 1.0 if accuracy else 0.0
            return float(accuracy)
        finally:
            self.environment.answer = original_answer
    
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of task performance over time."""
        if not self.predictions:
            return {"error": "No predictions recorded"}
        
        correct_predictions = [p for p in self.predictions if p.is_correct]
        total_predictions = len(self.predictions)
        
        if total_predictions == 0:
            return {"error": "No valid predictions"}
        
        # Filter out None values to avoid arithmetic with None
        accuracy_over_time = [p.accuracy for p in self.predictions if p.accuracy is not None]
        confidence_over_time = [
            p.confidence for p in self.predictions
            if isinstance(p.confidence, (int, float))
        ]
        final_accuracy = accuracy_over_time[-1] if accuracy_over_time else None
        
        return {
            "total_predictions": total_predictions,
            "correct_predictions": len(correct_predictions),
            "overall_accuracy": len(correct_predictions) / total_predictions if total_predictions > 0 else 0,
            "final_accuracy": final_accuracy,
            "average_confidence": (
                sum(confidence_over_time) / len(confidence_over_time)
                if confidence_over_time else 0.0
            ),
            "accuracy_trend": accuracy_over_time,
            "confidence_trend": confidence_over_time
        }
    
    def save_predictions(self, filepath: str) -> None:
        """Save predictions to a JSON file."""
        data = {
            "task_type": self.task_type,
            "ground_truth": self.ground_truth,
            "predictions": [asdict(p) for p in self.predictions],
            "performance_summary": self.get_performance_summary()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.predictions)} predictions to {filepath}")
    
    @classmethod
    def load_predictions(cls, filepath: str) -> 'TaskPerformanceTracker':
        """Load predictions from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create tracker instance
        tracker = cls()
        tracker.task_type = data.get('task_type')
        tracker.ground_truth = data.get('ground_truth')
        
        # Reconstruct predictions
        tracker.predictions = []
        for pred_data in data.get('predictions', []):
            prediction = PredictionResult(**pred_data)
            tracker.predictions.append(prediction)
        
        return tracker
