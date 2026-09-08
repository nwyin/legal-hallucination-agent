"""Scoring: hallucination-entry evaluation and final-prediction tracking.

A ground truth item is a "hit" if it is a substring of a predicted item (or a
predicted item is a substring of it). Scoring is kept separate from recording.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --- Hallucination checker scoring ---


def _extract_list_from_string(s: str) -> List[str]:
    s = s.strip()
    start = s.find("[")
    if start == -1:
        return []
    depth = 0
    i = start
    in_string = False
    quote_char = None
    escape = False
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == quote_char:
                in_string = False
            i += 1
            continue
        if c in ('"', "'"):
            in_string = True
            quote_char = c
            i += 1
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(s[start : i + 1])
                    return [str(x).strip() for x in parsed if x is not None] if isinstance(parsed, list) else []
                except json.JSONDecodeError:
                    return []
        i += 1
    return []


def parse_predictions(raw: Any) -> List[str]:
    """
    Parse predicted_hallucinations into a list of individual predictions.

    Handles:
    - Already a list of items: return as-is (but parse any element that is a string
      containing a JSON list)
    - String with JSON list + trailing text: extract list
    - None, empty, or invalid: return []
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        result = []
        for x in raw:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            if s.startswith("["):
                extracted = _extract_list_from_string(s)
                if extracted:
                    result.extend(extracted)
                    continue
            result.append(s)
        return result
    if not isinstance(raw, str) or not raw.strip():
        return []
    return _extract_list_from_string(raw.strip())


def _normalize(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _fragments_in_order(fragments: List[str], text: str) -> bool:
    pos = 0
    for frag in fragments:
        if not frag:
            continue
        idx = text.find(frag, pos)
        if idx == -1:
            return False
        pos = idx + len(frag)
    return True


def _is_match(gt_item: str, pred_item: str) -> bool:
    """Check if ground truth and prediction match.

    Handles:
    - Substring match in either direction.
    - Ellipsis abbreviation: if either string contains '...' or '…', treat each
      segment as a literal fragment and check they all appear in order in the
      other string (e.g. 'foo ... baz' matches 'foo bar baz').
    """
    if not gt_item or not pred_item:
        return False
    g = _normalize(gt_item)
    p = _normalize(pred_item)
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    # Ellipsis wildcard matching
    for ellipsis in ("...", "\u2026"):
        if ellipsis in p:
            parts = [part.strip() for part in p.split(ellipsis)]
            if _fragments_in_order(parts, g):
                return True
        if ellipsis in g:
            parts = [part.strip() for part in g.split(ellipsis)]
            if _fragments_in_order(parts, p):
                return True
    return False


def _normalize_ground_truth(list_hallucinations: Any) -> List[Tuple[str, Optional[str]]]:
    """
    Normalize list_hallucinations to a list of (span, type).
    Handles: list of spans, dict {span: type}.
    """
    if not list_hallucinations:
        return []
    if isinstance(list_hallucinations, dict):
        return [(str(k).strip(), (str(v).strip() if v else None)) for k, v in list_hallucinations.items() if k]
    if isinstance(list_hallucinations, list):
        return [(str(x).strip(), None) for x in list_hallucinations if x is not None and str(x).strip()]
    return []


def extract_ground_truth(entry: Dict[str, Any]) -> Any:
    if not entry:
        return []
    for key in ("list_hallucinations", "listed_hallucinations", "list_hallucinationss"):
        if key in entry and entry.get(key) is not None:
            return entry.get(key)
    return []


def evaluate_entry(
    ground_truth: Any,
    predicted_hallucinations: Any,
    type_filter: Optional[List[str]] = None,
) -> Tuple[int, int, int, int]:
    """
    Evaluate a single entry.

    Args:
        ground_truth: list_hallucinations - list of spans, or dict {span: hallucination_type}
        predicted_hallucinations: model's predicted list (raw, will be parsed)
        type_filter: if set, only include ground truth items whose type is in this list (case-insensitive).
                     Ignored when ground_truth is a list (no type info).
                     When set: predictions that match only excluded-type ground truth are not counted
                     in the precision denominator (so they don't penalize precision).

    Returns:
        (ground_truth_found, total_ground_truth, correct_predictions, total_predictions)
        total_predictions is the count of predictions used for precision (excludes predictions
        that match only excluded-type ground truth when type_filter is set).
    """
    predictions = parse_predictions(predicted_hallucinations)
    gt_pairs = _normalize_ground_truth(ground_truth)
    types_lower = {t.strip().lower() for t in type_filter if t} if type_filter else set()
    if type_filter and types_lower:
        included_pairs = [(span, t) for span, t in gt_pairs if span and (t is None or (t and t.lower() in types_lower))]
        excluded_pairs = [(span, t) for span, t in gt_pairs if span and (t is not None and t and t.lower() not in types_lower)]
    else:
        included_pairs = gt_pairs
        excluded_pairs = []
    gt_spans = list(dict.fromkeys(span for span, _ in included_pairs))
    excluded_gt_spans = list(dict.fromkeys(span for span, _ in excluded_pairs)) if excluded_pairs else []

    ground_truth_found = 0
    for g in gt_spans:
        if any(_is_match(g, p) for p in predictions):
            ground_truth_found += 1

    correct_predictions = 0
    total_predictions_for_precision = 0
    for p in predictions:
        matches_included = any(_is_match(g, p) for g in gt_spans)
        matches_excluded = any(_is_match(g, p) for g in excluded_gt_spans)
        # When excluding types: don't count this prediction in precision if it only matches excluded-type ground truth
        if matches_excluded and not matches_included:
            continue
        total_predictions_for_precision += 1
        if matches_included:
            correct_predictions += 1

    return ground_truth_found, len(gt_spans), correct_predictions, total_predictions_for_precision


def compute_metrics(
    ground_truth_found: int,
    total_ground_truth: int,
    correct_predictions: int,
    total_predictions: int,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 from counts."""
    recall = ground_truth_found / total_ground_truth if total_ground_truth > 0 else 0.0
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ground_truth_found": float(ground_truth_found),
        "total_ground_truth": float(total_ground_truth),
        "correct_predictions": float(correct_predictions),
        "total_predictions": float(total_predictions),
    }


def evaluate_hallucination_entry(
    entry: Dict[str, Any],
    type_filter: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate a single hallucination checker entry (dict with list_hallucinations and predicted_hallucinations).

    Args:
        entry: dict with list_hallucinations (list or dict {span: type}) and predicted_hallucinations
        type_filter: if set, only evaluate ground truth items whose hallucination type is in this list

    Returns:
        Dict with precision, recall, f1, and raw counts.
    """
    ground_truth = extract_ground_truth(entry)
    predicted = entry.get("predicted_hallucinations")
    gt_found, gt_total, correct_pred, pred_total = evaluate_entry(
        ground_truth, predicted, type_filter=type_filter
    )
    return compute_metrics(gt_found, gt_total, correct_pred, pred_total)


def aggregate_metrics(entries_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate precision, recall, F1 across multiple entries (micro-averaged).

    Args:
        entries_metrics: List of metrics dicts from evaluate_hallucination_entry

    Returns:
        Dict with aggregate precision, recall, f1 and total counts
    """
    total_gt_found = sum(int(m["ground_truth_found"]) for m in entries_metrics)
    total_gt = sum(int(m["total_ground_truth"]) for m in entries_metrics)
    total_correct_pred = sum(int(m["correct_predictions"]) for m in entries_metrics)
    total_pred = sum(int(m["total_predictions"]) for m in entries_metrics)
    return compute_metrics(total_gt_found, total_gt, total_correct_pred, total_pred)


# --- Final-prediction tracking ---


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
    Evaluates final submitted predictions without querying the agent's model.

    Uses citation precision/recall/F1 for the legal hallucination checker and the
    environment's is_correct() method for other tasks.
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
