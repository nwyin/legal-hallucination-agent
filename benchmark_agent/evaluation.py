"""Scoring for the hallucination checker: precision, recall, and F1.

A ground truth item is a "hit" if it is a substring of a predicted item (or a
predicted item is a substring of it). Scoring is kept separate from recording.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --- Hallucination checker scoring ---


def _extract_list_from_string(s: str) -> Optional[List[str]]:
    """Return the first JSON list embedded in `s`, or None if there is none.

    The scanner is quote- and escape-aware so brackets inside string items do
    not end the list early. A list that is present but invalid JSON also yields
    None; a valid empty list yields [].
    """
    s = s.strip()
    start = s.find("[")
    if start == -1:
        return None
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
                    return [str(x).strip() for x in parsed if x is not None] if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    return None
        i += 1
    return None


def parse_predictions(raw: Any) -> List[str]:
    """
    Parse a final response (or a stored predicted_hallucinations value) into a
    list of individual predictions. This is the single parser used both when the
    agent records its final answer and when that answer is scored.

    Handles:
    - Already a list of items: return as-is (but parse any element that is a string
      containing a JSON list)
    - String containing a JSON list (possibly with surrounding text): extract the list
    - Any other non-empty string: the whole stripped string is one prediction
    - None or empty: return []
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
    if not isinstance(raw, str):
        return []
    stripped = raw.strip()
    if not stripped:
        return []
    extracted = _extract_list_from_string(stripped)
    return extracted if extracted is not None else [stripped]


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
    """Resolve the ground-truth field from known dataset variants (including a legacy typo)."""
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
