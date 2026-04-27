"""
Context-aware metadata filtering for closed search results.

This module keeps metadata heuristics close to the retrieval stack so that
environments/agents remain agnostic to field-filter mechanics.  It provides
regex-driven intent detection, validation against retrieved results, and
pruning utilities that can be applied after a single search pass.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, MutableMapping, Optional, Protocol, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

AUTO_FILTER_METADATA_KEY = "auto_metadata_filters"


# ---------------------------------------------------------------------------
# Policy + helpers
# ---------------------------------------------------------------------------

class MetadataIntentResolver(Protocol):
    """Interface for task-specific metadata intent detection."""

    def infer_filters(
        self,
        query: str,
        search_type: str,
        get_field_values: Callable[[str, str], Set[str]],
    ) -> Dict[str, str]:
        ...


@dataclass
class MetadataFilterPolicy:
    """
    Configuration for query-aware metadata pruning.

    Attributes:
        resolver: task-specific component that inspects the query and proposes
            metadata filters (without touching retrieval prompts).
        applicable_search_types: optional whitelist of search types
            (e.g., {"docket_files"}).  If omitted the policy applies to all.
        max_results_to_consider: cap on how many top results to inspect
            when validating candidate filters.
        min_support: minimum number of matching results required before a
            filter is applied.
        prefetch_multiplier: multiplier used by callers to request additional
            results in a single pass so that pruning still yields enough hits.
    """

    resolver: MetadataIntentResolver
    get_field_values_fn: Optional[Callable[[str, str], Set[str]]] = None
    applicable_search_types: Optional[Sequence[str]] = None
    max_results_to_consider: int = 12
    min_support: int = 1
    prefetch_multiplier: float = 2.0
    metadata_key: str = field(default=AUTO_FILTER_METADATA_KEY, init=False)

    def applies_to(self, search_type: str) -> bool:
        if not self.applicable_search_types:
            return True
        return search_type in self.applicable_search_types


def apply_metadata_filters(
    query: str,
    results: Sequence,
    search_type: str,
    policy: Optional[MetadataFilterPolicy],
) -> Tuple[List, Dict[str, str]]:
    """
    Apply context-aware metadata pruning.

    Returns:
        (filtered_results, applied_filters).  If no filters were applied the
        original results and an empty dict are returned.
    """
    if not policy or not results or not policy.applies_to(search_type):
        return list(results), {}

    get_values = policy.get_field_values_fn or (lambda _field, _stype: set())
    try:
        candidates = policy.resolver.infer_filters(query, search_type, get_values)
    except Exception as exc:
        logger.warning(f"Metadata intent resolver failed: {exc}")
        candidates = {}
    if not candidates:
        return list(results), {}
    
    inferred_filters = _materialize_filters_from_results(
        candidates, results, policy
    )
    if not inferred_filters:
        return list(results), {}
    
    pruned = _prune_results(results, inferred_filters, policy.metadata_key)
    return pruned, inferred_filters


def _materialize_filters_from_results(
    candidates: Dict[str, str],
    results: Sequence,
    policy: MetadataFilterPolicy,
) -> Dict[str, str]:
    inferred: Dict[str, str] = {}
    max_results = min(policy.max_results_to_consider, len(results))
    
    for field_name, candidate_value in candidates.items():
        counter: Counter[str] = Counter()
        candidate_lower = str(candidate_value).lower()
        
        for result in results[:max_results]:
            value = _extract_field_value(result, field_name)
            if value:
                value_lower = str(value).lower()
                if value_lower == candidate_lower:
                    counter[str(value)] += 1
        
        if not counter:
            continue
        
        option, count = counter.most_common(1)[0]
        if count >= policy.min_support:
            inferred[field_name] = option
    
    return inferred


def _prune_results(
    results: Sequence,
    filters: Dict[str, str],
    metadata_key: str,
) -> List:
    filtered: List = []
    for result in results:
        if _matches_filters(result, filters):
            metadata = _ensure_metadata(result)
            metadata.setdefault(metadata_key, dict(filters))
            filtered.append(result)
    
    # If pruning removes everything, return empty list so caller can fall back.
    return filtered


def _matches_filters(result: object, filters: Dict[str, str]) -> bool:
    metadata = getattr(result, "metadata", None) or {}
    for field_name, expected in filters.items():
        value = metadata.get(field_name)
        if value is None:
            return False
        if str(value).strip().lower() != str(expected).strip().lower():
            return False
    return True


def _extract_field_value(result: object, field_name: str) -> Optional[str]:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, MutableMapping):
        value = metadata.get(field_name)
        if value:
            return str(value)
    return None


def _ensure_metadata(result: object) -> MutableMapping:
    if not hasattr(result, "metadata") or result.metadata is None:
        result.metadata = {}
    return result.metadata


def compute_prefetch_k(
    base_k: int, policy: Optional[MetadataFilterPolicy]
) -> int:
    """
    Utility for callers to decide how many results to request in a single pass.
    """
    if not policy:
        return base_k

    multiplier = max(1.0, policy.prefetch_multiplier)
    min_for_validation = max(base_k, policy.max_results_to_consider)
    return max(int(base_k * multiplier), min_for_validation)


