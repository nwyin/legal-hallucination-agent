"""
Evaluation module for agent performance analysis and metrics collection.
"""

from .metrics_collector import MetricsCollector, StepMetrics, EpisodeMetrics
from .task_performance_tracker import TaskPerformanceTracker, PredictionResult
from .execution_integration import (
    create_metrics_collector,
    create_task_performance_tracker,
)
from .hallucination_checker_evaluator import (
    evaluate_entry,
    evaluate_hallucination_entry,
    compute_metrics,
    aggregate_metrics,
    parse_predictions,
)

__all__ = [
    'MetricsCollector',
    'StepMetrics',
    'EpisodeMetrics',
    'create_metrics_collector',
    'create_task_performance_tracker',
    'TaskPerformanceTracker',
    'PredictionResult',
    'evaluate_entry',
    'evaluate_hallucination_entry',
    'compute_metrics',
    'aggregate_metrics',
    'parse_predictions',
]
