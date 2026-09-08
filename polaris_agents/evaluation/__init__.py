"""
Evaluation module for agent performance analysis and metrics collection.
"""

from .action_classifier import ActionClassifier, ActionClassificationResult
from .metrics_collector import MetricsCollector, StepMetrics, EpisodeMetrics
from .task_performance_tracker import TaskPerformanceTracker, PredictionResult
from .eig_estimator import EIGEstimator, EIGEstimationResult
from .belief_evolution_tracker import BeliefEvolutionTracker, BeliefEvolutionResult
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
    'ActionClassifier',
    'ActionClassificationResult',
    'MetricsCollector',
    'StepMetrics',
    'EpisodeMetrics',
    'create_metrics_collector',
    'create_task_performance_tracker',
    'TaskPerformanceTracker',
    'PredictionResult',
    'EIGEstimator',
    'EIGEstimationResult',
    'BeliefEvolutionTracker',
    'BeliefEvolutionResult',
    'evaluate_entry',
    'evaluate_hallucination_entry',
    'compute_metrics',
    'aggregate_metrics',
    'parse_predictions',
]
