"""
Evaluation module for agent performance analysis and metrics collection.
"""

from .action_classifier import ActionClassifier, ActionClassificationResult
from .metrics_collector import MetricsCollector, StepMetrics, EpisodeMetrics
from .task_performance_tracker import TaskPerformanceTracker, PredictionResult
from .eig_estimator import EIGEstimator, EIGEstimationResult
from .belief_evolution_tracker import BeliefEvolutionTracker, BeliefEvolutionResult
from .metrics_summarizer import MetricsSummarizer, EpisodeSummary, AggregatedMetrics, summarize_single_episode, aggregate_episodes_from_directory
from .execution_integration import (
    create_metrics_collector,
    create_task_performance_tracker,
    integrate_metrics_collection,
    create_instrumented_execution_function
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
    'integrate_metrics_collection',
    'create_instrumented_execution_function',
    'TaskPerformanceTracker',
    'PredictionResult',
    'EIGEstimator',
    'EIGEstimationResult',
    'BeliefEvolutionTracker',
    'BeliefEvolutionResult',
    'MetricsSummarizer',
    'EpisodeSummary',
    'AggregatedMetrics',
    'summarize_single_episode',
    'aggregate_episodes_from_directory',
    'evaluate_entry',
    'evaluate_hallucination_entry',
    'compute_metrics',
    'aggregate_metrics',
    'parse_predictions',
]
