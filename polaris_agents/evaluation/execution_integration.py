"""Factories for metrics collection and final-prediction tracking."""

import logging
from .metrics_collector import MetricsCollector
from .task_performance_tracker import TaskPerformanceTracker

logger = logging.getLogger(__name__)


def create_metrics_collector(
    save_dir: str = "metrics",
    enable_task_performance_tracking: bool = True,
) -> MetricsCollector:
    """Create an episode recorder with optional final-prediction tracking.

    Disabling task performance tracking leaves trajectory recording available.
    No model calls are needed to collect metrics or score submitted predictions.
    """
    tracker = create_task_performance_tracker() if enable_task_performance_tracking else None
    collector = MetricsCollector(task_performance_tracker=tracker, save_dir=save_dir)
    logger.info(f"Created metrics collector with save directory: {save_dir}")
    return collector


def create_task_performance_tracker() -> TaskPerformanceTracker:
    """
    Create a configured task performance tracker.
    
    Returns:
        Configured TaskPerformanceTracker instance
    """
    tracker = TaskPerformanceTracker()
    logger.info("Created task performance tracker")
    return tracker
