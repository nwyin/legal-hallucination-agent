"""
Factories for metrics collection and final-prediction tracking.
"""

import logging
from polaris_agents.models.llm import ModelAPI
from .action_classifier import ActionClassifier
from .metrics_collector import MetricsCollector
from .task_performance_tracker import TaskPerformanceTracker
from .eig_estimator import EIGEstimator
from .belief_evolution_tracker import BeliefEvolutionTracker

logger = logging.getLogger(__name__)

def create_metrics_collector(
    model_api: ModelAPI,
    model_name: str = "openai/gpt-4o-mini",
    provider: str = "openrouter",
    max_tokens: int = 10000,
    save_dir: str = "metrics",
    enable_action_classification: bool = True,
    enable_task_performance_tracking: bool = True,
    enable_eig_estimation: bool = True,
    enable_belief_evolution_tracking: bool = True,
    domain_knowledge = None
) -> MetricsCollector:
    """
    Create a configured metrics collector with optional metrics tracking.
    
    Available metrics:
    - Action Classification (LLM-as-a-judge): Task vs design focus scoring
    - Task Performance Tracking: Evaluation of the final submitted prediction
    - EIG Estimation (agent-centric): Agent's information gain estimates
    - Belief Evolution Tracking (LLM-as-a-judge): Belief complexity and accuracy
    
    Args:
        model_api: Model API for LLM calls
        model_name: Model to use for action classification and belief evolution tracking
        save_dir: Directory to save metrics files
        enable_action_classification: Whether to enable action classification
        enable_task_performance_tracking: Whether to enable task performance tracking
        enable_eig_estimation: Whether to enable EIG estimation
        enable_belief_evolution_tracking: Whether to enable belief evolution tracking
        
    Returns:
        Configured MetricsCollector instance
    """
    action_classifier = None
    if enable_action_classification:
        action_classifier = ActionClassifier(model_api, model_name, provider, max_tokens=max_tokens, domain_knowledge=domain_knowledge)
        logger.info(f"Created action classifier using {model_name} ({provider})")
    
    eig_estimator = None
    if enable_eig_estimation:
        eig_estimator = EIGEstimator(model_api, model_name, provider, max_tokens=max_tokens, domain_knowledge=domain_knowledge)
        logger.info(f"Created EIG estimator using {model_name} ({provider}) for post-hoc true information gain estimation")
    
    belief_evolution_tracker = None
    if enable_belief_evolution_tracking:
        belief_evolution_tracker = BeliefEvolutionTracker(model_api, model_name, provider, max_tokens=max_tokens)
        logger.info(f"Created belief evolution tracker using {model_name} ({provider})")

    task_performance_tracker = None
    if enable_task_performance_tracking:
        task_performance_tracker = TaskPerformanceTracker()
        logger.info(f"Created task performance tracker for final submitted predictions")
    
    collector = MetricsCollector(action_classifier, task_performance_tracker, eig_estimator, belief_evolution_tracker, save_dir)
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
