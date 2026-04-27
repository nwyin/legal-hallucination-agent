"""
Integration utilities for adding metrics collection to execution loops.
"""

import logging
from typing import Dict, Any, Optional, Union
from polaris_agents.models.llm import ModelAPI
from .action_classifier import ActionClassifier
from .metrics_collector import MetricsCollector
from .task_performance_tracker import TaskPerformanceTracker
from .eig_estimator import EIGEstimator
from .belief_evolution_tracker import BeliefEvolutionTracker

logger = logging.getLogger(__name__)

def create_metrics_collector(
    model_api: ModelAPI,
    model_name: str = "gpt-4o-mini",
    provider: str = "openai",
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
    - Task Performance Tracking (agent-centric): Agent's predictions and confidence
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
        logger.info(f"Created task performance tracker using agent's own model")
    
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

def integrate_metrics_collection(
    run_episode_func,
    model_api: Optional[ModelAPI] = None,
    model_name: str = "gpt-4o-mini",
    save_dir: str = "metrics",
    enable_action_classification: bool = True
):
    """
    Decorator to integrate metrics collection into existing run_episode functions.
    
    This decorator wraps existing run_episode functions to automatically collect
    metrics during execution without modifying the original function.
    
    Args:
        run_episode_func: The original run_episode function to wrap
        model_api: Model API for LLM calls (required if enable_action_classification=True)
        model_name: Model to use for action classification
        save_dir: Directory to save metrics files
        enable_action_classification: Whether to enable action classification
        
    Returns:
        Wrapped function that includes metrics collection
    """
    def wrapper(agent, environment, *args, **kwargs):
        # Create metrics collector
        collector = create_metrics_collector(
            model_api=model_api,
            model_name=model_name,
            save_dir=save_dir,
            enable_action_classification=enable_action_classification
        )
        
        # Generate episode ID
        import uuid
        episode_id = f"{environment.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        
        # Start metrics collection
        collector.start_episode(
            episode_id=episode_id,
            task_name=getattr(environment, 'task_name', environment.__class__.__name__),
            agent_type=agent.__class__.__name__,
            environment_info=getattr(environment, 'get_info', lambda: {})()
        )
        
        # Monkey patch the environment's step method to record metrics
        original_step = environment.step
        original_update_state = getattr(agent, 'update_state', lambda a, o: None)
        
        def instrumented_step(action):
            # Execute the original step
            observation = original_step(action)
            
            # Record metrics
            reward = None
            if hasattr(observation, 'metadata') and 'reward' in observation.metadata:
                reward = observation.metadata['reward']
            
            collector.record_step(action, agent, reward)
            
            return observation
        
        def instrumented_update_state(action, observation):
            # Call original update state
            original_update_state(action, observation)
        
        # Replace methods with instrumented versions
        environment.step = instrumented_step
        if hasattr(agent, 'update_state'):
            agent.update_state = instrumented_update_state
        
        try:
            # Run the original episode function
            result = run_episode_func(agent, environment, *args, **kwargs)
            
            # End metrics collection
            metrics_filepath = collector.end_episode()
            
            # Add metrics info to result
            if isinstance(result, dict):
                result['metrics_filepath'] = metrics_filepath
                result['episode_id'] = episode_id
            
            logger.info(f"Episode completed with metrics saved to: {metrics_filepath}")
            
            return result
            
        except Exception as e:
            # End metrics collection even on error
            collector.end_episode(f"Error: {str(e)}")
            raise
        
        finally:
            # Restore original methods
            environment.step = original_step
            if hasattr(agent, 'update_state'):
                agent.update_state = original_update_state
    
    return wrapper

def create_instrumented_execution_function(
    original_run_episode_func,
    model_api: ModelAPI,
    model_name: str = "gpt-4o-mini",
    save_dir: str = "metrics",
    enable_action_classification: bool = True
):
    """
    Create a new execution function with integrated metrics collection.
    
    This is an alternative to the decorator approach that creates a completely
    new function with metrics collection built in.
    
    Args:
        original_run_episode_func: The original run_episode function
        model_api: Model API for LLM calls
        model_name: Model to use for action classification
        save_dir: Directory to save metrics files
        enable_action_classification: Whether to enable action classification
        
    Returns:
        New function that includes metrics collection
    """
    def instrumented_run_episode(agent, environment, *args, **kwargs):
        # Create metrics collector
        collector = create_metrics_collector(
            model_api=model_api,
            model_name=model_name,
            save_dir=save_dir,
            enable_action_classification=enable_action_classification
        )
        
        # Generate episode ID
        import uuid
        episode_id = f"{environment.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        
        # Start metrics collection
        collector.start_episode(
            episode_id=episode_id,
            task_name=getattr(environment, 'task_name', environment.__class__.__name__),
            agent_type=agent.__class__.__name__,
            environment_info=getattr(environment, 'get_info', lambda: {})()
        )
        
        # Run the original episode function with metrics collection
        try:
            result = run_episode_func(agent, environment, collector, *args, **kwargs)
            
            # End metrics collection
            metrics_filepath = collector.end_episode()
            
            # Add metrics info to result
            if isinstance(result, dict):
                result['metrics_filepath'] = metrics_filepath
                result['episode_id'] = episode_id
            
            return result
            
        except Exception as e:
            # End metrics collection even on error
            collector.end_episode(f"Error: {str(e)}")
            raise
    
    return instrumented_run_episode
