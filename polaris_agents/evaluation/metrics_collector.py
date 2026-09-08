"""
Metrics Collection Module for Agent Execution Evaluation

This module provides a unified way to collect and store metrics during agent execution,
enabling later analysis and plotting without re-running episodes.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from .action_classifier import ActionClassifier, ActionClassificationResult
from .task_performance_tracker import TaskPerformanceTracker, PredictionResult
from .eig_estimator import EIGEstimator, EIGEstimationResult
from .belief_evolution_tracker import BeliefEvolutionTracker, BeliefEvolutionResult

logger = logging.getLogger(__name__)

@dataclass
class StepMetrics:
    """Metrics collected for a single execution step"""
    step: int
    action_type: str
    action_content: str
    reward: Optional[float] = None
    task_focus_score: Optional[float] = None
    classification_confidence: Optional[float] = None
    classification_reasoning: Optional[str] = None
    # Task performance tracking fields
    prediction: Optional[str] = None
    prediction_confidence: Optional[float] = None
    prediction_correct: Optional[bool] = None
    prediction_accuracy: Optional[float] = None
    # EIG estimation fields
    task_eig: Optional[float] = None
    design_eig: Optional[float] = None
    joint_eig: Optional[float] = None
    eig_confidence: Optional[float] = None
    eig_reasoning: Optional[str] = None
    # Belief evolution fields
    task_belief_accuracy: Optional[float] = None
    task_belief_mean_kl_bits: Optional[float] = None
    task_belief_baseline_kl_bits: Optional[float] = None
    task_belief_uncertainty: Optional[float] = None
    inferred_arm_probs: Optional[List[float]] = None
    task_belief_complexity: Optional[float] = None
    design_belief_complexity: Optional[float] = None
    belief_coherence: Optional[float] = None
    information_density: Optional[float] = None
    complexity_confidence: Optional[float] = None
    complexity_reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EpisodeMetrics:
    """Complete metrics for an episode"""
    episode_id: str
    task_name: str
    agent_type: str
    environment_info: Dict[str, Any]
    start_time: str
    method: Optional[str] = None
    model_id: Optional[str] = None
    end_time: Optional[str] = None
    total_steps: int = 0
    total_reward: float = 0.0
    final_outcome: Optional[str] = None
    step_metrics: List[StepMetrics] = None
    classification_summary: Optional[Dict[str, Any]] = None
    task_performance_summary: Optional[Dict[str, Any]] = None
    eig_summary: Optional[Dict[str, Any]] = None
    belief_evolution_summary: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.step_metrics is None:
            self.step_metrics = []

class MetricsCollector:
    """
    Collects and stores metrics during agent execution for later analysis.
    """
    
    def __init__(
        self, 
        action_classifier: Optional[ActionClassifier] = None,
        task_performance_tracker: Optional[TaskPerformanceTracker] = None,
        eig_estimator: Optional[EIGEstimator] = None,
        belief_evolution_tracker: Optional[BeliefEvolutionTracker] = None,
        save_dir: str = "metrics"
    ):
        """
        Initialize the metrics collector.
        
        Args:
            action_classifier: Optional action classifier for task/design focus evaluation
            task_performance_tracker: Optional task performance tracker for prediction evaluation
            eig_estimator: Optional EIG estimator for information gain evaluation
            save_dir: Directory to save metrics files
        """
        self.action_classifier = action_classifier
        self.task_performance_tracker = task_performance_tracker
        self.eig_estimator = eig_estimator
        self.belief_evolution_tracker = belief_evolution_tracker
        self.save_dir = save_dir
        self.current_episode: Optional[EpisodeMetrics] = None
        self.step_count = 0
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
    
    def start_episode(
        self, 
        episode_id: str, 
        task_name: str, 
        agent_type: str,
        environment_info: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> None:
        """
        Start collecting metrics for a new episode.
        
        Args:
            episode_id: Unique identifier for this episode
            task_name: Name of the task being executed
            agent_type: Type of agent being used
            environment_info: Information about the environment
            method: Method name (e.g., 'ids_oed', 'boed', 'reflexion')
            model_id: Model ID used (e.g., 'openrouter ID like openai/gpt-4o-mini')
        """
        self.current_episode = EpisodeMetrics(
            episode_id=episode_id,
            task_name=task_name,
            method=method,
            model_id=model_id,
            agent_type=agent_type,
            environment_info=environment_info or {},
            start_time=datetime.now().isoformat(),
            step_metrics=[]
        )
        # Store method separately for directory structure
        self.current_episode_method = method
        self.step_count = 0
        
        logger.info(f"Started metrics collection for episode {episode_id}")
    
    def set_task_performance_ground_truth(self, ground_truth: Any, environment: Any = None) -> None:
        """
        Set the ground truth for task performance tracking.
        
        Args:
            ground_truth: The correct answer for evaluation
            environment: Optional environment instance to use for is_correct() evaluation
        """
        if self.task_performance_tracker:
            self.task_performance_tracker.set_ground_truth(ground_truth, self.current_episode.task_name, environment)
            logger.info(f"Set ground truth for task performance tracking: {ground_truth}")
        else:
            logger.warning("Task performance tracker not available - cannot set ground truth")
    
    def set_belief_evolution_ground_truth(self, ground_truth: Any, task_type: str) -> None:
        """
        Set ground truth for belief evolution tracking.
        
        Args:
            ground_truth: Ground truth for belief accuracy evaluation (e.g., true arm probabilities)
            task_type: Type of task (e.g., "multi_armed_bandit")
        """
        if self.belief_evolution_tracker:
            self.belief_evolution_tracker.set_ground_truth(ground_truth, task_type)
            logger.info(f"Set ground truth for belief evolution tracking: {ground_truth}")
        else:
            logger.warning("Belief evolution tracker not available - cannot set ground truth")
    
    def record_step(
        self, 
        action, 
        agent,
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record metrics for a single execution step.
        
        Args:
            action: The action taken
            agent: The agent instance
            reward: Optional reward received
            metadata: Optional additional metadata
        """
        if self.current_episode is None:
            logger.warning("No active episode - call start_episode() first")
            return
        
        self.step_count += 1
        
        # Extract basic step information
        action_type = action.action_type.value if hasattr(action, 'action_type') else str(type(action))
        action_content = self._extract_action_content(action)
        
        # Initialize step metrics
        step_metrics = StepMetrics(
            step=self.step_count,
            action_type=action_type,
            action_content=action_content,
            reward=reward,
            metadata=metadata or {}
        )
        
        # Extract beliefs and observation from metadata if available
        current_beliefs = None
        if metadata:
            current_beliefs = metadata.get('current_beliefs')
        
        observation_result = None
        observation_metadata = None
        if metadata:
            observation_result = metadata.get('observation_result')
            observation_metadata = metadata.get('observation_metadata')
        
        # External LLM-based metrics
        # Uses external LLM to classify task vs design focus of current action
        if self.action_classifier:
            try:
                context = {
                    'task_name': self.current_episode.task_name,
                    'environment_info': str(self.current_episode.environment_info),
                    'current_beliefs': current_beliefs  # Pass full beliefs
                }
                classification_result = self.action_classifier.classify_action(
                    step=self.step_count,
                    action=action,
                    context=context
                )
                
                step_metrics.task_focus_score = classification_result.task_focus_score
                step_metrics.classification_confidence = classification_result.confidence
                step_metrics.classification_reasoning = classification_result.reasoning
                
            except Exception as e:
                logger.warning(f"Action classification failed at step {self.step_count}: {e}")
        
        # Uses external LLM to estimate TRUE information gain (post-hoc) of the current action
        if self.eig_estimator:
            try:
                context = {
                    'task_name': self.current_episode.task_name,
                    'environment_info': str(self.current_episode.environment_info),
                    'current_beliefs': current_beliefs,  # Pass full beliefs
                    'observation_result': observation_result,  # Pass full observation
                    'observation_metadata': observation_metadata
                }
                eig_result = self.eig_estimator.estimate_eig_at_step(
                    step=self.step_count,
                    agent=agent,
                    action=action,
                    context=context
                )
                
                step_metrics.task_eig = eig_result.task_eig
                step_metrics.design_eig = eig_result.design_eig
                step_metrics.joint_eig = eig_result.joint_eig
                step_metrics.eig_confidence = eig_result.confidence
                step_metrics.eig_reasoning = eig_result.reasoning
                
            except Exception as e:
                logger.warning(f"EIG estimation failed at step {self.step_count}: {e}")
        
        # Prediction evaluation is done once at end of episode via record_final_prediction(), not every step.

        # At each step, evaluates belief complexity (LLM-as-a-judge) and bandit-specific accuracy (agent model-based)
        if self.belief_evolution_tracker:
            try:
                # Extract beliefs from agent if available
                task_beliefs = getattr(agent, 'task_beliefs', '')
                design_beliefs = getattr(agent, 'design_beliefs', '')
                
                if task_beliefs or design_beliefs:
                    belief_result = self.belief_evolution_tracker.track_belief_evolution(
                        step=self.step_count,
                        task_beliefs=task_beliefs,
                        design_beliefs=design_beliefs,
                        agent=agent
                    )
                    
                    step_metrics.task_belief_accuracy = belief_result.task_belief_accuracy
                    step_metrics.task_belief_mean_kl_bits = belief_result.task_belief_mean_kl_bits
                    step_metrics.task_belief_baseline_kl_bits = belief_result.task_belief_baseline_kl_bits
                    step_metrics.task_belief_uncertainty = belief_result.task_belief_uncertainty
                    step_metrics.inferred_arm_probs = belief_result.inferred_arm_probs
                    step_metrics.task_belief_complexity = belief_result.task_belief_complexity
                    step_metrics.design_belief_complexity = belief_result.design_belief_complexity
                    step_metrics.belief_coherence = belief_result.belief_coherence
                    step_metrics.information_density = belief_result.information_density
                    step_metrics.complexity_confidence = belief_result.complexity_confidence
                    step_metrics.complexity_reasoning = belief_result.complexity_reasoning
                    
            except Exception as e:
                logger.warning(f"Belief evolution tracking failed at step {self.step_count}: {e}")
        
        # Add to episode
        self.current_episode.step_metrics.append(step_metrics)
        
        # Update running totals
        if reward is not None:
            self.current_episode.total_reward += reward
        
        logger.info(f"Recorded metrics for step {self.step_count}: {action_type}")

    def record_final_prediction(self, agent, environment=None) -> None:
        """
        Evaluate the agent's final submitted prediction once (no per-step evaluation).
        Uses the already-submitted answer; does not call the LLM.

        Call this after the episode loop, before end_episode().
        """
        if not self.task_performance_tracker or not self.current_episode:
            return
        try:
            self.task_performance_tracker.evaluate_final_prediction(
                agent=agent,
                environment=environment,
                step=self.step_count,
            )
        except Exception as e:
            logger.warning(f"Final prediction evaluation failed: {e}")

    def evaluate_task_performance(self, agent) -> None:
        """
        Evaluate task performance for all steps using the provided agent.
        
        Args:
            agent: The agent instance to evaluate predictions for
        """
        if not self.task_performance_tracker or not self.current_episode:
            return
        
        logger.info("Evaluating task performance for all steps...")
        
        for step_metric in self.current_episode.step_metrics:
            try:
                # Get prediction from agent
                prediction, confidence = agent.get_current_prediction()
                
                # Evaluate correctness if ground truth is set
                is_correct = None
                
                if self.task_performance_tracker.ground_truth is not None:
                    is_correct = self.task_performance_tracker._evaluate_correctness(
                        prediction, self.task_performance_tracker.ground_truth
                    )
                
                # Update step metrics
                step_metric.prediction = prediction
                step_metric.prediction_confidence = confidence
                step_metric.prediction_correct = is_correct
                
                logger.debug(f"Step {step_metric.step}: Prediction='{prediction}', "
                           f"Confidence={confidence:.3f}, Correct={is_correct}")
                
            except Exception as e:
                logger.warning(f"Task performance evaluation failed for step {step_metric.step}: {e}")
    
    def end_episode(self, final_outcome: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None) -> str:
        """
        End the current episode and save metrics.
        
        Args:
            final_outcome: Optional final outcome description
            extra_data: Optional dict to merge into saved JSON (e.g. list_hallucinations, predicted_hallucinations for legal_hallucination_checker)
            
        Returns:
            Path to the saved metrics file
        """
        if self.current_episode is None:
            logger.warning("No active episode to end")
            return ""
        
        # Finalize episode
        self.current_episode.end_time = datetime.now().isoformat()
        self.current_episode.total_steps = self.step_count
        self.current_episode.final_outcome = final_outcome
        
        # Get classification summary if available
        if self.action_classifier:
            self.current_episode.classification_summary = self.action_classifier.get_classification_summary()
        
        # Get task performance summary if available
        if self.task_performance_tracker:
            self.current_episode.task_performance_summary = self.task_performance_tracker.get_performance_summary()
        
        # Get EIG summary if available
        if self.eig_estimator:
            self.current_episode.eig_summary = self.eig_estimator.get_eig_summary()
        
        # Get belief evolution summary if available
        if self.belief_evolution_tracker:
            self.current_episode.belief_evolution_summary = self.belief_evolution_tracker.get_belief_evolution_summary()
        
        
        # Save metrics directly to save_dir/{episode_id}.json
        # The caller is responsible for setting save_dir to the appropriate path
        # (e.g., metrics/{task}/{model_id}/{method}/)
        os.makedirs(self.save_dir, exist_ok=True)
        
        filename = f"{self.current_episode.episode_id}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert to dict for JSON serialization
        episode_dict = asdict(self.current_episode)
        
        # Convert SearchResult objects to dictionaries for JSON serialization
        episode_dict = self._convert_search_results_to_dict(episode_dict)
        
        # Merge extra task-specific data (e.g. ground truth + prediction for hallucination checker)
        if extra_data:
            episode_dict.update(extra_data)
        
        with open(filepath, 'w') as f:
            json.dump(episode_dict, f, indent=2)
        
        logger.info(f"Episode metrics saved to {filepath}")
        
        # Clear current episode
        episode_id = self.current_episode.episode_id
        self.current_episode = None
        self.current_episode_method = None
        self.step_count = 0
        
        return filepath
    
    def _convert_search_results_to_dict(self, obj):
        """
        Recursively convert SearchResult objects to dictionaries for JSON serialization.
        """
        if isinstance(obj, dict):
            return {key: self._convert_search_results_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_search_results_to_dict(item) for item in obj]
        elif hasattr(obj, '__class__') and 'SearchResult' in obj.__class__.__name__:
            # Convert SearchResult object to dictionary
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                # Fallback if SearchResult doesn't have __dict__
                return {
                    "title": getattr(obj, 'title', ''),
                    "url": getattr(obj, 'url', ''),
                    "content": getattr(obj, 'content', ''),
                    "snippet": getattr(obj, 'snippet', ''),
                    "score": getattr(obj, 'score', 0.0),
                    "published_date": getattr(obj, 'published_date', None),
                    "result_id": getattr(obj, 'result_id', ''),
                    "metadata": getattr(obj, 'metadata', {})
                }
        else:
            return obj
    
    def _extract_action_content(self, action) -> str:
        content_parts = []
        
        # For CLOSED_SEARCH actions, show Search Type first, then Query
        if hasattr(action, 'search_type') and action.search_type:
            content_parts.append(f"Search Type: {action.search_type}")
        if hasattr(action, 'query') and action.query:
            content_parts.append(f"Query: {action.query}")
        if hasattr(action, 'thought') and action.thought:
            content_parts.append(f"Thought: {action.thought}")
        if hasattr(action, 'response') and action.response:
            content_parts.append(f"Response: {action.response}")
        
        return "\n".join(content_parts) if content_parts else str(action)
    
    
    
    @staticmethod
    def load_episode_metrics(filepath: str) -> EpisodeMetrics:
        """
        Load episode metrics from a JSON file.
        
        Args:
            filepath: Path to the metrics JSON file
            
        Returns:
            EpisodeMetrics object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert step metrics back to StepMetrics objects
        step_metrics = [
            StepMetrics(**step_data) for step_data in data.get('step_metrics', [])
        ]
        
        # Create EpisodeMetrics object
        episode = EpisodeMetrics(
            episode_id=data['episode_id'],
            task_name=data['task_name'],
            agent_type=data['agent_type'],
            environment_info=data['environment_info'],
            start_time=data['start_time'],
            end_time=data.get('end_time'),
            total_steps=data.get('total_steps', 0),
            total_reward=data.get('total_reward', 0.0),
            final_outcome=data.get('final_outcome'),
            step_metrics=step_metrics,
            classification_summary=data.get('classification_summary'),
            task_performance_summary=data.get('task_performance_summary'),
            eig_summary=data.get('eig_summary'),
            belief_evolution_summary=data.get('belief_evolution_summary')
        )
        
        return episode
    
    @staticmethod
    def load_all_metrics(metrics_dir: str) -> List[EpisodeMetrics]:
        """
        Load all episode metrics from a directory with new structure.
        Searches in metrics/{task_name}/{method}/{episode_id}.json
        
        Args:
            metrics_dir: Directory containing metrics JSON files
            
        Returns:
            List of EpisodeMetrics objects
        """
        episodes = []
        
        # Walk through the directory structure: metrics/{task_name}/{method}/
        for root, dirs, files in os.walk(metrics_dir):
            for filename in files:
                if filename.endswith('.json') and not filename.endswith('_summary.json'):
                    filepath = os.path.join(root, filename)
                    try:
                        episode = MetricsCollector.load_episode_metrics(filepath)
                        episodes.append(episode)
                    except Exception as e:
                        logger.warning(f"Failed to load metrics from {filename}: {e}")
        
        return episodes
