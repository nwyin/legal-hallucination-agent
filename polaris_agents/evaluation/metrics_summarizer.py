"""
Metrics summarization utilities for analyzing agent performance.

This module provides functionality to summarize key metrics from episode data,
both for single examples and for aggregation across multiple examples.
"""

import json
import os
import glob
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from .metrics_collector import EpisodeMetrics


@dataclass
class EpisodeSummary:
    """Summary of key metrics for a single episode."""
    episode_id: str
    task_type: str
    total_steps: int
    total_reward: float
    
    # Task performance metrics
    final_accuracy: Optional[float] = None
    overall_accuracy: Optional[float] = None
    average_confidence: Optional[float] = None
    final_confidence: Optional[float] = None
    
    # Action focus metrics
    average_task_focus: Optional[float] = None
    task_focused_actions: int = 0
    design_focused_actions: int = 0
    mixed_actions: int = 0
    
    # EIG metrics
    average_task_eig: Optional[float] = None
    average_design_eig: Optional[float] = None
    average_joint_eig: Optional[float] = None
    
    # Belief evolution metrics
    final_belief_complexity: Optional[float] = None
    final_belief_coherence: Optional[float] = None
    final_belief_information_density: Optional[float] = None
    
    # Bandit-specific metrics (if applicable)
    final_cumulative_regret: Optional[float] = None
    final_kl_divergence: Optional[float] = None
    best_arm_identification: Optional[bool] = None
    
    # Time series data for aggregation
    accuracy_over_time: List[Optional[float]] = None
    confidence_over_time: List[float] = None
    task_focus_over_time: List[Optional[float]] = None
    task_eig_over_time: List[Optional[float]] = None
    design_eig_over_time: List[Optional[float]] = None
    joint_eig_over_time: List[Optional[float]] = None
    cumulative_regret_over_time: List[float] = None
    kl_divergence_over_time: List[Optional[float]] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple episodes."""
    task_type: str
    num_episodes: int
    max_steps: int
    
    # Aggregated final metrics
    mean_final_accuracy: Optional[float] = None
    std_final_accuracy: Optional[float] = None
    mean_overall_accuracy: Optional[float] = None
    std_overall_accuracy: Optional[float] = None
    mean_average_confidence: Optional[float] = None
    std_average_confidence: Optional[float] = None
    
    # Aggregated action focus metrics
    mean_average_task_focus: Optional[float] = None
    std_average_task_focus: Optional[float] = None
    mean_task_focused_actions: Optional[float] = None
    mean_design_focused_actions: Optional[float] = None
    mean_mixed_actions: Optional[float] = None
    
    # Aggregated EIG metrics
    mean_average_task_eig: Optional[float] = None
    std_average_task_eig: Optional[float] = None
    mean_average_design_eig: Optional[float] = None
    std_average_design_eig: Optional[float] = None
    mean_average_joint_eig: Optional[float] = None
    std_average_joint_eig: Optional[float] = None
    
    # Bandit-specific aggregated metrics
    mean_final_cumulative_regret: Optional[float] = None
    std_final_cumulative_regret: Optional[float] = None
    best_arm_identification_rate: Optional[float] = None
    
    # Time series aggregated data
    mean_accuracy_over_time: List[Optional[float]] = None
    std_accuracy_over_time: List[Optional[float]] = None
    mean_confidence_over_time: List[Optional[float]] = None
    std_confidence_over_time: List[Optional[float]] = None
    mean_task_focus_over_time: List[Optional[float]] = None
    std_task_focus_over_time: List[Optional[float]] = None
    mean_cumulative_regret_over_time: List[Optional[float]] = None
    std_cumulative_regret_over_time: List[Optional[float]] = None


class MetricsSummarizer:
    """Summarizes metrics from episode data for analysis and aggregation."""
    
    def __init__(self):
        self.summaries: List[EpisodeSummary] = []
    
    def summarize_episode(self, episode_metrics: EpisodeMetrics) -> EpisodeSummary:
        """
        Create a summary of key metrics from a single episode.
        
        Args:
            episode_metrics: Episode metrics to summarize
            
        Returns:
            EpisodeSummary with key metrics extracted
        """
        # Basic episode info
        episode_id = episode_metrics.episode_id
        task_type = episode_metrics.environment_info.get('task', 'unknown') if episode_metrics.environment_info else 'unknown'
        total_steps = episode_metrics.total_steps
        total_reward = episode_metrics.total_reward
        
        # Extract time series data
        accuracy_over_time = []
        confidence_over_time = []
        task_focus_over_time = []
        task_eig_over_time = []
        design_eig_over_time = []
        joint_eig_over_time = []
        cumulative_regret_over_time = []
        kl_divergence_over_time = []
        
        # Initialize cumulative regret tracking
        current_regret = 0.0
        best_arm_prob = None
        
        for step_metric in episode_metrics.step_metrics:
            # Task performance
            accuracy_over_time.append(step_metric.prediction_accuracy)
            confidence_over_time.append(step_metric.prediction_confidence)
            
            # Task focus
            task_focus_over_time.append(step_metric.task_focus_score)
            
            # EIG metrics
            task_eig_over_time.append(step_metric.task_eig)
            design_eig_over_time.append(step_metric.design_eig)
            joint_eig_over_time.append(step_metric.joint_eig)
            
            # Bandit-specific metrics
            if task_type == 'multi_armed_bandit' and episode_metrics.environment_info:
                env_info = episode_metrics.environment_info
                true_arm_probs = env_info.get("true_arm_probs", [])
                arm_names = env_info.get("arm_names", [])
                best_arm_prob = env_info.get("best_arm_prob", max(true_arm_probs) if true_arm_probs else 0.5)
                
                # Calculate regret for arm pulls
                if (step_metric.action_type == 'PROVIDE_FINAL_RESPONSE' and 
                    step_metric.reward is not None):
                    
                    action_content = getattr(step_metric, 'action_content', '')
                    if 'Action:' in action_content:
                        arm_name = action_content.split('Action:')[1].strip()
                        if arm_name in arm_names:
                            arm_idx = arm_names.index(arm_name)
                            true_prob = true_arm_probs[arm_idx]
                            step_regret = best_arm_prob - true_prob
                            current_regret += step_regret
                
                cumulative_regret_over_time.append(current_regret)
            else:
                cumulative_regret_over_time.append(0.0)
            
            # KL divergence (placeholder - would need belief evolution data)
            kl_divergence_over_time.append(None)
        
        # Calculate summary statistics
        final_accuracy = accuracy_over_time[-1] if accuracy_over_time else None
        # Convert None to np.nan for numpy calculations
        accuracy_values = [a if a is not None else np.nan for a in accuracy_over_time]
        confidence_values = [c if c is not None else np.nan for c in confidence_over_time]
        overall_accuracy = np.nanmean(accuracy_values) if accuracy_over_time else None
        average_confidence = np.nanmean(confidence_values) if confidence_over_time else None
        final_confidence = confidence_over_time[-1] if confidence_over_time else None
        
        # Task focus analysis
        task_focus_values = [tf if tf is not None else np.nan for tf in task_focus_over_time]
        average_task_focus = np.nanmean(task_focus_values) if task_focus_over_time else None
        
        # Count actions (excluding NaN values)
        valid_task_focus = [tf for tf in task_focus_values if not np.isnan(tf)]
        task_focused_actions = sum(1 for tf in valid_task_focus if tf > 0.6)
        design_focused_actions = sum(1 for tf in valid_task_focus if tf < 0.4)
        mixed_actions = len(valid_task_focus) - task_focused_actions - design_focused_actions
        
        # EIG analysis
        task_eig_values = [eig if eig is not None else np.nan for eig in task_eig_over_time]
        design_eig_values = [eig if eig is not None else np.nan for eig in design_eig_over_time]
        joint_eig_values = [eig if eig is not None else np.nan for eig in joint_eig_over_time]
        
        average_task_eig = np.nanmean(task_eig_values) if task_eig_over_time else None
        average_design_eig = np.nanmean(design_eig_values) if design_eig_over_time else None
        average_joint_eig = np.nanmean(joint_eig_values) if joint_eig_over_time else None
        
        # Belief evolution (placeholder - would need actual belief data)
        final_belief_complexity = None
        final_belief_coherence = None
        final_belief_information_density = None
        
        # Bandit-specific analysis
        final_cumulative_regret = cumulative_regret_over_time[-1] if cumulative_regret_over_time else None
        final_kl_divergence = None
        best_arm_identification = None
        
        if task_type == 'multi_armed_bandit' and episode_metrics.environment_info:
            # Check if agent identified the best arm (simplified heuristic)
            env_info = episode_metrics.environment_info
            true_arm_probs = env_info.get("true_arm_probs", [])
            arm_names = env_info.get("arm_names", [])
            
            if true_arm_probs and arm_names:
                best_arm_idx = np.argmax(true_arm_probs)
                best_arm_name = arm_names[best_arm_idx]
                
                # Check last few arm pulls to see if they're pulling the best arm
                recent_pulls = []
                for step_metric in episode_metrics.step_metrics[-5:]:  # Last 5 steps
                    if (step_metric.action_type == 'PROVIDE_FINAL_RESPONSE' and 
                        step_metric.reward is not None):
                        action_content = getattr(step_metric, 'action_content', '')
                        if 'Action:' in action_content:
                            arm_name = action_content.split('Action:')[1].strip()
                            recent_pulls.append(arm_name)
                
                # If more than half of recent pulls are the best arm, consider it identified
                if recent_pulls:
                    best_arm_pulls = sum(1 for arm in recent_pulls if arm == best_arm_name)
                    best_arm_identification = best_arm_pulls >= len(recent_pulls) * 0.5
        
        summary = EpisodeSummary(
            episode_id=episode_id,
            task_type=task_type,
            total_steps=total_steps,
            total_reward=total_reward,
            final_accuracy=final_accuracy,
            overall_accuracy=overall_accuracy,
            average_confidence=average_confidence,
            final_confidence=final_confidence,
            average_task_focus=average_task_focus,
            task_focused_actions=task_focused_actions,
            design_focused_actions=design_focused_actions,
            mixed_actions=mixed_actions,
            average_task_eig=average_task_eig,
            average_design_eig=average_design_eig,
            average_joint_eig=average_joint_eig,
            final_belief_complexity=final_belief_complexity,
            final_belief_coherence=final_belief_coherence,
            final_belief_information_density=final_belief_information_density,
            final_cumulative_regret=final_cumulative_regret,
            final_kl_divergence=final_kl_divergence,
            best_arm_identification=best_arm_identification,
            accuracy_over_time=accuracy_over_time,
            confidence_over_time=confidence_over_time,
            task_focus_over_time=task_focus_over_time,
            task_eig_over_time=task_eig_over_time,
            design_eig_over_time=design_eig_over_time,
            joint_eig_over_time=joint_eig_over_time,
            cumulative_regret_over_time=cumulative_regret_over_time,
            kl_divergence_over_time=kl_divergence_over_time
        )
        
        return summary
    
    def print_episode_summary(self, summary: EpisodeSummary, verbose: bool = False) -> None:
        """
        Print a formatted summary of episode metrics.
        
        Args:
            summary: Episode summary to print
            verbose: Whether to include detailed time series information
        """
        print("=" * 80)
        print(f"EPISODE SUMMARY: {summary.episode_id}")
        print("=" * 80)
        
        print(f"Task Type: {summary.task_type}")
        print(f"Total Steps: {summary.total_steps}")
        print(f"Total Reward: {summary.total_reward:.3f}")
        print()
        
        # Task Performance
        print("TASK PERFORMANCE:")
        if summary.final_accuracy is not None:
            print(f"  Final Accuracy: {summary.final_accuracy:.3f}")
        if summary.overall_accuracy is not None:
            print(f"  Overall Accuracy: {summary.overall_accuracy:.3f}")
        if summary.average_confidence is not None:
            print(f"  Average Confidence: {summary.average_confidence:.3f}")
        if summary.final_confidence is not None:
            print(f"  Final Confidence: {summary.final_confidence:.3f}")
        print()
        
        # Action Focus
        print("ACTION FOCUS:")
        if summary.average_task_focus is not None:
            print(f"  Average Task Focus: {summary.average_task_focus:.3f}")
        print(f"  Task-focused Actions: {summary.task_focused_actions}")
        print(f"  Design-focused Actions: {summary.design_focused_actions}")
        print(f"  Mixed Actions: {summary.mixed_actions}")
        print()
        
        # EIG Metrics
        print("EXPECTED INFORMATION GAIN:")
        if summary.average_task_eig is not None:
            print(f"  Average Task EIG: {summary.average_task_eig:.3f}")
        if summary.average_design_eig is not None:
            print(f"  Average Design EIG: {summary.average_design_eig:.3f}")
        if summary.average_joint_eig is not None:
            print(f"  Average Joint EIG: {summary.average_joint_eig:.3f}")
        print()
        
        # Task-specific metrics
        if summary.task_type == 'multi_armed_bandit':
            print("BANDIT PERFORMANCE:")
            if summary.final_cumulative_regret is not None:
                print(f"  Final Cumulative Regret: {summary.final_cumulative_regret:.3f}")
            if summary.best_arm_identification is not None:
                status = "✓" if summary.best_arm_identification else "✗"
                print(f"  Best Arm Identified: {status}")
            print()
        
        # Verbose time series (optional)
        if verbose:
            print("TIME SERIES DATA:")
            print(f"  Accuracy over time: {summary.accuracy_over_time}")
            print(f"  Confidence over time: {[f'{c:.2f}' for c in summary.confidence_over_time]}")
            if summary.task_type == 'multi_armed_bandit':
                print(f"  Cumulative regret over time: {[f'{r:.2f}' for r in summary.cumulative_regret_over_time]}")
        
        print("=" * 80)
    
    def save_episode_summary(self, summary: EpisodeSummary, filepath: str) -> None:
        """
        Save episode summary to JSON file.
        
        Args:
            summary: Episode summary to save
            filepath: Path to save the summary
        """
        # Convert to dictionary and handle numpy types
        summary_dict = asdict(summary)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        summary_dict = recursive_convert(summary_dict)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        print(f"Episode summary saved to: {filepath}")
    
    def load_episode_summary(self, filepath: str) -> EpisodeSummary:
        """
        Load episode summary from JSON file.
        
        Args:
            filepath: Path to load the summary from
            
        Returns:
            EpisodeSummary object
        """
        with open(filepath, 'r') as f:
            summary_dict = json.load(f)
        
        return EpisodeSummary(**summary_dict)
    
    def load_episodes_from_directory(self, directory: str, pattern: str = "*.json") -> List[EpisodeSummary]:
        """
        Load episode summaries from a directory.
        
        Args:
            directory: Directory containing episode summary JSON files
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            List of EpisodeSummary objects
        """
        summaries = []
        file_pattern = os.path.join(directory, pattern)
        
        for filepath in glob.glob(file_pattern):
            try:
                summary = self.load_episode_summary(filepath)
                summaries.append(summary)
                print(f"Loaded episode summary: {summary.episode_id}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return summaries
    
    def aggregate_episodes(self, summaries: List[EpisodeSummary]) -> AggregatedMetrics:
        """
        Aggregate metrics across multiple episodes.
        
        Args:
            summaries: List of EpisodeSummary objects to aggregate
            
        Returns:
            AggregatedMetrics object with aggregated statistics
        """
        if not summaries:
            raise ValueError("No summaries provided for aggregation")
        
        # Check that all episodes are of the same task type
        task_types = set(s.task_type for s in summaries)
        if len(task_types) > 1:
            print(f"Warning: Mixed task types found: {task_types}. Using first type: {list(task_types)[0]}")
        
        task_type = list(task_types)[0]
        num_episodes = len(summaries)
        
        # Find maximum steps across all episodes
        max_steps = max(s.total_steps for s in summaries)
        
        # Aggregate final metrics
        final_accuracies = [s.final_accuracy for s in summaries if s.final_accuracy is not None]
        overall_accuracies = [s.overall_accuracy for s in summaries if s.overall_accuracy is not None]
        average_confidences = [s.average_confidence for s in summaries if s.average_confidence is not None]
        
        # Convert None to np.nan for final metrics aggregation
        final_acc_values = [acc if acc is not None else np.nan for acc in final_accuracies]
        overall_acc_values = [acc if acc is not None else np.nan for acc in overall_accuracies]
        avg_conf_values = [conf if conf is not None else np.nan for conf in average_confidences]
        
        mean_final_accuracy = np.nanmean(final_acc_values) if final_accuracies else None
        std_final_accuracy = np.nanstd(final_acc_values) if final_accuracies else None
        mean_overall_accuracy = np.nanmean(overall_acc_values) if overall_accuracies else None
        std_overall_accuracy = np.nanstd(overall_acc_values) if overall_accuracies else None
        mean_average_confidence = np.nanmean(avg_conf_values) if average_confidences else None
        std_average_confidence = np.nanstd(avg_conf_values) if average_confidences else None
        
        # Aggregate action focus metrics
        task_focus_scores = [s.average_task_focus for s in summaries if s.average_task_focus is not None]
        task_focused_actions = [s.task_focused_actions for s in summaries]
        design_focused_actions = [s.design_focused_actions for s in summaries]
        mixed_actions = [s.mixed_actions for s in summaries]
        
        # Convert None to np.nan for task focus aggregation
        task_focus_values = [tf if tf is not None else np.nan for tf in task_focus_scores]
        mean_average_task_focus = np.nanmean(task_focus_values) if task_focus_scores else None
        std_average_task_focus = np.nanstd(task_focus_values) if task_focus_scores else None
        mean_task_focused_actions = np.mean(task_focused_actions)
        mean_design_focused_actions = np.mean(design_focused_actions)
        mean_mixed_actions = np.mean(mixed_actions)
        
        # Aggregate EIG metrics
        task_eig_scores = [s.average_task_eig for s in summaries if s.average_task_eig is not None]
        design_eig_scores = [s.average_design_eig for s in summaries if s.average_design_eig is not None]
        joint_eig_scores = [s.average_joint_eig for s in summaries if s.average_joint_eig is not None]
        
        # Convert None to np.nan for EIG aggregation
        task_eig_values = [eig if eig is not None else np.nan for eig in task_eig_scores]
        design_eig_values = [eig if eig is not None else np.nan for eig in design_eig_scores]
        joint_eig_values = [eig if eig is not None else np.nan for eig in joint_eig_scores]
        
        mean_average_task_eig = np.nanmean(task_eig_values) if task_eig_scores else None
        std_average_task_eig = np.nanstd(task_eig_values) if task_eig_scores else None
        mean_average_design_eig = np.nanmean(design_eig_values) if design_eig_scores else None
        std_average_design_eig = np.nanstd(design_eig_values) if design_eig_scores else None
        mean_average_joint_eig = np.nanmean(joint_eig_values) if joint_eig_scores else None
        std_average_joint_eig = np.nanstd(joint_eig_values) if joint_eig_scores else None
        
        # Aggregate bandit-specific metrics
        mean_final_cumulative_regret = None
        std_final_cumulative_regret = None
        best_arm_identification_rate = None
        
        if task_type == 'multi_armed_bandit':
            final_regrets = [s.final_cumulative_regret for s in summaries if s.final_cumulative_regret is not None]
            best_arm_identifications = [s.best_arm_identification for s in summaries if s.best_arm_identification is not None]
            
            if final_regrets:
                mean_final_cumulative_regret = np.mean(final_regrets)
                std_final_cumulative_regret = np.std(final_regrets)
            
            if best_arm_identifications:
                best_arm_identification_rate = np.mean(best_arm_identifications)
        
        # Aggregate time series data
        mean_accuracy_over_time = []
        std_accuracy_over_time = []
        mean_confidence_over_time = []
        std_confidence_over_time = []
        mean_task_focus_over_time = []
        std_task_focus_over_time = []
        mean_cumulative_regret_over_time = []
        std_cumulative_regret_over_time = []
        
        for step in range(max_steps):
            # Aggregate accuracy over time
            step_accuracies = []
            for summary in summaries:
                if step < len(summary.accuracy_over_time):
                    acc_val = summary.accuracy_over_time[step]
                    step_accuracies.append(acc_val if acc_val is not None else np.nan)
                else:
                    step_accuracies.append(np.nan)
            
            mean_acc = np.nanmean(step_accuracies) if step_accuracies else None
            std_acc = np.nanstd(step_accuracies) if len(step_accuracies) > 1 else None
            mean_accuracy_over_time.append(mean_acc)
            std_accuracy_over_time.append(std_acc)
            
            # Aggregate confidence over time
            step_confidences = []
            for summary in summaries:
                if step < len(summary.confidence_over_time):
                    conf_val = summary.confidence_over_time[step]
                    step_confidences.append(conf_val if conf_val is not None else np.nan)
                else:
                    step_confidences.append(np.nan)
            
            mean_conf = np.nanmean(step_confidences) if step_confidences else None
            std_conf = np.nanstd(step_confidences) if len(step_confidences) > 1 else None
            mean_confidence_over_time.append(mean_conf)
            std_confidence_over_time.append(std_conf)
            
            # Aggregate task focus over time
            step_task_focus = []
            for summary in summaries:
                if step < len(summary.task_focus_over_time):
                    tf_val = summary.task_focus_over_time[step]
                    step_task_focus.append(tf_val if tf_val is not None else np.nan)
                else:
                    step_task_focus.append(np.nan)
            
            mean_tf = np.nanmean(step_task_focus) if step_task_focus else None
            std_tf = np.nanstd(step_task_focus) if len(step_task_focus) > 1 else None
            mean_task_focus_over_time.append(mean_tf)
            std_task_focus_over_time.append(std_tf)
            
            # Aggregate cumulative regret over time (for bandit tasks)
            step_regrets = []
            for summary in summaries:
                if step < len(summary.cumulative_regret_over_time):
                    regret_val = summary.cumulative_regret_over_time[step]
                    step_regrets.append(regret_val if regret_val is not None else np.nan)
                else:
                    step_regrets.append(np.nan)
            
            mean_regret = np.nanmean(step_regrets) if step_regrets else None
            std_regret = np.nanstd(step_regrets) if len(step_regrets) > 1 else None
            mean_cumulative_regret_over_time.append(mean_regret)
            std_cumulative_regret_over_time.append(std_regret)
        
        return AggregatedMetrics(
            task_type=task_type,
            num_episodes=num_episodes,
            max_steps=max_steps,
            mean_final_accuracy=mean_final_accuracy,
            std_final_accuracy=std_final_accuracy,
            mean_overall_accuracy=mean_overall_accuracy,
            std_overall_accuracy=std_overall_accuracy,
            mean_average_confidence=mean_average_confidence,
            std_average_confidence=std_average_confidence,
            mean_average_task_focus=mean_average_task_focus,
            std_average_task_focus=std_average_task_focus,
            mean_task_focused_actions=mean_task_focused_actions,
            mean_design_focused_actions=mean_design_focused_actions,
            mean_mixed_actions=mean_mixed_actions,
            mean_average_task_eig=mean_average_task_eig,
            std_average_task_eig=std_average_task_eig,
            mean_average_design_eig=mean_average_design_eig,
            std_average_design_eig=std_average_design_eig,
            mean_average_joint_eig=mean_average_joint_eig,
            std_average_joint_eig=std_average_joint_eig,
            mean_final_cumulative_regret=mean_final_cumulative_regret,
            std_final_cumulative_regret=std_final_cumulative_regret,
            best_arm_identification_rate=best_arm_identification_rate,
            mean_accuracy_over_time=mean_accuracy_over_time,
            std_accuracy_over_time=std_accuracy_over_time,
            mean_confidence_over_time=mean_confidence_over_time,
            std_confidence_over_time=std_confidence_over_time,
            mean_task_focus_over_time=mean_task_focus_over_time,
            std_task_focus_over_time=std_task_focus_over_time,
            mean_cumulative_regret_over_time=mean_cumulative_regret_over_time,
            std_cumulative_regret_over_time=std_cumulative_regret_over_time
        )
    
    def print_aggregated_summary(self, aggregated: AggregatedMetrics) -> None:
        """
        Print a formatted summary of aggregated metrics.
        
        Args:
            aggregated: AggregatedMetrics to print
        """
        print("=" * 80)
        print(f"AGGREGATED METRICS SUMMARY ({aggregated.num_episodes} episodes)")
        print("=" * 80)
        
        print(f"Task Type: {aggregated.task_type}")
        print(f"Number of Episodes: {aggregated.num_episodes}")
        print(f"Max Steps: {aggregated.max_steps}")
        print()
        
        # Task Performance
        print("TASK PERFORMANCE:")
        if aggregated.mean_final_accuracy is not None:
            print(f"  Mean Final Accuracy: {aggregated.mean_final_accuracy:.3f} ± {aggregated.std_final_accuracy:.3f}")
        if aggregated.mean_overall_accuracy is not None:
            print(f"  Mean Overall Accuracy: {aggregated.mean_overall_accuracy:.3f} ± {aggregated.std_overall_accuracy:.3f}")
        if aggregated.mean_average_confidence is not None:
            print(f"  Mean Average Confidence: {aggregated.mean_average_confidence:.3f} ± {aggregated.std_average_confidence:.3f}")
        print()
        
        # Action Focus
        print("ACTION FOCUS:")
        if aggregated.mean_average_task_focus is not None:
            print(f"  Mean Average Task Focus: {aggregated.mean_average_task_focus:.3f} ± {aggregated.std_average_task_focus:.3f}")
        print(f"  Mean Task-focused Actions: {aggregated.mean_task_focused_actions:.1f}")
        print(f"  Mean Design-focused Actions: {aggregated.mean_design_focused_actions:.1f}")
        print(f"  Mean Mixed Actions: {aggregated.mean_mixed_actions:.1f}")
        print()
        
        # EIG Metrics
        print("EXPECTED INFORMATION GAIN:")
        if aggregated.mean_average_task_eig is not None:
            print(f"  Mean Task EIG: {aggregated.mean_average_task_eig:.3f} ± {aggregated.std_average_task_eig:.3f}")
        if aggregated.mean_average_design_eig is not None:
            print(f"  Mean Design EIG: {aggregated.mean_average_design_eig:.3f} ± {aggregated.std_average_design_eig:.3f}")
        if aggregated.mean_average_joint_eig is not None:
            print(f"  Mean Joint EIG: {aggregated.mean_average_joint_eig:.3f} ± {aggregated.std_average_joint_eig:.3f}")
        print()
        
        # Task-specific metrics
        if aggregated.task_type == 'multi_armed_bandit':
            print("BANDIT PERFORMANCE:")
            if aggregated.mean_final_cumulative_regret is not None:
                print(f"  Mean Final Cumulative Regret: {aggregated.mean_final_cumulative_regret:.3f} ± {aggregated.std_final_cumulative_regret:.3f}")
            if aggregated.best_arm_identification_rate is not None:
                print(f"  Best Arm Identification Rate: {aggregated.best_arm_identification_rate:.3f}")
            print()
        
        print("=" * 80)
    
    def save_aggregated_summary(self, aggregated: AggregatedMetrics, filepath: str) -> None:
        """
        Save aggregated summary to JSON file.
        
        Args:
            aggregated: AggregatedMetrics to save
            filepath: Path to save the summary
        """
        # Convert to dictionary and handle numpy types
        aggregated_dict = asdict(aggregated)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        aggregated_dict = recursive_convert(aggregated_dict)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(aggregated_dict, f, indent=2)
        
        print(f"Aggregated summary saved to: {filepath}")


def summarize_single_episode(
    episode_metrics: EpisodeMetrics, 
    output_dir: str = "metrics_summaries",
    save_json: bool = True,
    verbose: bool = False
) -> EpisodeSummary:
    """
    Convenience function to summarize a single episode.
    
    Args:
        episode_metrics: Episode metrics to summarize
        output_dir: Directory to save summary files
        save_json: Whether to save summary to JSON file
        verbose: Whether to print verbose output
        
    Returns:
        EpisodeSummary object
    """
    summarizer = MetricsSummarizer()
    summary = summarizer.summarize_episode(episode_metrics)
    
    # Print summary
    summarizer.print_episode_summary(summary, verbose=verbose)
    
    # Save to JSON if requested
    if save_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{summary.episode_id}_summary_{timestamp}.json"
        
        # Create proper directory structure: output_dir/task_type/method/episode_id/
        task_type = summary.task_type
        episode_id = summary.episode_id
        
        # Determine method based on episode_id patterns
        if "test" in episode_id.lower():
            method = "test"
        elif "ids_oed" in episode_id.lower() or any(char.isdigit() for char in episode_id):
            method = "ids_oed"
        else:
            method = "unknown"
        
        # Create directory path
        episode_dir = os.path.join(output_dir, task_type, method)
        os.makedirs(episode_dir, exist_ok=True)
        
        filepath = os.path.join(episode_dir, filename)
        summarizer.save_episode_summary(summary, filepath)
    
    return summary


def aggregate_episodes_from_directory(
    directory: str, 
    pattern: str = "*.json",
    output_dir: str = "metrics_summaries",
    save_json: bool = True
) -> AggregatedMetrics:
    """
    Convenience function to aggregate episodes from a directory.
    
    Args:
        directory: Directory containing episode summary JSON files
        pattern: File pattern to match (default: "*.json")
        output_dir: Directory to save aggregated summary
        save_json: Whether to save aggregated summary to JSON file
        
    Returns:
        AggregatedMetrics object
    """
    summarizer = MetricsSummarizer()
    
    # Load all episode summaries
    summaries = summarizer.load_episodes_from_directory(directory, pattern)
    
    if not summaries:
        raise ValueError(f"No episode summaries found in {directory} with pattern {pattern}")
    
    print(f"\nAggregating {len(summaries)} episodes...")
    
    # Aggregate the summaries
    aggregated = summarizer.aggregate_episodes(summaries)
    
    # Print summary
    summarizer.print_aggregated_summary(aggregated)
    
    # Save to JSON if requested
    if save_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aggregated_summary_{aggregated.task_type}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        summarizer.save_aggregated_summary(aggregated, filepath)
    
    return aggregated
