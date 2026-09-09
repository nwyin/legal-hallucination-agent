"""Recording: episode metrics collection and episode logging."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .agent import Agent
    from .environment import Environment

logger = logging.getLogger(__name__)


# --- Episode metrics collection ---


@dataclass
class StepMetrics:
    """Metrics collected for a single execution step"""
    step: int
    action_type: str
    action_content: str
    reward: Optional[float] = None
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
    
    def __post_init__(self):
        if self.step_metrics is None:
            self.step_metrics = []

class MetricsCollector:
    """
    Collects and stores metrics during agent execution for later analysis.
    """
    
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = save_dir
        self.current_episode: Optional[EpisodeMetrics] = None
        self.step_count = 0
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
    
    def record_step(
        self, 
        action, 
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record metrics for a single execution step.
        
        Args:
            action: The action taken
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
        
        # Add to episode
        self.current_episode.step_metrics.append(step_metrics)
        
        # Update running totals
        if reward is not None:
            self.current_episode.total_reward += reward
        
        logger.info(f"Recorded metrics for step {self.step_count}: {action_type}")

    def end_episode(self, final_outcome: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None) -> str:
        """
        End the current episode and save metrics.
        
        Args:
            final_outcome: Optional final outcome description
            extra_data: Optional dict to merge into the saved JSON (ground truth, prediction, scores)
            
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
        
        # Save metrics directly to save_dir/{episode_id}.json
        # The caller is responsible for setting save_dir to the appropriate path
        # (e.g., metrics/{task}/{model_id}/{method}/)
        os.makedirs(self.save_dir, exist_ok=True)
        
        filename = f"{self.current_episode.episode_id}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        episode_dict = asdict(self.current_episode)
        
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
    
    def _extract_action_content(self, action) -> str:
        content_parts = []
        
        # Show Search Type first, then Query
        if hasattr(action, 'search_type') and action.search_type:
            content_parts.append(f"Search Type: {action.search_type}")
        if hasattr(action, 'query') and action.query:
            content_parts.append(f"Query: {action.query}")
        if hasattr(action, 'thought') and action.thought:
            content_parts.append(f"Thought: {action.thought}")
        if hasattr(action, 'response') and action.response:
            content_parts.append(f"Response: {action.response}")
        
        return "\n".join(content_parts) if content_parts else str(action)
    
    
    
# --- Episode logging ---


def log_section(title: str, char: str = "="):
    logger.info("\n" + char * 80)
    logger.info(title)
    logger.info(char * 80)


# =============================================================================
# INITIAL STATE LOGGING
# =============================================================================

def log_initial_state(agent: "Agent", environment: "Environment", observation, task_name: str):
    log_section(f"STARTING {task_name.upper()} EPISODE")
    
    log_section("INITIAL STATE")
    logger.info(f"Initial Observation: {observation.result}")
    logger.info(f"Available Actions: {[a.value for a in agent.action_space]}")
    logger.info(f"Max Steps: {environment.max_steps}")
    
    if not hasattr(agent, 'action_selection_prompt_constructor'):
        return
        
    log_section("AGENT CONFIGURATION")
    logger.info(f"Agent Type: {agent.__class__.__name__}")
    logger.info(f"Model: {agent.model_id}")
    logger.info(f"Temperature: {agent.temperature}")
    logger.info(f"Max Tokens Config: {getattr(agent, 'max_tokens_config', 'N/A')}")
    logger.info(f"Action Space: {[a.value for a in agent.action_space]}")
    logger.info(f"Thinking Enabled: {getattr(agent, 'thinking_enabled', 'N/A')}")
    logger.info(f"Open Web Search Enabled: {getattr(agent, 'open_web_search_enabled', 'N/A')}")


# =============================================================================
# STEP LOGGING
# =============================================================================

def _preview(result, limit: int) -> str:
    text = json.dumps(result) if isinstance(result, dict) else str(result)
    return text[:limit] + "..." if len(text) > limit else text


def log_step_header(step_num: int, observation):
    log_section(f"STEP {step_num}")
    logger.info(f"Current Observation: {_preview(observation.result, 300)}")


def log_action_basic(action) -> str:
    action_type = action.action_type.value
    logger.info(f"\nAction Selected: {action_type}")
    logger.info(f"  Full Action: {action}")
    logger.info(f"  Action Parameters: {action.get_input_parameters()}")
    return action_type


# =============================================================================
# ACTION-SPECIFIC LOGGING
# =============================================================================

def log_final_response(action, observation) -> tuple:
    final_response = getattr(action, 'response', '')
    is_correct = observation.metadata.get('is_correct') if observation.metadata else None
    accuracy = observation.metadata.get('accuracy', None) if observation.metadata else None
    
    logger.info(f"  Prediction: {final_response}")
    accuracy_str = f"{accuracy:.3f}" if accuracy is not None else "N/A"
    logger.info(f"  Accuracy: {accuracy_str}")
    logger.info(f"  Fully Correct: {is_correct}")
    
    return final_response, is_correct, accuracy


def log_think_action(action):
    thought = getattr(action, 'thought', '')
    thought_preview = thought[:500] + '...' if len(thought) > 500 else thought
    logger.info(f"  Thought: {thought_preview}")


def log_search_action(action, observation, action_type: str):
    query = getattr(action, 'query', '')
    num_results = observation.metadata.get('num_results', None)
    
    # Check for errors first and log them prominently
    error_msg = observation.metadata.get('error')
    if error_msg:
        logger.error(f"  ❌ SEARCH ERROR: {error_msg}")
        logger.info(f"  Query: {query}")
        return
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Number of Results: {num_results}")


def log_generic_observation(observation):
    logger.info(f"\n  Observation Result: {_preview(observation.result, 500)}")


def log_beliefs(agent: "Agent"):
    if not hasattr(agent, 'get_current_beliefs'):
        return
    
    current_beliefs = agent.get_current_beliefs()
    if not current_beliefs:
        return
    
    if isinstance(current_beliefs, dict):
        task_beliefs = current_beliefs.get('task_beliefs', 'N/A')[:500]
        design_beliefs = current_beliefs.get('design_beliefs', 'N/A')[:500]
        beliefs_str = f"Task: {task_beliefs}\nDesign: {design_beliefs}"
    else:
        beliefs_str = str(current_beliefs)
    
    logger.info(f"\n  Current Beliefs: {beliefs_str}")
