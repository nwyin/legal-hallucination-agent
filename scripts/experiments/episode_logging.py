"""
Episode logging helpers for run_experiment.py.

Provides structured logging for agent episodes including:
- Initial state and agent configuration
- Step-by-step action logging
- Search result formatting
- Belief state logging
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polaris_agents.agents.base import Agent
    from polaris_agents.environments.base import Environment

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION HEADERS
# =============================================================================

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
    logger.info(f"Model: {agent.model_id} ({agent.provider})")
    logger.info(f"Temperature: {agent.temperature}")
    logger.info(f"Max Tokens Config: {getattr(agent, 'max_tokens_config', 'N/A')}")
    logger.info(f"Action Space: {[a.value for a in agent.action_space]}")
    logger.info(f"Thinking Enabled: {getattr(agent, 'thinking_enabled', 'N/A')}")
    logger.info(f"Closed Search Enabled: {getattr(agent, 'closed_search_enabled', 'N/A')}")
    logger.info(f"Open Web Search Enabled: {getattr(agent, 'open_web_search_enabled', 'N/A')}")
    
    log_first_step_prompts(agent, environment, observation)


def log_first_step_prompts(agent: "Agent", environment: "Environment", observation):
    if environment.max_steps == 0:
        return

    # Action selection prompt
    try:
        action_space = agent.action_space
        env_desc = getattr(environment, 'get_environment_description', lambda: str(environment))()
        search_capabilities = getattr(environment, 'get_search_capabilities', lambda: "")()
        
        system_prompt = agent.action_selection_prompt_constructor.get_system_prompt(
            action_space=action_space,
            environment_description=env_desc,
            search_capabilities=search_capabilities
        )
        user_prompt = agent.action_selection_prompt_constructor.get_user_prompt(
            observation=observation,
            history=[],
            current_beliefs=getattr(agent, 'get_current_beliefs', lambda: "No beliefs yet")(),
            max_steps=environment.max_steps,
            task_instance_description=getattr(environment, 'question', '')
        )
        
        log_section("ACTION SELECTION PROMPT (First Step)", "-")
        logger.info(f"System Prompt:\n{system_prompt[:1000]}...\n[truncated]")
        logger.info(f"\nUser Prompt:\n{user_prompt[:1000]}...\n[truncated]")
    except Exception as e:
        logger.warning(f"Could not log action selection prompt: {e}")
    
    # Belief update prompt (not used when max_steps=0)
    if hasattr(agent, 'belief_update_prompt_constructor') and environment.max_steps != 0:
        try:
            env_desc = getattr(environment, 'get_environment_description', lambda: str(environment))()
            belief_system = agent.belief_update_prompt_constructor.get_system_prompt(
                environment_description=env_desc
            )
            log_section("BELIEF UPDATE PROMPT (First Step)", "-")
            logger.info(f"System Prompt:\n{belief_system[:1000]}...\n[truncated]")
        except Exception as e:
            logger.warning(f"Could not log belief update prompt: {e}")


# =============================================================================
# STEP LOGGING
# =============================================================================

def log_step_header(step_num: int, observation):
    log_section(f"STEP {step_num}")
    result = observation.result
    if isinstance(result, str):
        obs_preview = result[:300] + '...' if len(result) > 300 else result
    elif isinstance(result, dict):
        import json
        obs_preview = json.dumps(result)[:300] + '...' if len(json.dumps(result)) > 300 else json.dumps(result)
    else:
        obs_preview = str(result)[:300] + '...' if len(str(result)) > 300 else str(result)
    logger.info(f"Current Observation: {obs_preview}")


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
    if action_type == "CLOSED_SEARCH":
        search_type = getattr(action, 'search_type', '')
    elif action_type == "OPEN_COURTLISTENER_SEARCH":
        search_type = getattr(action, 'search_type', 'opinions')
    else:
        search_type = 'web'
    k = getattr(action, 'k', None)
    num_results = observation.metadata.get('num_results', None)
    
    # Check for errors first and log them prominently
    error_msg = observation.metadata.get('error')
    if error_msg:
        logger.error(f"  ❌ SEARCH ERROR: {error_msg}")
        logger.info(f"  Query: {query}")
        if action_type == "CLOSED_SEARCH":
            logger.info(f"  Search Type: {search_type}")
        return
    
    logger.info(f"  Query: {query}")
    if action_type == "CLOSED_SEARCH":
        logger.info(f"  Search Type: {search_type}")
        if k is not None:
            logger.info(f"  k: {k}")
    logger.info(f"  Number of Results: {num_results}")
    
    # Log inferred/applied filters
    if action_type == "CLOSED_SEARCH":
        _log_applied_filters(observation)
        _log_closed_search_results(observation, query, search_type, num_results)
    elif action_type == "OPEN_WEB_SEARCH":
        _log_web_search_results(observation)


def log_generic_observation(observation):
    result = observation.result
    if isinstance(result, str):
        obs_preview = result[:500] + '...' if len(result) > 500 else result
    elif isinstance(result, dict):
        import json
        obs_preview = json.dumps(result)[:500] + '...' if len(json.dumps(result)) > 500 else json.dumps(result)
    else:
        obs_preview = str(result)[:500] + '...' if len(str(result)) > 500 else str(result)
    logger.info(f"\n  Observation Result: {obs_preview}")


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


# =============================================================================
# SEARCH RESULT HELPERS (internal)
# =============================================================================

def _log_applied_filters(observation):
    date_filter = observation.metadata.get('date_filter')
    field_filters = observation.metadata.get('field_filters')
    if date_filter or field_filters:
        log_parts = []
        if date_filter:
            log_parts.append(f"date_filter={date_filter}")
        if field_filters:
            log_parts.append(f"field_filters={field_filters}")
        logger.info(f"  Inferred/Applied: {', '.join(log_parts)}")


def _log_closed_search_results(observation, query: str, search_type: str, num_results):
    search_results = observation.metadata.get('search_results', [])
    
    logger.info(f"\n  Search Details:")
    logger.info(f"    Query: '{query}'")
    logger.info(f"    Search Type: {search_type}")
    logger.info(f"    Number of Results: {num_results}")
    
    # Log applied field filters from first result
    if search_results and hasattr(search_results[0], 'metadata'):
        first_meta = search_results[0].metadata or {}
        auto_filters = first_meta.get('auto_metadata_filters')
        if auto_filters:
            logger.info(f"    Applied Field Filters: {auto_filters}")
    
    if not search_results:
        logger.warning(f"  ⚠️ No results returned for CLOSED_SEARCH query: '{query}' (search_type: {search_type})")
        return
    
    logger.info(f"  Closed Search Results ({search_type}):")
    for i, result in enumerate(search_results[:3]):
        _log_single_search_result(i, result)


def _log_single_search_result(index: int, result):
    # Extract fields from SearchResult object or dict
    if hasattr(result, 'metadata'):
        result_id = getattr(result, 'result_id', 'N/A')
        title = getattr(result, 'title', 'N/A')
        contents = result.metadata.get('contents', '') if result.metadata else ''
        score = result.metadata.get('score', 'N/A') if result.metadata else 'N/A'
        link_name = result.metadata.get('link_name', 'N/A') if result.metadata else 'N/A'
        proceeding_title = result.metadata.get('proceeding_title', 'N/A') if result.metadata else 'N/A'
        published_date = result.metadata.get('published_epoch_seconds', 'N/A') if result.metadata else 'N/A'
    elif isinstance(result, dict):
        result_id = result.get('result_id', 'N/A')
        title = result.get('title', 'N/A')
        if 'metadata' in result and isinstance(result['metadata'], dict):
            contents = result['metadata'].get('contents', '')
            score = result['metadata'].get('score', 'N/A')
            link_name = result['metadata'].get('link_name', 'N/A')
            proceeding_title = result['metadata'].get('proceeding_title', 'N/A')
            published_date = result['metadata'].get('published_epoch_seconds', 'N/A')
        else:
            contents = result.get('contents', '')
            score = result.get('score', 'N/A')
            link_name = 'N/A'
            proceeding_title = 'N/A'
            published_date = 'N/A'
    else:
        result_id, title, contents, score = 'N/A', 'N/A', '', 'N/A'
        link_name, proceeding_title, published_date = 'N/A', 'N/A', 'N/A'
    
    truncated_contents = contents[:300] + "..." if len(contents) > 300 else contents
    title_preview = title[:100] + '...' if len(title) > 100 else title
    proc_preview = str(proceeding_title)[:80] + '...' if len(str(proceeding_title)) > 80 else proceeding_title
    
    logger.info(f"\n    Result {index + 1}:")
    logger.info(f"      ID: {result_id}")
    logger.info(f"      Title: {title_preview}")
    logger.info(f"      Score: {score}")
    logger.info(f"      Link Name: {link_name}")
    logger.info(f"      Proceeding Title: {proc_preview}")
    logger.info(f"      Published Date: {published_date}")
    if contents:
        logger.info(f"      Contents Preview: {truncated_contents}")


def _log_web_search_results(observation):
    web_results = observation.metadata.get('web_search_results', [])
    if not web_results:
        return
    
    logger.info(f"  Web Search Results:")
    for i, result in enumerate(web_results[:3]):
        if hasattr(result, 'snippet'):
            snippet = result.snippet
            result_id = getattr(result, 'result_id', 'N/A')
            title = getattr(result, 'title', 'N/A')
            url = getattr(result, 'url', 'N/A')
        else:
            snippet = result.get('snippet', '') if isinstance(result, dict) else ''
            result_id = result.get('result_id', 'N/A') if isinstance(result, dict) else 'N/A'
            title = result.get('title', 'N/A') if isinstance(result, dict) else 'N/A'
            url = result.get('url', 'N/A') if isinstance(result, dict) else 'N/A'
        
        truncated_snippet = snippet[:300] + "..." if len(snippet) > 300 else snippet
        logger.info(f"    {i + 1}. {result_id}: {title}")
        logger.info(f"       URL: {url}")
        logger.info(f"       Snippet: {truncated_snippet}")
        logger.info("")
