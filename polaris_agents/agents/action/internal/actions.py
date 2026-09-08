"""
Internal actions for the agent system.

This module implements the PROVIDE_FINAL_RESPONSE and THINK actions that are used
for internal agent reasoning and final response generation.
"""

from polaris_agents.agents.action.action_types import Action, ActionType


class ProvideFinalResponse(Action):
    """
    Internal action for providing the final response to a given task example.
    
    This action is used when the agent has completed its exploration and
    wants to provide a final response.
    """
    
    action_type = ActionType.PROVIDE_FINAL_RESPONSE
    description = "Provide the final response to the given task example. Use this when you have completed your exploration and want to provide a final response."
    inputs = {
        "response": {
            "type": "string",
            "description": "The final response to the given task example",
            "required": True
        }
    }


    def __init__(self, response: str):
        super().__init__(response=response)
        self.response = response


class Think(Action):
    """
    Internal action for agent reasoning and thought process.
    
    This action allows the agent to think,
    showing its reasoning process before taking other actions.
    """
    
    action_type = ActionType.THINK
    description = "Thinking. Use this to reason about your history and current state, plan your next steps, or analyze information before taking action. Use only briefly to synthesize or transition between actions. Do not use repeatedly for planning; proceed to search actions instead."
    inputs = {
        "thought": {
            "type": "string", 
            "description": "Your reasoning, analysis, or thought process about the current situation",
            "required": True
        }
    }

    def __init__(self, thought: str):
        super().__init__(thought=thought)
        self.thought = thought
