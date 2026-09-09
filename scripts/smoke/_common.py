"""Bootstrap and action restrictions for the live smoke command."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def restrict_actions(env, allowed):
    """Advertise only allowed actions and reject others before dispatch."""
    allowed = frozenset(allowed)
    env.action_space = [action for action in env.action_space if action in allowed]
    env.initial_observation.metadata["available_actions"] = [action.value for action in env.action_space]
    original_step = env.step

    def step(action):
        if action is not None and action.action_type not in allowed:
            raise RuntimeError("Smoke run blocked an external/retrieval action")
        return original_step(action)

    env.step = step
    return env
