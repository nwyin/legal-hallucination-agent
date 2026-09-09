"""Shared bootstrap and safety helpers for standalone smoke scripts."""

import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def block_network():
    """Reject socket connections and restore them even when a check fails."""

    def no_network(*args, **kwargs):
        raise RuntimeError("Network access forbidden during offline smoke checks")

    with (
        patch.object(socket.socket, "connect", no_network),
        patch.object(socket.socket, "connect_ex", no_network),
        patch.object(socket, "create_connection", no_network),
    ):
        yield


def restrict_actions(env, allowed):
    """Advertise only allowed actions and reject others before dispatch."""
    allowed = frozenset(allowed)
    env.action_space = [action for action in env.action_space if action in allowed]
    env.initial_observation.metadata["available_actions"] = [
        action.value for action in env.action_space
    ]
    original_step = env.step

    def step(action):
        if action is not None and action.action_type not in allowed:
            raise RuntimeError("Smoke run blocked an external/retrieval action")
        return original_step(action)

    env.step = step
    return env
