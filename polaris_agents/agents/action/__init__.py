# Action package - submodules for specific action types
# Import specific functions as needed to avoid circular imports

from .action_types import ActionType, Action, register_action_class, get_action_class, get_all_action_classes
from .internal.actions import ProvideFinalResponse, Think
from .courtlistener_search.actions import OpenCourtListenerSearch, AccessCourtListenerOpinion, CourtListenerCitationLookup, SearchLocalOpinion
from .open_search.actions import OpenWebSearch
from .filesystem.actions import ReadDocument, EditScratchpad
# TODO: Add other action types here
# TODO: Consider adding execute method to Action class to execute the action instead of having separate execute functions, like in smol-agents.

__all__ = [
    "ActionType",
    "Action",
    "ProvideFinalResponse",
    "Think",
    "OpenCourtListenerSearch",
    "AccessCourtListenerOpinion",
    "CourtListenerCitationLookup",
    "SearchLocalOpinion",
    "OpenWebSearch",
    "ReadDocument",
    "EditScratchpad",
    "register_action_class",
    "get_action_class",
    "get_all_action_classes"
]
