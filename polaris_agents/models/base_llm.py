from typing import Protocol, Optional

class LLMResponse:
    model_id: str
    completion: str
    stop_reason: str
    cost: Optional[float] = None
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    logprobs: Optional[list[dict[str, float]]] = None
    
class ModelAPIProtocol(Protocol):
    def __call__(self, model_id, prompt, max_attempts, **kwargs):
        raise NotImplementedError