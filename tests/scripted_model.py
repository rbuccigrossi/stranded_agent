"""A Strands model provider that replays canned turns.

Because it satisfies the same ``Model`` interface as the real providers, tests run
the genuine agent loop, tool executor, and approval interventions without a network
call or an API key.
"""

import json
from typing import Any, AsyncGenerator, Dict, List

from strands.models.model import Model


USAGE = {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
         "metrics": {"latencyMs": 1}}


def text_turn(text: str) -> List[Dict[str, Any]]:
    """A turn where the model just answers."""
    return [{"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {}}},
            {"contentBlockDelta": {"delta": {"text": text}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": USAGE}]


def tool_turn(name: str, tool_use_id: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A turn where the model calls one tool."""
    return [{"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(arguments)}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": USAGE}]


class ScriptedModel(Model):
    """Yields one prepared turn per model call, then answers "done"."""

    def __init__(self, *turns: List[Dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def update_config(self, **model_config: Any) -> None:
        return None

    def get_config(self) -> Dict[str, Any]:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError("the scripted model does not produce structured output")
        yield  # pragma: no cover - keeps this an async generator

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None,
                     **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        self.calls += 1
        for event in self.turns.pop(0) if self.turns else text_turn("done"):
            yield event
