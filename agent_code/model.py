from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    stop_reason: str = "end_turn"


class MockProvider:
    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        last = messages[-1]

        if last["role"] == "user":
            text = last["content"].replace("use echo tool to say", "").strip() or last["content"]
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_echo_1",
                        name="echo",
                        arguments={"text": text},
                    )
                ],
                stop_reason="tool_use",
            )
        
        if last["role"] == "tool":
            return ModelResponse(text=f"echo tool returns: {last["content"]}")
        
        return ModelResponse(text=f"I am only able to demonstrate echo tool")