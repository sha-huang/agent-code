from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

from datetime import datetime
from .model import ToolCall, ToolResult


ToolFunc = Callable[[dict[str, Any]], str]


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


def echo(args: dict[str, Any]) -> str:
    return str(args.get("text", ""))


def system_date(args: dict[str, Any]) -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )
        
        return ToolResult(tool_call_id=call.id, content=tool.run(call.arguments))


def default_tools() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="echo",
            description="Return the input text.",
            run=echo,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to return."}},
                "required": ["text"],
            },
        )
    )

    registry.register(
        Tool(
            name="system_date",
            description="Return the current system date and time.",
            run=system_date,
        )
    )
    
    return registry