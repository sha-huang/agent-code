from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ModelProvider, ModelResponse
from .tools import ToolContext, ToolRegistry


@dataclass
class AgentResult:
    final: str
    trace: list[str]
    messages: list[dict[str, Any]]


def _gemini_message(response: ModelResponse) -> dict[str, Any]:
    if response.assistant_content:
        return {"role": "model", "parts": response.assistant_content}
    
    parts: list[dict[str, Any]] = []

    if response.text:
        parts.append({"text": response.text})
    
    for fn_call in response.tool_calls or []:
        parts.append(
            {
                "function_call": {
                    "id": getattr(fn_call, "id", None),
                    "name": fn_call.name,
                    "args": fn_call.arguments,
                }
            }
        )
    
    return {"role": "model", "parts": parts}


def _tool_result_message(
    function_name: str,
    tool_call_id: str,
    content: Any,
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": function_name,
                    "id": tool_call_id,
                    "response": {
                        "result": content,
                        "is_error": is_error,
                    }
                }
            }
        ]
    }


def run_agent(
    prompt: str,
    provider: ModelProvider,
    tools: ToolRegistry,
    max_steps: int = 5,
    cwd: Path | None = None,
) -> AgentResult:

    ctx = ToolContext(cwd=cwd or Path.cwd())
    messages: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]
    trace: list[str] = []

    for step in range(max_steps):
        response = provider.complete(messages, tools=tools.list())
        messages.append(_gemini_message(response))

        if not response.tool_calls:
            final = response.text or ""
            trace.append(f"final: {final}")
            return AgentResult(final=final, trace=trace, messages=messages)
        
        for call in response.tool_calls:
            trace.append(f"tool_call: {call.name} {call.arguments}")
            result = tools.run(call, ctx)
            trace.append(f"observation: {result.content}")
            messages.append(_tool_result_message(result.name, result.tool_call_id, result.content, result.is_error))
    
    final = f"reached max_steps={max_steps}"
    trace.append(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)