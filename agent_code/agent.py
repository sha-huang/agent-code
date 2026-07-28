from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .model import ModelProvider, ModelResponse
from .tools import ToolRegistry


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


def _tool_result_message(tool_call_id: str, content: Any, is_error: bool = False) -> dict[str, Any]:
    return {
        "role": "tool",
        "parts": [
            {
                "function_response": {
                    "name": tool_call_id,
                    "response": {
                        "result": content,
                        "is_error": is_error,
                    }
                }
            }
        ]
    }


def run_agent(prompt: str, provider: ModelProvider, tools: ToolRegistry) -> AgentResult:
    messages: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]
    trace: list[str] = []

    response = provider.complete(messages, tools=tools.list())
    messages.append(_gemini_message(response))

    for call in response.tool_calls or []:
        trace.append(f"tool_call: {call.name} {call.arguments}")

        result = tools.run(call)
        trace.append(f"observation: {result.content}")
        messages.append(_tool_result_message(result.tool_call_id, result.content, result.is_error))

        response = provider.complete(messages, tools=tools.list())
    
    final = response.text or ""
    trace.append(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)