from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from google import genai


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    assistant_content: list[dict[str, Any]] | None = None
    finish_reason: str = "STOP"


class ModelProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> ModelResponse:
        ...


def _to_gemini_tools(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


def _parse_tool_input(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__iter__"):
        return dict(value)
    return {}


def _content_part_to_dict(part: Any) -> dict[str, Any]:
    # Serialize a Gemini content block part into a standard library
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    if hasattr(part, "dict"):
        return part.dict(exclude_none=True)
    if hasattr(part, "to_dict"):
        return part.to_dict()
    
    data = {}

    gemini_fields = (
        "text",
        "inline_data",
        "file_data",
        "function_call",
        "function_response",
        "thought"
    )

    for name in gemini_fields:
        if hasattr(part, name) and (val := getattr(part, name)) is not None:
            # Recursively handle nested structures
            if hasattr(val, "model_dump"):
                data[name] = val.model_dump(exclude_none=True)
            elif hasattr(val, "dict"):
                data[name] = val.dict(exclude_none=True)
            elif hasattr(val, "to_dict"):
                data[name] = val.to_dict()
            elif hasattr(val, "args"):
                data[name] = {**val, "args": dict(val.args)} if hasattr(val.args, "__iter__") else val
            else:
                data[name] = val
    
    return data


class GeminiProvider:
    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        max_tokens: int = 128,
        base_url: str = "https://generativelanguage.googleapis.com",
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set the API KEY first.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.client = genai.Client(
            api_key=api_key,
            http_options={"base_url": base_url}
        )
    
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "contents": messages,
            "config": {"max_output_tokens": self.max_tokens},
        }

        if tools:
            kwargs["config"]["tools"] = [{"function_declarations": _to_gemini_tools(tools)}]
        
        response = self.client.models.generate_content(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_content: list[dict[str, Any]] = []

        for part in response.candidates[0].content.parts:
            assistant_content.append(_content_part_to_dict(part))

            # Text check
            if part.text:
                text_parts.append(part.text)
            
            # Tool call check
            elif part.function_call:
                fn_call = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=getattr(fn_call, "id", None),
                        name=fn_call.name,
                        arguments=_parse_tool_input(fn_call.args)
                    )
                )
        
        finish_reason = response.candidates[0].finish_reason.name

        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            assistant_content=assistant_content or None,
            finish_reason=finish_reason or "STOP",
        )


class MockProvider:
    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        last = messages[-1]

        if last["role"] == "user":
            # text = last["parts"].replace("use echo tool to say", "").strip() or last["parts"]
            text = last["parts"][0]["text"]
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_echo_1",
                        name="echo",
                        arguments={"text": text},
                    )
                ],
                finish_reason="STOP",
            )
        
        if last["role"] == "model":
            return ModelResponse(text=f"echo tool returns: ")
        
        return ModelResponse(text=f"I am only able to demonstrate echo tool")