from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from rich.console import Console

from .model import ModelProvider, ModelResponse, ToolResult
from .tools import ToolContext, ToolRegistry
from .fs_safety import SkipPolicy, load_gitignore, resolve_in_cwd
from .diff_ui import confirm_edit, render_diff


console = Console()


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


def run_agent(
    prompt: str,
    provider: ModelProvider,
    tools: ToolRegistry,
    max_steps: int,
    cwd: Path | None = None,
) -> AgentResult:

    resolved_cwd = cwd or Path.cwd()
    ctx = ToolContext(
        cwd=resolved_cwd,
        skip_policy=SkipPolicy.default(gitignore=load_gitignore(resolved_cwd)),
    )

    def emit(line: str) -> None:
        trace.append(line)
        console.print(line)

    messages: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]
    trace: list[str] = []

    for step in range(max_steps):
        response = provider.complete(messages, tools=tools.list())
        messages.append(_gemini_message(response))

        if not response.tool_calls:
            final = response.text or ""
            emit(f"final: {final}")
            return AgentResult(final=final, trace=trace, messages=messages)

        tool_result_blocks: list[dict[str, Any]] = []
        for call in response.tool_calls:
            emit(f"tool_call: {call.name} {call.arguments}")

            if call.name in ("file_write", "file_edit"):
                path_str = call.arguments.get("file_path", "")

                try:
                    path = resolve_in_cwd(ctx.cwd, path_str)
                except (ValueError, OSError) as exc:
                    result = ToolResult(
                        name=call.name,
                        tool_call_id=call.id,
                        content=f"error: {exc}",
                        is_error=True,
                    )
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append(
                        {
                            "function_response": {
                                "name": result.name,
                                "id": result.tool_call_id,
                                "response": {
                                    "result": result.content,
                                    "is_error": result.is_error,
                                },
                            }
                        }
                    )
                    continue

                old_content = path.read_text(encoding="utf-8") if path.exists() else ""

                # if the file exists, ensure it has been read
                validation_error: str | None = None
                if call.name == "file_write" and path.exists():
                    if path not in ctx.read_state.entries:
                        validation_error = (
                            f"error: file has not been read yet. "
                            f"Read {path_str} first before editing."
                        )

                new_content: str | None = None
                if call.name == "file_write":
                    new_content = call.arguments.get("content", "")

                if validation_error is not None:
                    result = ToolResult(
                        name=call.name,
                        tool_call_id=call.id,
                        content=validation_error,
                        is_error=True,
                    )
                    emit(f"observation: {result.content}")
                    tool_result_blocks.append(
                        {
                            "function_response": {
                                "name": result.name,
                                "id": result.tool_call_id,
                                "response": {
                                    "result": result.content,
                                    "is_error": result.is_error,
                                },
                            }
                        }
                    )
                    continue

                if new_content is not None:
                    diff_text = render_diff(old_content, new_content, path_str)
                    console.print(f"\n[bold]Diff for {path_str}:[/bold]")
                    console.print(diff_text)
                    if not confirm_edit(path_str):
                        result = ToolResult(
                            name=call.name,
                            tool_call_id=call.id,
                            content="error: edit rejected by user",
                            is_error=True,
                        )
                        emit(f"observation: {result.content}")
                        tool_result_blocks.append(
                            {
                                "function_response": {
                                    "name": result.name,
                                    "id": result.tool_call_id,
                                    "response": {
                                        "result": result.content,
                                        "is_error": result.is_error,
                                    },
                                }
                            }
                        )
                        continue

            result = tools.run(call, ctx)
            emit(f"observation: {result.content}")
            tool_result_blocks.append(
                {
                    "function_response": {
                        "name": result.name,
                        "id": result.tool_call_id,
                        "response": {
                            "result": result.content,
                            "is_error": result.is_error,
                        },
                    }
                }
            )

        messages.append({"role": "user", "parts": tool_result_blocks})
    
    final = f"reached max_steps={max_steps}"
    emit(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)