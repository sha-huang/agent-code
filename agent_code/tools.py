from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from datetime import datetime

from .fs_safety import (
    ReadFileState,
    SkipPolicy,
    ensure_text_file,
    ensure_within_size,
    resolve_in_cwd,
    should_skip,
    truncate_output,
)

from .model import ToolCall, ToolResult


@dataclass
class ToolContext:
    # Runtime context for tools
    cwd: Path
    skip_policy: SkipPolicy = field(default_factory=SkipPolicy.default)
    read_state: ReadFileState = field(default_factory=ReadFileState)


ToolFunc = Callable[[dict[str, Any], ToolContext], str]


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


def echo(args: dict[str, Any], ctx: ToolContext) -> str:
    return str(args.get("text", ""))


def system_date(args: dict[str, Any], ctx: ToolContext) -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "error: missing required argument 'path'"

    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
        ensure_text_file(path)
        ensure_within_size(path)
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        return f"error: {exc}"

    ctx.read_state.record(path, text)
    return truncate_output(text)


def list_files(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path", ".")

    try:
        base = resolve_in_cwd(ctx.cwd, path_str)
    except ValueError as exc:
        return f"error: {exc}"

    if not base.is_dir():
        return f"error: not a directory: {path_str}"

    entries: list[str] = []
    for child in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        rel = child.relative_to(ctx.cwd)
        if should_skip(rel, ctx.skip_policy):
            continue
        entries.append(f"{child.name}/" if child.is_dir() else child.name)

    return truncate_output("\n".join(entries) or "(empty)")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                name=call.name,
                tool_call_id=call.id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )
        
        return ToolResult(
            name=call.name,
            tool_call_id=call.id,
            content=tool.run(call.arguments, ctx)
        )


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

    registry.register(
        Tool(
            name="read_file",
            description="Read a text file inside the project. Path is relative to cwd.",
            run=read_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside cwd."},
                },
                "required": ["path"],
            },
        )
    )

    registry.register(
        Tool(
            name="list_files",
            description="List files and directories at a path inside cwd.",
            run=list_files,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path; defaults to '.'.",
                        "default": ".",
                    },
                },
                "required": [],
            },
        )
    )
    
    return registry