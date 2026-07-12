from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .model import MockProvider
from .agent import run_agent
from .tools import default_tools


console = Console()
app = typer.Typer(add_completion=False)


def render_header(cwd: Path) -> None:
    console.print("[bold]Agent Code[/bold]")
    console.print(f"[dim]cwd: {cwd}[/dim]\n")


def handle_slash(line: str) -> bool:
    # Slash commands are CLI commands, not passed to model
    if line == "/help":
        console.print("available commands: /help, /exit")
        return True
    return False


def run_once(prompt: str, cwd: Path) -> None:
    render_header(cwd)
    result = run_agent(prompt, MockProvider(), default_tools())
    for line in result.trace:
        console.print(line)


@app.callback(invoke_without_command=True)
def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
) -> None:
    # Resolve cwd on start once only
    resolved_cwd = cwd.resolve()
    text = prompt.strip()

    if text:
        run_once(text, resolved_cwd)
        return
    
    # Interactive loop below if no prompt after command
    render_header(resolved_cwd)
    console.print("/help for a list of commands, /exit to exit.")
    while True:
        line = typer.prompt(">").strip()
        if not line:
            continue
        if line == "/exit":
            console.print("Bye.")
            return
        if line.startswith("/") and handle_slash(line):
            continue
        run_once(line, resolved_cwd)


def main() -> None:
    app()