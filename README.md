## Simple SWE Agent CLI

This project is a simple SWE Agent , inspired by this [SWE agent educational project](https://github.com/bozhouDev/14days-build-claude-code-cli).

Features:
- Minimal dependency: only command line, networking and OS libraries, no orchestration frameworks used
- Adapted for the Google Gemini generateContent API

To start:
- Add dependecies by `uv sync`
- Set API key
- Run by `uv run agent-code "Prompt to the agent"`

CLI flags:

| CLI flag | Details | Default value |
| -- | -- | -- |
| `--provider` | API provider, use `mock` for testing | `gemini` |
| `--model` | Model by the provider | `gemini-3.1-flash-lite` |
| `--base-url` | URL of LLM server | `None` |
| `--max-step` | Max number of steps in a multi-step agent loop | `10` |