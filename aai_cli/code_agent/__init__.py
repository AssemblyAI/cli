"""`assembly code` — a terminal coding agent built on the deepagents SDK.

A bespoke port of langchain-ai/deepagents' `code` agent, wired so it **only**
talks to the AssemblyAI LLM Gateway (an OpenAI-compatible endpoint reached via
`langchain_openai.ChatOpenAI`; see `model.py`). The agent gets deepagents'
built-in filesystem + shell tools — rooted at the working directory through a
`LocalShellBackend` — plus a custom `assembly` tool that invokes this very CLI,
so it can transcribe/stream/run-LLM as part of a coding task (`cli_tool.py`).

The pieces are split so the orchestration (`session.py`) is unit-tested against
a fake chat model driving the *real* deepagents graph, with no network: `agent.py`
builds the graph, `render.py` draws the conversation, and the Typer command in
`aai_cli/commands/code/` wires the gateway model + real CLI runner in.
"""

from __future__ import annotations
