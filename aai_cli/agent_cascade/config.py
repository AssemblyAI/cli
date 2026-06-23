"""Per-run configuration for the terminal voice cascade.

Defaults mirror the ``agent-cascade`` ``assembly init`` template's
``api/settings.py`` so the CLI conversation and the scaffolded app behave the
same out of the box.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from aai_cli.agent_cascade.voices import DEFAULT_VOICE
from aai_cli.core import llm

# `assembly live` defaults to a fast, low-latency gateway model (override with --model) —
# a literal rather than llm.DEFAULT_MODEL so the live agent's default is independent of the
# one-shot `assembly llm` default. Latency matters most for a spoken back-and-forth.
DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_MAX_TOKENS = llm.DEFAULT_MAX_TOKENS
# The realtime model the cascade transcribes with (same as the agent-cascade template).
DEFAULT_SPEECH_MODEL = "universal-3-5-pro"
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. Keep replies as short as "
    "possible — usually a single sentence, never more than two. Answer directly "
    "without preamble or filler. Your reply is read aloud by a text-to-speech "
    "engine, so write plain spoken prose — no markdown, emoji, bullet lists, or code."
)
DEFAULT_GREETING = "Hi! I'm your AssemblyAI voice agent. What can I help you with?"
# Sliding-window size: keep the last N messages of conversation as LLM context.
DEFAULT_MAX_HISTORY = 40
# Per-turn cap on how many tool calls the deepagents brain may make before it must answer.
# Enforced by a ToolCallLimitMiddleware with exit_behavior="continue": once the budget is hit,
# further tool calls are blocked and the model is forced to answer with what it has gathered —
# a graceful stop, never a GraphRecursionError. (langgraph's own recursion_limit stays at the
# deepagents default as a far-off safety backstop; this middleware is the real, soft cap.)
DEFAULT_TOOL_CALL_LIMIT = 10


@dataclass(frozen=True)
class CascadeConfig:
    """The static knobs for one cascade run, fixed once the flags are parsed."""

    voice: str = DEFAULT_VOICE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    greeting: str = DEFAULT_GREETING
    model: str = DEFAULT_MODEL
    max_history: int = DEFAULT_MAX_HISTORY
    # Per-turn tool-call budget: after this many tool calls the brain is forced to answer with
    # what it has (a graceful stop), rather than looping until langgraph's recursion backstop errors.
    tool_call_limit: int = DEFAULT_TOOL_CALL_LIMIT
    # TTS language (None lets the server pick from the voice).
    language: str | None = None
    # LLM: cap per-reply tokens and pass through any extra gateway request fields.
    max_tokens: int = DEFAULT_MAX_TOKENS
    llm_extra: Mapping[str, object] = field(default_factory=dict[str, object])
    # Extra streaming-TTS query params (the --tts-config escape hatch).
    tts_extra: Mapping[str, str] = field(default_factory=dict[str, str])
    # MCP servers (name -> launch spec) whose tools the deepagents brain can call. Empty
    # here by default; the live command populates it with the curated default set plus any
    # --mcp-config files.
    mcp_servers: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict[str, Mapping[str, object]]
    )
    # Whether STT formats finalized turns. The reply trigger waits for the formatted
    # turn when on; with it off, an unformatted end-of-turn is the cue instead.
    format_turns: bool = True
    # Opt-in: let the agent read/write files in the launch directory. Off by default keeps
    # behavior unchanged (the default in-memory backend, no gating, nothing advertised); on
    # swaps to a real-cwd FilesystemBackend and gates writes behind human approval.
    files: bool = False
    # The launch directory's AGENTS.md/CLAUDE.md, read into the system prompt so the agent
    # answers grounded in the project it's run from (None when no instruction file is present).
    project_context: str | None = None
