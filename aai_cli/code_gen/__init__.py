from __future__ import annotations

from typing import TYPE_CHECKING

from aai_cli.code_gen import agent as _agent
from aai_cli.code_gen import agent_cascade as _agent_cascade
from aai_cli.code_gen import stream as _stream
from aai_cli.code_gen import transcribe as _transcribe

if TYPE_CHECKING:
    from aai_cli.agent_cascade.config import CascadeConfig


def gateway_options(
    prompts: list[str], model: str, max_tokens: int, *, interval: float = 0.0
) -> dict[str, object] | None:
    """The LLM-gateway options dict consumed by `transcribe`/`stream`, or None if no prompts.

    `interval` (streaming only) is the seconds between summary refreshes baked into the
    generated `stream --llm` loop; 0 refreshes on every turn. `transcribe` ignores it.
    """
    if not prompts:
        return None
    return {
        "prompts": list(prompts),
        "model": model,
        "max_tokens": max_tokens,
        "interval": interval,
    }


def agent(voice: str, system_prompt: str, greeting: str) -> str:
    """Generate runnable Python that reproduces this voice-agent session."""
    return _agent.render(voice, system_prompt, greeting)


def agent_cascade(config: CascadeConfig, *, speech_model: str) -> str:
    """Generate runnable Python that reproduces this terminal cascade session.

    Unlike `agent` (one Voice Agent socket), the cascade wires the three primitives
    itself — Streaming STT, the LLM Gateway, and streaming TTS — so the script mirrors
    the CLI's client-side orchestration. Sandbox hosts only, since streaming TTS has no
    production host.
    """
    return _agent_cascade.render(config, speech_model=speech_model)


def transcribe(
    merged: dict[str, object],
    source: str,
    *,
    llm_gateway: dict[str, object] | None = None,
    download_sections: list[str] | None = None,
) -> str:
    """Generate runnable Python that reproduces this transcribe invocation."""
    return _transcribe.render(
        merged, source, llm_gateway=llm_gateway, download_sections=download_sections
    )


def stream(
    merged: dict[str, object],
    *,
    llm: dict[str, object] | None = None,
    source: str | None = None,
) -> str:
    """Generate runnable Python that reproduces this streaming invocation.

    ``source`` mirrors the CLI argument: ``None`` streams the microphone, ``"-"``
    reads raw PCM16 from stdin, and anything else is a file path/URL decoded through
    ffmpeg — so the generated script reads the same input the real run would. With
    `llm` (a dict of ``prompts``/``model``/``max_tokens``/``interval``), the script
    refreshes a prompt-chain over the growing transcript every ``interval`` seconds (0 =
    every turn) — the live sibling of `transcribe --llm` — mirroring how `stream --llm` runs.
    """
    return _stream.render(merged, llm=llm, source=source)
