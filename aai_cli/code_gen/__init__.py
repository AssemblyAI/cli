from __future__ import annotations

from aai_cli.code_gen import agent as _agent
from aai_cli.code_gen import stream as _stream
from aai_cli.code_gen import transcribe as _transcribe


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


def transcribe(
    merged: dict[str, object],
    source: str,
    *,
    llm_gateway: dict[str, object] | None = None,
) -> str:
    """Generate runnable Python that reproduces this transcribe invocation."""
    return _transcribe.render(merged, source, llm_gateway=llm_gateway)


def stream(
    merged: dict[str, object],
    *,
    llm: dict[str, object] | None = None,
) -> str:
    """Generate runnable Python that reproduces this streaming invocation.

    With `llm` (a dict of ``prompts``/``model``/``max_tokens``/``interval``), the script
    refreshes a prompt-chain over the growing transcript every ``interval`` seconds (0 =
    every turn) — the live sibling of `transcribe --llm` — mirroring how `stream --llm` runs.
    """
    return _stream.render(merged, llm=llm)
