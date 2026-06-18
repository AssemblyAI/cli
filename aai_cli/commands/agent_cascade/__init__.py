from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.agent_cascade import voices
from aai_cli.agent_cascade.config import (
    DEFAULT_GREETING,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SPEECH_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)
from aai_cli.agent_cascade.voices import DEFAULT_VOICE
from aai_cli.app.context import AppState, run_command, run_with_options
from aai_cli.commands.agent_cascade import _exec as agent_cascade_exec
from aai_cli.core import choices, llm
from aai_cli.streaming.turn_presets import TurnDetectionPreset
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

# Option panels that group the per-leg knobs in `--help` instead of one flat wall.
_PANEL_STT = "Speech-to-text"
_PANEL_LLM = "Language model"
_PANEL_TTS = "Text-to-speech"

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=45,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("live",),
)


def _emit_voice_list(_state: AppState, json_mode: bool) -> None:
    """--list-voices body, routed through run_command so --json yields a machine-readable
    array instead of the human list; needs no auth."""
    payload = [{"name": name} for name in voices.VOICE_NAMES]
    output.emit(payload, lambda _voices: voices.format_voice_list(), json_mode=json_mode)


@app.command(
    name="live",
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Start a live voice conversation", "assembly --sandbox live"),
            (
                "Pick a voice and opening line",
                'assembly --sandbox live --voice michael --greeting "Hi there"',
            ),
            (
                "Give the agent a persona",
                'assembly --sandbox live --system-prompt "You are a terse pirate."',
            ),
            ("See available voices", "assembly --sandbox live --list-voices"),
            (
                "Print equivalent Python instead of running",
                "assembly --sandbox live --show-code",
            ),
        ]
    ),
)
def live(
    ctx: typer.Context,
    source: str | None = typer.Argument(
        None, help="Audio file path or URL to speak to the agent. Omit to use the microphone."
    ),
    sample: bool = typer.Option(
        False, "--sample", help="Speak the hosted wildfires.mp3 sample to the agent"
    ),
    voice: str = typer.Option(
        DEFAULT_VOICE,
        "--voice",
        help="TTS voice. See --list-voices.",
        autocompletion=voices.complete_voice,
        rich_help_panel=_PANEL_TTS,
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="TTS language (defaults to the voice's language)",
        rich_help_panel=_PANEL_TTS,
    ),
    tts_config: list[str] | None = typer.Option(
        None,
        "--tts-config",
        help="Set any extra streaming-TTS query field as KEY=VALUE (repeatable)",
        rich_help_panel=_PANEL_TTS,
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model that powers the agent's replies",
        autocompletion=llm.complete_model,
        rich_help_panel=_PANEL_LLM,
    ),
    max_tokens: int = typer.Option(
        DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens per reply",
        min=1,
        rich_help_panel=_PANEL_LLM,
    ),
    llm_config: list[str] | None = typer.Option(
        None,
        "--llm-config",
        help="Set any LLM Gateway request field as KEY=VALUE (repeatable)",
        rich_help_panel=_PANEL_LLM,
    ),
    speech_model: str = typer.Option(
        DEFAULT_SPEECH_MODEL,
        "--speech-model",
        help="Streaming speech model",
        rich_help_panel=_PANEL_STT,
    ),
    format_turns: bool = typer.Option(
        True,
        "--format-turns/--no-format-turns",
        help="Format (punctuate) finalized turns before replying",
        rich_help_panel=_PANEL_STT,
    ),
    turn_detection: TurnDetectionPreset | None = typer.Option(
        None,
        "--turn-detection",
        help="Turn-detection sensitivity preset",
        rich_help_panel=_PANEL_STT,
    ),
    stt_config: list[str] | None = typer.Option(
        None,
        "--stt-config",
        help="Set any StreamingParameters field as KEY=VALUE (repeatable)",
        rich_help_panel=_PANEL_STT,
    ),
    stt_config_file: Path | None = typer.Option(
        None,
        "--stt-config-file",
        help="JSON file of streaming fields",
        exists=True,
        dir_okay=False,
        rich_help_panel=_PANEL_STT,
    ),
    system_prompt: str = typer.Option(
        DEFAULT_SYSTEM_PROMPT, "--system-prompt", help="System prompt (the agent's persona)"
    ),
    system_prompt_file: Path | None = typer.Option(
        None,
        "--system-prompt-file",
        help="Read the system prompt from a file (overrides --system-prompt)",
        exists=True,
        dir_okay=False,
    ),
    greeting: str = typer.Option(DEFAULT_GREETING, "--greeting", help="Spoken greeting"),
    device: int | None = typer.Option(None, "--device", help="Microphone device index"),
    list_voices: bool = typer.Option(False, "--list-voices", help="Print known voices and exit"),
    json_out: bool = options.json_option("Emit newline-delimited JSON events"),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: text (you:/agent: lines as plain stdout, pipe-friendly) or json",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not start a session)",
    ),
) -> None:
    """\\[sandbox] Talk live to a tool-using voice agent

    A real-time spoken conversation, wired client-side from three primitives —
    Streaming STT, a deepagents brain on the LLM Gateway, and streaming TTS. Unlike
    'assembly agent' (the Voice Agent API), the brain here is an agent that can use
    tools mid-conversation — web search, URL fetch, and the AssemblyAI docs — so it
    answers like a live multimodal assistant. Because it uses streaming TTS it only
    runs in the sandbox: run it as 'assembly --sandbox live' (--sandbox goes before
    the subcommand).

    Use headphones: the mic stays open while the agent speaks, so on speakers it
    would hear itself and loop. Pass an audio file/URL (or --sample) to speak a
    recorded clip instead of the microphone; the session then ends after the
    agent's reply.

    This only runs a conversation in the terminal — it writes no code. To build
    an agent-cascade app, run 'assembly init agent-cascade' instead.

    Web search needs a TAVILY_API_KEY in the environment; without it the agent
    keeps its URL-fetch and docs tools.
    """

    if list_voices:
        run_command(ctx, _emit_voice_list, json=json_out)
        return

    opts = agent_cascade_exec.AgentCascadeOptions(
        source=source,
        sample=sample,
        voice=voice,
        model=model,
        system_prompt=system_prompt,
        system_prompt_file=system_prompt_file,
        greeting=greeting,
        device=device,
        output_field=output_field,
        speech_model=speech_model,
        format_turns=format_turns,
        turn_detection=turn_detection,
        stt_config=tuple(stt_config or ()),
        stt_config_file=stt_config_file,
        max_tokens=max_tokens,
        llm_config=tuple(llm_config or ()),
        language=language,
        tts_config=tuple(tts_config or ()),
        show_code=show_code,
    )
    run_with_options(ctx, agent_cascade_exec.run_agent_cascade, opts, json=json_out)
