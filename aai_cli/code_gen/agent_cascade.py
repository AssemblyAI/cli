from __future__ import annotations

import json
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from aai_cli.code_gen import agent_cascade_body
from aai_cli.core import environments

if TYPE_CHECKING:
    from aai_cli.agent_cascade.config import CascadeConfig

# The header carries only the injected constants and the reply-cue predicate, so it
# has no literal braces and is safe to fill with str.format. All the brace-heavy
# orchestration (dict/set literals, the protocol loops) lives in the static body,
# which is never formatted — so no brace has to be doubled.
_HEADER = """\
# Live voice cascade: Streaming STT -> LLM Gateway -> streaming TTS, wired client-side.
# This is what `assembly --sandbox agent-cascade` runs: it transcribes your speech,
# sends each finalized turn to the LLM Gateway, and speaks the reply through streaming
# TTS — the same three primitives the agent-cascade init template wires server-side.
# Requires audio + websockets:  pip install sounddevice websockets openai
# Tip: use headphones — the mic stays open while the agent speaks, so on speakers it
# would hear itself and loop.
import base64
import json
import os
import queue
import threading

import sounddevice as sd
from openai import OpenAI
from websockets.sync.client import connect

# Export your key first:  export ASSEMBLYAI_API_KEY="<your key>"
API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
STT_URL = {stt_url}
TTS_URL = {tts_url}
GATEWAY_URL = {gateway_url}
MODEL = {model}
MAX_TOKENS = {max_tokens}
MAX_HISTORY = {max_history}
SYSTEM_PROMPT = {system_prompt}
GREETING = {greeting}
RATE = 24000  # one full-duplex rate for mic capture + TTS playback (TTS native PCM16 mono)


def is_reply_cue(event):
    # The cue to generate a reply. {cue_comment}
    return {cue_expr}
"""


def _stt_url(speech_model: str, *, format_turns: bool) -> str:
    """The Streaming v3 socket URL for the active environment.

    The mic is captured and streamed at 24 kHz (the one full-duplex rate), so the
    sample_rate query param matches — a mismatch corrupts the audio server-side.
    """
    params = urlencode(
        {
            "sample_rate": 24000,
            "encoding": "pcm_s16le",
            "speech_model": speech_model,
            "format_turns": "true" if format_turns else "false",
        }
    )
    return f"wss://{environments.active().streaming_host}/v3/ws?{params}"


def _tts_url(voice: str, language: str | None) -> str:
    """The streaming-TTS socket URL for the configured voice (sandbox-only host)."""
    params: dict[str, str] = {"voice": voice, "sample_rate": "24000"}
    if language is not None:
        params["language"] = language
    return f"wss://{environments.active().streaming_tts_host}/v1/ws/?{urlencode(params)}"


def _cue(*, format_turns: bool) -> tuple[str, str]:
    """The (comment, predicate) for the reply trigger.

    With formatting on, wait for the *formatted* end-of-turn (better text for the LLM);
    with it off the server never sets turn_is_formatted, so a bare end-of-turn is the cue.
    """
    if format_turns:
        return (
            "With --format-turns, wait for the punctuated end-of-turn.",
            'bool(event.get("end_of_turn")) and bool(event.get("turn_is_formatted"))',
        )
    return (
        "With --no-format-turns the server never formats, so a bare end-of-turn is the cue.",
        'bool(event.get("end_of_turn"))',
    )


def render(config: CascadeConfig, *, speech_model: str) -> str:
    """Generate a runnable terminal cascade script from a cascade config + STT model.

    Hosts come from the active environment, so a sandbox run generates a script that
    targets the sandbox its key was minted for. The script mirrors the CLI run path:
    one full-duplex mic+speaker stream, one LLM completion per finalized turn, spoken
    sentence-by-sentence through a fresh TTS socket, with barge-in on the next turn.
    The named per-leg knobs are reflected; the --stt/--llm/--tts-config escape hatches
    (config.llm_extra / config.tts_extra) are not.
    """
    cue_comment, cue_expr = _cue(format_turns=config.format_turns)
    header = _HEADER.format(
        stt_url=json.dumps(_stt_url(speech_model, format_turns=config.format_turns)),
        tts_url=json.dumps(_tts_url(config.voice, config.language)),
        gateway_url=json.dumps(environments.active().llm_gateway_base),
        model=json.dumps(config.model),
        max_tokens=config.max_tokens,
        max_history=config.max_history,
        system_prompt=json.dumps(config.system_prompt),
        greeting=json.dumps(config.greeting),
        cue_comment=cue_comment,
        cue_expr=cue_expr,
    )
    return header + agent_cascade_body.BODY
