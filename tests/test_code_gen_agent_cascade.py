"""Example-based code_gen tests for the agent-cascade scaffold.

The cascade wires three primitives client-side (Streaming STT -> LLM Gateway ->
streaming TTS), so the generated script is checked for all three legs plus the
session knobs it must inject. Sandbox hosts only — streaming TTS has no prod host.
"""

from __future__ import annotations

import ast
import dataclasses
import json

import pytest

from aai_cli import code_gen
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.core import environments


@pytest.fixture(autouse=True)
def _sandbox_env():
    # The cascade is sandbox-only (streaming TTS has no prod host), so generate against it.
    environments.set_active(environments.get("sandbox000"))


def _render(*, speech_model="u3-rt-pro", **overrides):
    config = dataclasses.replace(
        CascadeConfig(
            voice="jane",
            system_prompt="Be terse.",
            greeting="Hi there",
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            max_history=40,
        ),
        **overrides,
    )
    return code_gen.agent_cascade(config, speech_model=speech_model)


def test_render_parses_and_wires_all_three_legs():
    code = _render()
    ast.parse(code)
    sandbox = environments.get("sandbox000")
    # STT, LLM Gateway, and TTS hosts all come from the active (sandbox) environment.
    assert f"wss://{sandbox.streaming_host}/v3/ws" in code
    assert f"wss://{sandbox.streaming_tts_host}/v1/ws/" in code
    assert sandbox.llm_gateway_base in code
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code


def test_render_injects_session_knobs():
    code = _render(model="claude-x", max_tokens=321, max_history=12)
    ast.parse(code)
    assert "voice=jane" in code  # the voice rides the TTS URL
    assert "Be terse." in code  # the system prompt
    assert "Hi there" in code  # the greeting
    assert '"claude-x"' in code  # the LLM model
    assert "MAX_TOKENS = 321" in code
    assert "MAX_HISTORY = 12" in code


def test_render_streams_stt_at_the_full_duplex_rate():
    # One full-duplex stream means the STT sample_rate must match the 24 kHz capture rate;
    # a mismatch corrupts the audio server-side. Pin the STT URL's own sample_rate (not just
    # any "sample_rate=24000", which the TTS URL also carries) so a drift can't slip through.
    code = _render()
    ast.parse(code)
    assert "/v3/ws?sample_rate=24000&encoding=pcm_s16le" in code
    assert "RATE = 24000" in code


def test_render_format_turns_waits_for_the_formatted_turn():
    code = _render(format_turns=True)
    ast.parse(code)
    assert "format_turns=true" in code
    assert "turn_is_formatted" in code  # the reply cue waits for the punctuated turn


def test_render_no_format_turns_fires_on_bare_end_of_turn():
    code = _render(format_turns=False)
    ast.parse(code)
    assert "format_turns=false" in code
    # The server never formats, so a bare end-of-turn is the cue (no turn_is_formatted gate).
    assert "turn_is_formatted" not in code


def test_render_includes_language_only_when_set():
    assert "language=" not in _render(language=None)
    assert "language=de" in _render(language="de")


def test_render_uses_single_full_duplex_stream():
    # ONE sd.RawStream (mic + speaker); two separate streams fail on macOS CoreAudio.
    code = _render()
    ast.parse(code)
    assert "sd.RawStream(" in code
    assert "RawInputStream" not in code
    assert "RawOutputStream" not in code


def test_render_escapes_quotes_in_prompt():
    tricky = 'Say "hi"\nand stop'
    code = _render(system_prompt=tricky)
    ast.parse(code)  # valid Python despite embedded quotes/newlines
    assert json.dumps(tricky) in code  # injected via json.dumps, escaped form appears verbatim
