"""Direct tests of the `assembly stream` options/run seam (aai_cli.commands.stream._exec).

The command module only parses argv into a StreamOptions; everything after that is
run_stream, a plain function of data. These tests drive validation, flag mapping,
and session wiring by constructing options directly — no CliRunner argv round-trip,
no merged-stream output parsing.
"""

from __future__ import annotations

import dataclasses

import pytest

from aai_cli.app.context import AppState
from aai_cli.commands.stream import DEFAULT_SPEECH_MODEL
from aai_cli.commands.stream import _exec as stream_exec
from aai_cli.core import config, llm
from aai_cli.core.errors import UsageError

# The CLI's flag defaults, as data. Tests override per-case with dataclasses.replace.
DEFAULTS = stream_exec.StreamOptions(
    source=None,
    sample=False,
    sample_rate=None,
    device=None,
    system_audio=False,
    system_audio_only=False,
    speech_model=DEFAULT_SPEECH_MODEL,
    encoding=None,
    language_detection=None,
    domain=None,
    prompt=None,
    keyterms_prompt=None,
    end_of_turn_confidence_threshold=None,
    min_turn_silence=None,
    max_turn_silence=None,
    vad_threshold=None,
    format_turns=None,
    include_partial_turns=None,
    speaker_labels=None,
    max_speakers=None,
    voice_focus=None,
    voice_focus_threshold=None,
    inactivity_timeout=None,
    filter_profanity=None,
    redact_pii=None,
    redact_pii_policy=None,
    redact_pii_sub=None,
    webhook_url=None,
    webhook_auth_header=None,
    llm_prompt=None,
    llm_interval=10.0,
    model=llm.DEFAULT_MODEL,
    max_tokens=llm.DEFAULT_MAX_TOKENS,
    config_kv=None,
    config_file=None,
    output_field=None,
    show_code=False,
)


class FakeMic:
    """Mirrors MicrophoneSource's keyword signature (see microphone.py)."""

    def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
        self.sample_rate = capture_rate or 16000
        self.device = device

    def __iter__(self):
        return iter([b"\x00\x00"])


def test_run_stream_maps_flags_to_params_without_cli(monkeypatch):
    # The seam's payoff: assert the flag->StreamingParameters mapping by constructing
    # options directly, instead of threading a giant argv through CliRunner.
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        seen["api_key"] = api_key
        seen["params"] = params

    monkeypatch.setattr(stream_exec.client, "stream_audio", fake_stream_audio)
    monkeypatch.setattr(stream_exec, "MicrophoneSource", FakeMic)

    stream_exec.run_stream(
        dataclasses.replace(
            DEFAULTS,
            domain="medical-v1",
            prompt="expect drug names",
            keyterms_prompt=["AssemblyAI"],
        ),
        AppState(),
        json_mode=True,
    )
    assert seen["api_key"] == "sk_live"
    params = seen["params"]
    assert params.domain == "medical-v1"
    assert params.prompt == "expect drug names"
    assert params.keyterms_prompt == ["AssemblyAI"]


def test_run_stream_validates_before_resolving_credentials():
    # No API key is configured: a flag conflict must surface as a usage error, not
    # as NotAuthenticated — validation runs before any credential resolution.
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, system_audio=True, system_audio_only=True),
            AppState(),
            json_mode=False,
        )


def test_redact_pii_sub_enum_maps_to_its_string_value():
    # --redact-pii-sub is an SDK enum (validated at parse time), so base_flags must
    # unwrap it to the canonical string the streaming config expects, not pass the
    # enum member through.
    from assemblyai import PIISubstitutionPolicy

    opts = dataclasses.replace(DEFAULTS, redact_pii_sub=PIISubstitutionPolicy.hash)
    assert opts.base_flags()["redact_pii_sub"] == "hash"
    assert DEFAULTS.base_flags()["redact_pii_sub"] is None  # unset stays None


def test_stream_options_are_immutable():
    field_name = "sample"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DEFAULTS, field_name, True)
