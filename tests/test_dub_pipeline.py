"""Faked end-to-end runs of the `assembly dub` pipeline (aai_cli/dub_exec.py):
the transcribe → translate → synthesize → ffmpeg mux orchestration, voice
assignment, and the failure modes of each boundary. The LLM Gateway, streaming
TTS, and ffmpeg are faked at the modules dub_exec calls into (`llm.complete`,
`session.synthesize`, `client.transcribe`) and at `mediafile.run_ffmpeg`; the
pure helpers and validation order live in test_dub_exec.py."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_cli import client, llm, mediafile
from aai_cli.commands.dub import _exec as dub_exec
from aai_cli.context import AppState
from aai_cli.errors import APIError, CLIError
from aai_cli.tts import session
from aai_cli.tts.session import SpeakResult
from tests._clip_helpers import plain
from tests._dub_helpers import (
    DEFAULTS,
    SAMPLE_RATE,
    completion,
    enable_sandbox,
    fake_transcript,
    patch_api_key,
    record_ffmpeg,
    record_synthesize,
    record_transcribe,
    record_translate,
    utterance,
    write_media,
)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    return write_media(tmp_path)


@pytest.fixture(autouse=True)
def _sandbox_and_key(monkeypatch: pytest.MonkeyPatch):
    enable_sandbox(monkeypatch)
    patch_api_key(monkeypatch)


@pytest.fixture
def fake_transcribe(monkeypatch: pytest.MonkeyPatch):
    return record_transcribe(monkeypatch)


@pytest.fixture
def fake_translate(monkeypatch: pytest.MonkeyPatch):
    return record_translate(monkeypatch)


@pytest.fixture
def fake_synthesize(monkeypatch: pytest.MonkeyPatch):
    return record_synthesize(monkeypatch)


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    return record_ffmpeg(monkeypatch)


def _run(opts, *, json_mode):
    dub_exec.run_dub(opts, AppState(), json_mode=json_mode)


def test_run_dub_pipeline_end_to_end(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    _run(opts, json_mode=True)

    # Transcription: the local file, diarized so speakers keep distinct voices,
    # source language auto-detected (dub input is typically not English).
    assert fake_transcribe["audio"] == str(media)
    assert fake_transcribe["config"].speaker_labels is True
    assert fake_transcribe["config"].language_detection is True
    assert fake_transcribe["config"].language_code is None

    # Translation: one gateway call per utterance, in order, with the dubbing
    # system prompt naming the resolved language ("de" -> "German").
    assert [c["messages"][-1]["content"] for c in fake_translate] == ["Hello.", "World."]
    for call in fake_translate:
        assert call["model"] == llm.DEFAULT_MODEL
        assert call["max_tokens"] == llm.DEFAULT_MAX_TOKENS
        system = call["messages"][0]
        assert system["role"] == "system"
        assert "dubbing" in system["content"]
        assert "German" in system["content"]

    # Synthesis: the translated text in the target language, every speaker on
    # German's one native voice (the language selects the voice).
    assert [(cfg.voice, cfg.text) for cfg in fake_synthesize] == [
        ("juergen", "DE:Hello."),
        ("juergen", "DE:World."),
    ]
    assert all(cfg.language == "German" for cfg in fake_synthesize)

    # The dubbed track: silence to 1.0 s, segment 1, silence to 3.0 s, segment 2,
    # then a tail pad out to the source's 5 s duration (rate 100 -> 200 bytes/s).
    expected_track = b"\x00" * 200 + b"\xa1" * 100 + b"\x00" * 300 + b"\xa2" * 100 + b"\x00" * 300
    assert fake_ffmpeg["wav_frames"] == expected_track
    params = fake_ffmpeg["wav_params"]
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, SAMPLE_RATE)

    # The mux: video copied, WAV swapped in as the only audio, default out path.
    out = media.parent / "talk.dub.german.mp4"
    wav_path = fake_ffmpeg["args"][8]
    assert fake_ffmpeg["args"] == [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media),
        "-i",
        wav_path,
        "-map",
        "0:v?",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        str(out),
    ]
    assert wav_path.endswith("dub.wav")

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "source": str(media),
        "out": str(out),
        "language": "German",
        "transcript_id": "tr_dub",
        "utterances": 2,
        "speakers": {"A": "juergen", "B": "juergen"},
        "sample_rate": SAMPLE_RATE,
        "audio_duration_seconds": 5.0,
    }


def test_run_dub_human_summary(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    # A short --out keeps the one-line summary under the 80-column console width:
    # with the default (tmp_path-prefixed) out path, Rich would hard-wrap the line
    # mid-word and these substring asserts would depend on where the break lands.
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=Path("dub.de.mp4"))
    _run(opts, json_mode=False)
    # plain(): under FORCE_COLOR (CI) Rich's repr highlighter interleaves style
    # codes inside the line ("(2 utterances" renders with the 2 colored).
    out = plain(capsys.readouterr().out)
    assert "dub.de.mp4" in out
    assert "dubbed to German" in out
    assert "2 utterances" in out
    assert "A=juergen, B=juergen" in out


def test_human_summary_escapes_user_controlled_markup(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    # An unescaped "[/]" in --lang/--voice would raise rich.errors.MarkupError —
    # after the whole billed pipeline succeeded and the file was written.
    opts = dataclasses.replace(
        DEFAULTS, media=str(media), language="Ger[/]man", voice=["[/]bad"], out=Path("dub.x.mp4")
    )
    _run(opts, json_mode=False)
    out = plain(capsys.readouterr().out)
    assert "dubbed to Ger[/]man" in out
    assert "A=[/]bad" in out


def test_dash_prefixed_out_is_disambiguated_for_ffmpeg(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, monkeypatch, tmp_path
):
    # A bare "-dub.de.mp4" argv token would be parsed by ffmpeg as an option.
    monkeypatch.chdir(tmp_path)
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=Path("-dub.de.mp4"))
    _run(opts, json_mode=True)
    assert fake_ffmpeg["args"][-1] == "./-dub.de.mp4"


def test_run_dub_status_messages(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, monkeypatch
):
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(dub_exec.output, "status", fake_status)
    _run(dataclasses.replace(DEFAULTS, media=str(media)), json_mode=False)
    assert messages == [
        "Transcribing for dubbing…",
        f"Translating 2 utterance(s) to German with {llm.DEFAULT_MODEL}…",
        "Synthesizing 2 segment(s)…",
        "Writing the dubbed file…",
    ]


def test_bare_voice_dubs_every_speaker(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    opts = dataclasses.replace(DEFAULTS, media=str(media), voice=["paul"])
    _run(opts, json_mode=True)
    assert [cfg.voice for cfg in fake_synthesize] == ["paul", "paul"]


def test_voice_overrides_pin_speakers_without_consuming_rotation(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    opts = dataclasses.replace(DEFAULTS, media=str(media), voice=["A=mary"])
    _run(opts, json_mode=True)
    # A is pinned; B still takes German's native voice from the rotation.
    assert [cfg.voice for cfg in fake_synthesize] == ["mary", "juergen"]
    # Every mapping applied -> no "Ignoring" warning fires.
    assert "Ignoring" not in capsys.readouterr().err


def test_voice_pin_for_absent_speaker_warns_instead_of_silently_dropping(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    # Mirrors `assembly speak`: a requested --voice mapping is never dropped silently.
    opts = dataclasses.replace(DEFAULTS, media=str(media), voice=["Z=paul"])
    _run(opts, json_mode=False)
    err = plain(capsys.readouterr().err)
    assert "Ignoring --voice mapping(s) for speaker(s) not in the transcript: z." in err
    # Human mode warns as prose, not as a {"warning": …} JSON object.
    assert not err.lstrip().startswith("{")


def test_source_lang_pins_the_transcription_language(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    opts = dataclasses.replace(DEFAULTS, media=str(media), source_language="fr")
    _run(opts, json_mode=True)
    assert fake_transcribe["config"].language_code == "fr"
    assert fake_transcribe["config"].language_detection is None


def test_english_dub_keeps_the_multi_voice_rotation(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    # English has many voices, so speakers still rotate through the curated set
    # instead of collapsing onto one voice.
    opts = dataclasses.replace(DEFAULTS, media=str(media), language="en")
    _run(opts, json_mode=True)
    assert [cfg.voice for cfg in fake_synthesize] == ["jane", "michael"]


def test_language_without_a_native_voice_falls_back_to_english_rotation(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    # Japanese is translatable but has no catalog voice: the dub still runs,
    # on the English rotation.
    opts = dataclasses.replace(DEFAULTS, media=str(media), language="ja")
    _run(opts, json_mode=True)
    assert [cfg.voice for cfg in fake_synthesize] == ["jane", "michael"]
    assert all(cfg.language == "Japanese" for cfg in fake_synthesize)


def test_transcript_id_reuses_existing_transcript(
    media, fake_translate, fake_ffmpeg, monkeypatch, capsys
):
    fetched: dict[str, str] = {}

    def get_transcript(api_key, transcript_id):
        fetched["id"] = transcript_id
        return SimpleNamespace(
            id=transcript_id,
            utterances=[utterance(0, "A", "Hello.")],
            audio_duration=None,  # duration unknown -> no tail pad
        )

    monkeypatch.setattr(client, "get_transcript", get_transcript)
    monkeypatch.setattr(
        client,
        "transcribe",
        lambda *a, **k: pytest.fail("must not re-transcribe with --transcript-id"),
    )
    monkeypatch.setattr(
        session,
        "synthesize",
        lambda api_key, cfg, **_: SpeakResult(
            pcm=b"\xaa" * 2000, sample_rate=300, audio_duration_seconds=0.0
        ),
    )

    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_99")
    _run(opts, json_mode=True)
    assert fetched["id"] == "tr_99"
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "tr_99"
    # 1000 samples at 300 Hz, rounded to milliseconds: 3.3333... -> 3.333.
    assert payload["audio_duration_seconds"] == 3.333


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_transcript_id_still_in_flight_is_a_clear_error(media, fake_ffmpeg, monkeypatch, status):
    # Without the status check this would surface as a misleading "no utterances
    # to dub … pass one created with --speaker-labels".
    monkeypatch.setattr(
        client,
        "get_transcript",
        lambda *a: SimpleNamespace(id="tr_q", status=status, utterances=None),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_q")
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "transcript_not_ready"
    assert exc.value.exit_code == 2
    assert f"Transcript tr_q is still {status}" in exc.value.message
    assert "assembly transcripts get tr_q" in (exc.value.suggestion or "")


@pytest.mark.parametrize(
    ("stored_error", "expected"),
    [("Audio file unreadable", "Audio file unreadable"), (None, "Transcript failed.")],
    ids=["with-reason", "without-reason"],
)
def test_transcript_id_with_error_status_surfaces_the_real_error(
    media, fake_ffmpeg, monkeypatch, stored_error, expected
):
    monkeypatch.setattr(
        client,
        "get_transcript",
        lambda *a: SimpleNamespace(id="tr_e", status="error", error=stored_error, utterances=None),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_e")
    with pytest.raises(APIError) as exc:
        _run(opts, json_mode=False)
    assert expected in exc.value.message


@pytest.mark.parametrize("finish", ["length", "max_tokens"], ids=["openai", "anthropic"])
def test_truncated_translation_is_an_api_error(
    media, fake_transcribe, fake_synthesize, fake_ffmpeg, monkeypatch, finish
):
    # A reply clipped by max_tokens is non-empty but incomplete; dubbing it would
    # produce speech that stops mid-sentence with exit 0.
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: completion("Hallo, aber abgeschn", finish_reason=finish)
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(APIError) as exc:
        _run(opts, json_mode=False)
    assert "utterance 1" in exc.value.message
    assert f"cut off at --max-tokens ({llm.DEFAULT_MAX_TOKENS})" in exc.value.message
    assert "higher --max-tokens" in (exc.value.suggestion or "")


def test_empty_translation_is_an_api_error(media, fake_synthesize, fake_ffmpeg, monkeypatch):
    long_text = "a" * 50 + "TAIL!"
    transcript = fake_transcript([utterance(0, "A", "Hello."), utterance(1000, "B", long_text)])
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: transcript)
    replies = iter(["Hallo.", "   "])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: completion(next(replies)))

    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(APIError) as exc:
        _run(opts, json_mode=False)
    # The 1-based index and the (50-char) text preview pin which utterance failed.
    assert f"empty translation for utterance 2 ({'a' * 50!r})." in exc.value.message


def test_mixed_sample_rates_are_an_api_error(
    media, fake_transcribe, fake_translate, fake_ffmpeg, monkeypatch
):
    rates = iter([100, 200])
    monkeypatch.setattr(
        session,
        "synthesize",
        lambda api_key, cfg, **_: SpeakResult(
            pcm=b"\x01\x02", sample_rate=next(rates), audio_duration_seconds=0.0
        ),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(APIError) as exc:
        _run(opts, json_mode=False)
    assert "mixed sample rates ([100, 200])" in exc.value.message


def test_ffmpeg_failure_reports_last_stderr_line(
    media, fake_transcribe, fake_translate, fake_synthesize, monkeypatch
):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        mediafile,
        "run_ffmpeg",
        lambda args: subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="noise\nInvalid data found\n"
        ),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "dub_failed"
    assert "Could not write talk.dub.german.mp4" in exc.value.message
    # The last stderr line is the reason ffmpeg gives; earlier noise is dropped.
    assert "Invalid data found" in exc.value.message
    assert "noise" not in exc.value.message
    assert "readable audio/video file" in (exc.value.suggestion or "")


def test_ffmpeg_silent_failure_reports_exit_code(
    media, fake_transcribe, fake_translate, fake_synthesize, monkeypatch
):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        mediafile,
        "run_ffmpeg",
        lambda args: subprocess.CompletedProcess(args=args, returncode=3, stdout="", stderr=""),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert "ffmpeg exited with code 3" in exc.value.message
