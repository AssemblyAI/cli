"""End-to-end tests that drive the real `aai` CLI against the live AssemblyAI API.

Speech is synthesized locally with kokoro TTS, then fed through the CLI as a
subprocess so the binary, argument parsing, auth, audio decoding, and network
path are all exercised for real — no mocks. Batch tests reuse the hosted
``--sample`` clip (wildfires.mp3) so they need no TTS.

These tests are marked `e2e` and skip (never fail) when the API key, kokoro, or
numpy is unavailable, so CI and keyless contributors are not blocked. The
precommit `pytest-e2e` hook runs them; the default unit run excludes them.

Coverage: batch transcribe (plain + summarization, auto-chapters, sentiment,
diarization, LLM transform), live streaming, the voice agent, the LLM command,
the transcripts list/get roundtrip, `doctor`, and the auth-failure path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
import wave
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

KOKORO_RATE = 24000  # kokoro emits 24 kHz float32 mono
STREAM_RATE = 16000  # what the CLI's fast WAV path expects (16 kHz mono PCM16)

# Stable content words from the hosted wildfires.mp3 sample transcript.
SAMPLE_WORDS = ("smoke", "wildfires", "canada")


@pytest.fixture(scope="session")
def kokoro_pipeline() -> Any:
    """Build the kokoro TTS pipeline once per session, or skip if unavailable.

    kokoro/torch emit benign UserWarnings (e.g. torch's single-layer dropout
    note, HF Hub unauthenticated-rate-limit note) on import and model build. The
    project runs pytest with ``filterwarnings = error``, so they are suppressed
    here; warnings from the CLI itself surface in the subprocess, not this filter.
    """
    pytest.importorskip("numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kokoro = pytest.importorskip("kokoro")
        return kokoro.KPipeline(lang_code="a")  # American English


def _synthesize_wav(pipeline: Any, text: str, path: Path, *, lead_silence_s: float = 0.6) -> Path:
    """Synthesize `text` to a 16 kHz mono PCM16 WAV the CLI can stream directly.

    Resamples kokoro's 24 kHz output to 16 kHz (linear) and prepends a short
    silence so nothing is clipped before the realtime session is ready.
    """
    import numpy as np

    chunks = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _gs, _ps, audio in pipeline(text, voice="af_heart"):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
    samples = np.concatenate(chunks)

    n_dst = round(len(samples) * STREAM_RATE / KOKORO_RATE)
    resampled = np.interp(
        np.linspace(0.0, len(samples) - 1, n_dst),
        np.arange(len(samples)),
        samples,
    )
    pcm = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2")
    silence = np.zeros(int(lead_silence_s * STREAM_RATE), dtype="<i2")
    pcm = np.concatenate([silence, pcm])

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(STREAM_RATE)
        w.writeframes(pcm.tobytes())
    return path


def _run_cli(args: list[str], key: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run `python -m aai_cli <args>` against the working tree with the real key."""
    env = dict(os.environ)
    env["ASSEMBLYAI_API_KEY"] = key
    return subprocess.run(
        [sys.executable, "-m", "aai_cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _ndjson(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def _transcribe_sample(key: str, *flags: str, timeout: int = 180) -> dict[str, Any]:
    """Transcribe the hosted sample with `flags`, asserting success, return JSON."""
    proc = _run_cli(["transcribe", "--sample", *flags, "--json"], key, timeout=timeout)
    assert proc.returncode == 0, f"args={flags} stderr:\n{proc.stderr}"
    return json.loads(proc.stdout)  # type: ignore[no-any-return]


# --- Batch transcription --------------------------------------------------


def test_transcribe_sample_basic(real_api_key):
    data = _transcribe_sample(real_api_key)
    assert data["status"] == "completed", data
    text = data["text"].lower()
    assert text.strip(), f"no transcript text: {data!r}"
    for word in SAMPLE_WORDS:
        assert word in text, f"{word!r} missing from transcript: {text[:200]!r}"
    assert data["words"], "expected word-level timestamps"
    assert data["audio_duration"] > 0


def test_transcribe_summarization(real_api_key):
    # summarization and auto_chapters are mutually exclusive on the API, so each
    # analysis feature gets its own run.
    data = _transcribe_sample(real_api_key, "--summarization")
    assert data["summary"] and data["summary"].strip(), f"no summary: {data!r}"


def test_transcribe_auto_chapters(real_api_key):
    data = _transcribe_sample(real_api_key, "--auto-chapters")
    chapters = data["chapters"]
    assert chapters, f"no chapters: {data!r}"
    assert all(c.get("summary") for c in chapters), f"chapter missing summary: {chapters!r}"


def test_transcribe_sentiment_analysis(real_api_key):
    data = _transcribe_sample(real_api_key, "--sentiment-analysis")
    results = data["sentiment_analysis_results"]
    assert results, f"no sentiment results: {data!r}"
    sentiments = {r["sentiment"] for r in results}
    assert sentiments <= {"POSITIVE", "NEGATIVE", "NEUTRAL"}, sentiments


def test_transcribe_speaker_labels(real_api_key):
    data = _transcribe_sample(real_api_key, "--speaker-labels")
    utterances = data["utterances"]
    assert utterances, f"no utterances: {data!r}"
    assert all(u.get("text") for u in utterances), "utterance missing text"
    assert all(u.get("speaker") for u in utterances), "utterance missing speaker label"


def test_transcribe_prompt_transforms_via_gateway(real_api_key):
    data = _transcribe_sample(
        real_api_key, "--llm", "Summarize this transcript in one short sentence."
    )
    assert data["text"].strip(), f"no transcript produced: {data!r}"
    # The LLM transform is {model, steps:[{prompt, output}, ...]}.
    steps = data["transform"]["steps"]
    assert steps, f"gateway returned no transform steps: {data!r}"
    assert steps[0]["output"].strip(), f"transform step had no output: {data!r}"


# --- Streaming ------------------------------------------------------------


def test_stream_file_transcribes_spoken_text(real_api_key, kokoro_pipeline, tmp_path):
    spoken = "the quick brown fox jumps over the lazy dog"
    wav = _synthesize_wav(kokoro_pipeline, spoken, tmp_path / "fox.wav")

    proc = _run_cli(["stream", str(wav), "--json"], real_api_key)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"

    events = _ndjson(proc.stdout)
    transcript = " ".join(
        e.get("transcript", "") for e in events if e.get("type") == "turn"
    ).lower()
    assert transcript.strip(), f"no transcript produced; events={events}"
    for word in ("fox", "lazy", "dog"):
        assert word in transcript, f"{word!r} missing from streamed transcript: {transcript!r}"


def test_stream_prompt_transforms_live(real_api_key, kokoro_pipeline, tmp_path):
    spoken = "the quick brown fox jumps over the lazy dog"
    wav = _synthesize_wav(kokoro_pipeline, spoken, tmp_path / "fox.wav")

    proc = _run_cli(
        ["stream", str(wav), "--llm", "Summarize the transcript in one short sentence.", "--json"],
        real_api_key,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    events = _ndjson(proc.stdout)
    # Live mode re-runs the prompt over the growing transcript, emitting one refresh
    # ({"turns": N, "output": ...}) per finalized turn.
    refreshes = [e for e in events if e.get("output")]
    assert refreshes, f"no live transform refresh came back; events={events}"


# --- Voice agent ----------------------------------------------------------


def test_agent_file_gets_reply(real_api_key, kokoro_pipeline, tmp_path):
    spoken = "Hi there. Can you say hello back to me in one short sentence?"
    wav = _synthesize_wav(kokoro_pipeline, spoken, tmp_path / "hello.wav")

    proc = _run_cli(["agent", str(wav), "--json"], real_api_key)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"

    events = _ndjson(proc.stdout)
    user_finals = [
        e["text"] for e in events if e.get("type") == "transcript.user" and e.get("text")
    ]
    agent_replies = [
        e["text"] for e in events if e.get("type") == "transcript.agent" and e.get("text")
    ]

    assert user_finals, f"agent never transcribed the spoken input; events={events}"
    assert agent_replies, f"agent never replied; events={events}"


# --- LLM Gateway ----------------------------------------------------------


def test_llm_command_answers(real_api_key):
    proc = _run_cli(
        ["llm", "What is 2 + 2? Reply with just the number.", "--json"], real_api_key, timeout=60
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    data = json.loads(proc.stdout)
    assert "4" in data["output"], f"unexpected LLM output: {data!r}"


# --- Transcripts list / get -----------------------------------------------


def test_transcripts_list_and_get_roundtrip(real_api_key):
    # The batch tests above leave completed transcripts; list then fetch one by id.
    proc = _run_cli(["transcripts", "list", "--limit", "5", "--json"], real_api_key, timeout=60)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    listing = json.loads(proc.stdout)
    assert isinstance(listing, list) and listing, f"empty transcripts listing: {listing!r}"
    first = listing[0]
    assert first["id"] and first["status"], f"listing row missing id/status: {first!r}"

    tid = first["id"]
    got = _run_cli(["transcripts", "get", tid, "--json"], real_api_key, timeout=60)
    assert got.returncode == 0, f"stderr:\n{got.stderr}"
    fetched = json.loads(got.stdout)
    assert fetched["id"] == tid, f"id mismatch: asked {tid}, got {fetched!r}"


# --- Diagnostics & auth ---------------------------------------------------


def test_doctor_reports_healthy(real_api_key):
    proc = _run_cli(["doctor", "--json"], real_api_key, timeout=60)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    report = json.loads(proc.stdout)
    assert report["ok"] is True, f"doctor not ok: {report!r}"
    checks = {c["name"]: c["status"] for c in report["checks"]}
    assert checks.get("api-key") == "ok", f"api-key check not ok: {checks!r}"


def test_auth_failure_is_clean(real_api_key):
    # A rejected key must produce a clean JSON error on stderr (not a traceback),
    # leave stdout empty for pipelines, and exit non-zero.
    proc = _run_cli(
        ["transcripts", "list", "--json"], "deadbeefdeadbeefdeadbeefdeadbeef", timeout=60
    )
    assert proc.returncode != 0, "expected non-zero exit on auth failure"
    assert proc.stdout.strip() == "", f"stdout should stay clean: {proc.stdout!r}"
    err = json.loads(proc.stderr)
    assert err["error"]["type"] == "not_authenticated", f"unexpected error shape: {err!r}"
    assert "Traceback" not in proc.stderr
