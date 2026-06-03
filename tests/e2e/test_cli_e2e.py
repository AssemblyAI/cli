"""End-to-end tests that drive the real `aai` CLI against the live AssemblyAI API.

Speech is synthesized locally with kokoro TTS, then fed through the CLI as a
subprocess so the binary, argument parsing, auth, audio decoding, and network
path are all exercised for real — no mocks.

These tests are marked `e2e` and skip (never fail) when the API key, kokoro, or
numpy is unavailable, so CI and keyless contributors are not blocked. The
precommit `pytest-e2e` hook runs them; the default unit run excludes them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

KOKORO_RATE = 24000  # kokoro emits 24 kHz float32 mono
STREAM_RATE = 16000  # what the CLI's fast WAV path expects (16 kHz mono PCM16)


@pytest.fixture(scope="session")
def kokoro_pipeline() -> Any:
    """Build the kokoro TTS pipeline once per session, or skip if unavailable."""
    pytest.importorskip("numpy")
    kokoro = pytest.importorskip("kokoro")
    return kokoro.KPipeline(lang_code="a")  # American English


def _synthesize_wav(pipeline: Any, text: str, path: Path, *, lead_silence_s: float = 0.6) -> Path:
    """Synthesize `text` to a 16 kHz mono PCM16 WAV the CLI can stream directly.

    Resamples kokoro's 24 kHz output to 16 kHz (linear) and prepends a short
    silence so nothing is clipped before the realtime session is ready.
    """
    import numpy as np

    chunks = []
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
    """Run `python -m assemblyai_cli <args>` against the working tree with the real key."""
    env = dict(os.environ)
    env["ASSEMBLYAI_API_KEY"] = key
    return subprocess.run(
        [sys.executable, "-m", "assemblyai_cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _ndjson(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


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


# --- LLM Gateway -----------------------------------------------------------


def test_llm_command_answers(real_api_key):
    proc = _run_cli(
        ["llm", "What is 2 + 2? Reply with just the number.", "--json"], real_api_key, timeout=60
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    data = json.loads(proc.stdout)
    assert "4" in data["output"], f"unexpected LLM output: {data!r}"


def test_transcribe_prompt_transforms_via_gateway(real_api_key):
    proc = _run_cli(
        [
            "transcribe",
            "--sample",
            "--prompt",
            "Summarize this transcript in one short sentence.",
            "--json",
        ],
        real_api_key,
        timeout=180,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["text"].strip(), f"no transcript produced: {data!r}"
    assert data["transform"]["output"].strip(), f"gateway returned no transform: {data!r}"


def test_stream_prompt_transforms_at_end(real_api_key, kokoro_pipeline, tmp_path):
    spoken = "the quick brown fox jumps over the lazy dog"
    wav = _synthesize_wav(kokoro_pipeline, spoken, tmp_path / "fox.wav")

    proc = _run_cli(
        [
            "stream",
            str(wav),
            "--prompt",
            "Summarize the transcript in one short sentence.",
            "--json",
        ],
        real_api_key,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    events = _ndjson(proc.stdout)
    # The full transcript is transformed once after streaming, emitted as a final llm event.
    llm_events = [e for e in events if e.get("type") == "llm" and e.get("content")]
    assert llm_events, f"no transcript transform came back; events={events}"
