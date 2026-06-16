"""Unit tests for aai_cli.streaming.record — the --save-audio WAV tee."""

from __future__ import annotations

import wave

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.streaming import record


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())


def test_tee_wav_yields_chunks_unchanged(tmp_path):
    chunks = [b"\x01\x02", b"\x03\x04\x05\x06"]
    out = list(record.tee_wav(iter(chunks), tmp_path / "a.wav", rate=16000))
    assert out == chunks  # the tee must not alter what's streamed onward


def test_tee_wav_writes_a_valid_wav_with_the_source_rate(tmp_path):
    path = tmp_path / "a.wav"
    list(record.tee_wav(iter([b"\x01\x02", b"\x03\x04"]), path, rate=44100))
    channels, width, rate, frames = _read_wav(path)
    assert channels == 1
    assert width == 2
    assert rate == 44100  # the declared source rate, not a hardcoded default
    assert frames == b"\x01\x02\x03\x04"


def test_tee_wav_finalizes_a_valid_wav_on_early_close(tmp_path):
    # Ctrl-C closes the generator mid-stream; the partial file must still be valid WAV.
    path = tmp_path / "a.wav"
    gen = record.tee_wav(iter([b"\x01\x02", b"\x03\x04"]), path, rate=16000)
    assert next(gen) == b"\x01\x02"  # consume only the first chunk
    gen.close()  # raises GeneratorExit at the yield -> finally closes the WAV
    _channels, _width, _rate, frames = _read_wav(path)
    assert frames == b"\x01\x02"  # only the consumed chunk landed


def test_tee_wav_empty_stream_writes_a_zero_length_wav(tmp_path):
    path = tmp_path / "a.wav"
    assert list(record.tee_wav(iter([]), path, rate=16000)) == []
    _channels, _width, _rate, frames = _read_wav(path)
    assert frames == b""


def test_tee_wav_unopenable_path_is_a_clean_error(tmp_path):
    # Pointing at a directory can't be opened for writing -> a CLIError, not a raw OSError.
    with pytest.raises(CLIError) as excinfo:
        # tee_wav opens lazily on first iteration, so the generator must be started.
        next(record.tee_wav(iter([b"\x01\x02"]), tmp_path, rate=16000))
    assert excinfo.value.error_type == "save_audio_path"


def test_validate_target_accepts_an_existing_directory(tmp_path):
    record.validate_target(tmp_path / "rec.wav")  # parent exists -> no raise


def test_validate_target_rejects_a_missing_parent_directory(tmp_path):
    with pytest.raises(CLIError) as excinfo:
        record.validate_target(tmp_path / "nope" / "rec.wav")
    assert excinfo.value.error_type == "save_audio_path"
    assert excinfo.value.exit_code == 2
