"""Source fidelity of `stream --show-code` generation (mic vs stdin vs file/URL).

The generated script must read the same audio input the real run would, at the
same sample rate, and every variant must compile (`python -m py_compile` parity).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from aai_cli import code_gen

_LLM = {"prompts": ["summarize"], "model": "m", "max_tokens": 100, "interval": 5.0}


def _compiles(code: str) -> None:
    # compile() is stricter than ast.parse() and is what `python file.py` runs through.
    compile(code, "<generated>", "exec")


# --- microphone (default) ----------------------------------------------------
def test_mic_variant_is_unchanged_and_has_no_source_plumbing():
    code = code_gen.stream({"sample_rate": 16000})
    _compiles(code)
    assert "client.stream(aai.extras.MicrophoneStream(sample_rate=16000))" in code
    assert "print('Listening… press Ctrl-C to stop.')" in code
    assert "import subprocess" not in code
    assert "import sys" not in code
    assert "stdin_chunks" not in code
    assert "file_chunks" not in code


def test_mic_variant_honors_sample_rate():
    code = code_gen.stream({"sample_rate": 8000})
    _compiles(code)
    assert "MicrophoneStream(sample_rate=8000)" in code
    assert "sample_rate=8000," in code  # StreamingParameters matches the capture rate


# --- stdin (`-`) ---------------------------------------------------------------
def test_stdin_variant_reads_stdin_not_the_mic():
    code = code_gen.stream({"sample_rate": 16000}, source="-")
    _compiles(code)
    assert "client.stream(stdin_chunks())" in code
    assert "sys.stdin.buffer.read(chunk_bytes)" in code
    assert "import sys" in code
    assert "MicrophoneStream" not in code


def test_stdin_variant_honors_sample_rate():
    code = code_gen.stream({"sample_rate": 8000}, source="-")
    _compiles(code)
    assert "chunk_bytes = 8000 * 2 // 10" in code
    assert "-ar 8000" in code  # the example ffmpeg pipe matches the declared rate
    assert "sample_rate=8000," in code


# --- file / URL ---------------------------------------------------------------
def test_file_variant_decodes_that_file_through_ffmpeg():
    code = code_gen.stream({"sample_rate": 16000}, source="rec.wav")
    _compiles(code)
    assert "client.stream(file_chunks())" in code
    assert "'rec.wav'" in code  # the source is embedded as the ffmpeg input
    assert '"-ar", "16000"' in code
    assert "chunk_bytes = 16000 * 2 // 10" in code
    assert "time.sleep(len(data) / (16000 * 2))" in code  # ~real-time pacing
    assert "import subprocess" in code
    assert "import time" in code
    assert "print('Streaming rec.wav…')" in code
    assert "MicrophoneStream" not in code


def test_file_variant_honors_sample_rate():
    code = code_gen.stream({"sample_rate": 8000}, source="clip.mp3")
    _compiles(code)
    assert '"-ar", "8000"' in code  # decode rate == StreamingParameters.sample_rate
    assert "sample_rate=8000," in code


def test_url_source_is_passed_to_ffmpeg_verbatim():
    code = code_gen.stream({}, source="https://assembly.ai/wildfires.mp3")
    _compiles(code)
    assert "'https://assembly.ai/wildfires.mp3'" in code
    assert "file_chunks()" in code


def test_file_variant_with_quotes_in_name_still_compiles():
    code = code_gen.stream({}, source='rec\'s "weird" name.wav')
    _compiles(code)


# --- --llm composition ---------------------------------------------------------
def test_llm_with_file_source_streams_file_and_flushes_summary():
    code = code_gen.stream({"sample_rate": 16000}, llm=_LLM, source="rec.wav")
    _compiles(code)
    assert "client.stream(file_chunks())" in code
    assert "run_chain" in code
    assert "summarize(final=True)" in code
    assert code.count("import time") == 1  # llm + file both need time; imported once


def test_llm_with_stdin_source_keeps_both_imports():
    code = code_gen.stream({}, llm=_LLM, source="-")
    _compiles(code)
    assert "client.stream(stdin_chunks())" in code
    assert "import sys" in code
    assert "import time" in code


# --- fuzz: every source shape always compiles ----------------------------------
@given(st.text(st.characters(blacklist_categories=["Cs"]), max_size=40) | st.none())
def test_fuzz_any_source_always_compiles(source):
    # Arbitrary file names (quotes, newlines, braces, unicode), "-" (stdin), ""
    # and None (mic) must all yield a compilable script.
    _compiles(code_gen.stream({"sample_rate": 16000}, source=source))
    _compiles(code_gen.stream({"sample_rate": 16000}, llm=_LLM, source=source))
