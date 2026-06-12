"""Example-based code_gen tests for the stream and agent scaffolds.

Split from test_code_gen.py (serializers, snippets, transcribe rendering);
hypothesis fuzz/property tests live in test_code_gen_fuzz.py.
"""

from __future__ import annotations

import ast

from aai_cli import code_gen


def test_stream_render_parses_and_is_runnable_shape():
    from assemblyai.streaming.v3 import SpeechModel

    code = code_gen.stream(
        {"sample_rate": 16000, "format_turns": True, "speech_model": SpeechModel.u3_rt_pro}
    )
    ast.parse(code)
    assert "StreamingClient(" in code
    assert "StreamingParameters(" in code
    assert "SpeechModel.u3_rt_pro" in code
    assert "MicrophoneStream" in code
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code


def test_stream_render_mic_rate_matches_params():
    code = code_gen.stream({"sample_rate": 8000})
    ast.parse(code)
    assert "StreamingParameters(\n        sample_rate=8000," in code
    assert "MicrophoneStream(sample_rate=8000)" in code


def test_stream_render_empty_is_clean_and_has_no_speechmodel_import():
    code = code_gen.stream({})
    ast.parse(code)
    assert "StreamingParameters()" in code
    assert "    SpeechModel," not in code  # not imported when unused (keeps script lint-clean)
    assert "MicrophoneStream(sample_rate=16000)" in code  # default rate


def test_agent_render_parses_and_injects_session_fields():
    code = code_gen.agent(voice="ivy", system_prompt="Be terse.", greeting="Hi there")
    ast.parse(code)
    assert '"voice": "ivy"' in code
    assert "Be terse." in code
    assert "Hi there" in code
    assert "agents.assemblyai.com" in code
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code


def test_agent_render_mic_thread_ends_quietly_when_the_socket_closes():
    # The generated send_mic daemon thread blocks on ws.send(); when the session
    # ends and the socket closes, that send raises. Without the guard, the thread
    # would dump a traceback to stderr on every normal exit of the sample script.
    code = code_gen.agent(voice="ivy", system_prompt="Be terse.", greeting="Hi")
    ast.parse(code)
    assert "except Exception:" in code
    assert "return  # socket closed (session over): end the mic thread quietly" in code


def test_agent_render_escapes_quotes_in_prompt():
    import json as _json

    tricky = 'Say "hi"\nand stop'
    code = code_gen.agent(voice="ivy", system_prompt=tricky, greeting="Hello")
    ast.parse(code)  # valid Python despite embedded quotes/newlines
    # The prompt is injected via json.dumps, so its escaped form appears verbatim.
    assert _json.dumps(tricky) in code


def test_agent_show_code_uses_single_full_duplex_stream():
    # ONE sd.RawStream (mic+speaker); two separate streams fail on macOS CoreAudio.
    code = code_gen.agent(voice="ivy", system_prompt="p", greeting="g")
    ast.parse(code)
    assert "sd.RawStream(" in code
    assert "samplerate=RATE" in code  # opens at the agent's native 24 kHz, no resampling
    assert "RawInputStream" not in code
    assert "RawOutputStream" not in code
    # No audioop: it's deprecated and removed in Python 3.13, so the script stays portable.
    assert "audioop" not in code


def test_stream_show_code_includes_llm_follow_loop():
    code = code_gen.stream(
        {"speech_model": "universal_streaming"},
        llm={
            "prompts": ["summarize", "translate to french"],
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "interval": 30.0,
        },
    )
    ast.parse(code)
    assert "from openai import OpenAI" in code
    assert "llm-gateway.assemblyai.com" in code
    # Both prompts appear, in order, for the chain.
    assert code.index("summarize") < code.index("translate to french")
    # Still streams from the mic, refreshing the answer on the interval.
    assert "MicrophoneStream" in code
    assert "end_of_turn" in code
    assert "claude-haiku-4-5-20251001" in code
    # The generated loop mirrors --llm-interval: a baked-in throttle plus a closing flush.
    assert "LLM_INTERVAL = 30.0" in code
    assert "now - _last_summary < LLM_INTERVAL" in code
    assert "summarize(final=True)" in code


def test_gateway_options_defaults_interval_to_per_turn():
    # Called without an explicit interval (transcribe's path), the baked-in cadence is
    # per-turn (0.0); pins the default so it can't drift.
    opts = code_gen.gateway_options(["summarize"], "m", 100)
    assert opts is not None
    assert opts["interval"] == 0.0


def test_stream_show_code_defaults_interval_when_absent():
    # An llm dict with no "interval" key falls back to per-turn (LLM_INTERVAL = 0.0).
    code = code_gen.stream({}, llm={"prompts": ["s"], "model": "m", "max_tokens": 1})
    ast.parse(code)
    assert "LLM_INTERVAL = 0.0" in code


def test_stream_show_code_llm_interval_zero_is_per_turn():
    # --llm-interval 0 bakes in the legacy per-turn cadence (LLM_INTERVAL = 0.0 makes the
    # throttle a no-op), while still emitting the closing flush.
    code = code_gen.stream(
        {},
        llm=code_gen.gateway_options(["summarize"], "m", 100, interval=0.0),
    )
    ast.parse(code)
    assert "LLM_INTERVAL = 0.0" in code
    assert "summarize(final=True)" in code


def test_stream_show_code_without_llm_is_plain_scaffold():
    code = code_gen.stream({})
    ast.parse(code)
    assert "from openai import OpenAI" not in code  # no gateway when --llm absent
    assert "MicrophoneStream" in code
