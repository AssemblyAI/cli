import json
import types

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def test_stream_llm_refreshes_live_over_growing_transcript(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {"texts": []}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hola", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript="mundo", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript="partial", end_of_turn=False))  # ignored
            on_turn(types.SimpleNamespace(transcript="no-eot"))  # missing flag -> not final

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["texts"].append(transcript_text)
        seen["prompts"] = prompts
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return f"answer:{transcript_text}"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(
        app,
        [
            "stream",
            "--llm",
            "translate to english",
            "--model",
            "gpt-4.1",
            "--max-tokens",
            "50",
            "--json",
        ],
    )
    assert result.exit_code == 0
    # One refresh per finalized turn, over the growing transcript (partials ignored).
    assert seen["texts"] == ["hola", "hola mundo"]
    assert seen["prompts"] == ["translate to english"]
    assert seen["model"] == "gpt-4.1"
    assert seen["max_tokens"] == 50
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"turns": 1, "output": "answer:hola"} in lines
    assert {"turns": 2, "output": "answer:hola mundo"} in lines
    # Live mode replaces the raw turn envelopes; only follow refreshes reach stdout.
    assert '"type"' not in result.output


def test_stream_llm_chains_multiple_prompts(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True))

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["prompts"] = prompts
        return "done"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(
        app, ["stream", "--llm", "summarize", "--llm", "translate to french", "--json"]
    )
    assert result.exit_code == 0
    assert seen["prompts"] == ["summarize", "translate to french"]


def test_stream_llm_rejects_output_text(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not stream")),
    )
    result = runner.invoke(app, ["stream", "--llm", "summarize", "-o", "text"])
    assert result.exit_code == 2  # --llm renders a panel/NDJSON; -o text is contradictory


def test_stream_without_prompt_does_not_transform(monkeypatch):
    config.set_api_key("default", "sk_live")
    called = {"ran": False}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True))

    def fake_run_chain(*a, **k):
        called["ran"] = True
        return "x"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    assert called["ran"] is False  # no --llm -> no gateway call


def test_stream_show_code_with_llm_emits_follow_loop(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not stream")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _boom)
    result = runner.invoke(app, ["stream", "--llm", "summarize", "--show-code"])
    assert result.exit_code == 0
    assert "from openai import OpenAI" in result.output
    assert "summarize" in result.output
    assert "run_chain" in result.output  # the live transcribe->LLM-per-turn loop


def _eot_turn(text):
    # A finalized turn with no diarized speaker (the shape SDK emits without --speaker-labels).
    return types.SimpleNamespace(transcript=text, end_of_turn=True, speaker_label=None)


def _llm_session(*, interval, clock, monkeypatch, emitted):
    import io

    from aai_cli.commands.stream import StreamSession
    from aai_cli.follow import FollowRenderer
    from aai_cli.streaming.render import StreamRenderer

    # Capture each follow refresh (json mode emits one NDJSON object per refresh) and
    # make run_chain echo the transcript it summarized so assertions read the cadence.
    monkeypatch.setattr("aai_cli.follow.output.emit_ndjson", lambda obj: emitted.append(obj))
    monkeypatch.setattr(
        "aai_cli.streaming.session.llm.run_chain",
        lambda api_key, prompts, *, transcript_text, model, max_tokens: transcript_text,
    )
    return StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=FollowRenderer(json_mode=True),
        llm_prompts=["sum"],
        model="m",
        max_tokens=10,
        llm_interval=interval,
        clock=clock,
    )


def test_stream_llm_interval_throttles_between_ticks(monkeypatch):
    # --llm-interval re-runs the chain on the first turn, skips turns inside the window,
    # and runs again once the interval has elapsed (turns still accumulate throughout).
    now = {"t": 1000.0}
    emitted: list[dict] = []
    session = _llm_session(
        interval=30.0, clock=lambda: now["t"], monkeypatch=monkeypatch, emitted=emitted
    )
    session.on_turn(_eot_turn("one"))  # first turn -> immediate summary
    now["t"] = 1010.0
    session.on_turn(_eot_turn("two"))  # +10s within the window -> throttled
    now["t"] = 1040.0
    session.on_turn(_eot_turn("three"))  # 40s since the last refresh -> summary
    session._maybe_summarize(final=True)  # nothing new since -> no-op
    assert [e["output"] for e in emitted] == ["one", "one two three"]
    assert [e["turns"] for e in emitted] == [1, 3]


def test_stream_llm_final_flush_summarizes_tail(monkeypatch):
    # A closing flush summarizes turns that arrived after the last interval tick, so the
    # tail of the conversation is never dropped when the stream stops mid-window.
    now = {"t": 1000.0}
    emitted: list[dict] = []
    session = _llm_session(
        interval=30.0, clock=lambda: now["t"], monkeypatch=monkeypatch, emitted=emitted
    )
    session.on_turn(_eot_turn("a"))  # immediate summary "a"
    session.on_turn(_eot_turn("b"))  # same clock -> throttled
    session._maybe_summarize(final=True)  # flush the tail -> "a b"
    assert [e["output"] for e in emitted] == ["a", "a b"]


def test_stream_llm_interval_below_one_second_still_throttles(monkeypatch):
    # A sub-second interval is still interval mode (llm_interval > 0): turns inside the
    # window are batched into one closing flush, not emitted per turn. Pins the `> 0`
    # boundary so it can't drift to `> 1` and silently treat 0<interval<=1 as per-turn.
    now = {"t": 1000.0}
    emitted: list[dict] = []
    session = _llm_session(
        interval=0.5, clock=lambda: now["t"], monkeypatch=monkeypatch, emitted=emitted
    )
    session.on_turn(_eot_turn("a"))  # first turn -> "a"
    session.on_turn(_eot_turn("b"))  # within the 0.5s window -> throttled
    session.on_turn(_eot_turn("c"))  # still within the window -> throttled
    session._maybe_summarize(final=True)  # flush the batch -> "a b c"
    assert [e["output"] for e in emitted] == ["a", "a b c"]


def test_stream_llm_interval_zero_summarizes_every_turn(monkeypatch):
    # --llm-interval 0 keeps the legacy per-turn cadence: every finalized turn refreshes.
    now = {"t": 1000.0}
    emitted: list[dict] = []
    session = _llm_session(
        interval=0.0, clock=lambda: now["t"], monkeypatch=monkeypatch, emitted=emitted
    )
    session.on_turn(_eot_turn("a"))
    session.on_turn(_eot_turn("b"))  # not throttled despite the unchanged clock
    assert [e["output"] for e in emitted] == ["a", "a b"]


def test_maybe_summarize_is_noop_without_follow():
    # Defensive guard: with no FollowRenderer there's nothing to refresh, so the chain
    # is never run (no gateway call) regardless of transcript content.
    import io

    from aai_cli.commands.stream import StreamSession
    from aai_cli.streaming.render import StreamRenderer

    session = StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=["sum"],
        model="m",
        max_tokens=10,
    )
    assert session.llm_interval == 0.0  # the default cadence is per-turn until set
    session.transcript.append("x")
    session._maybe_summarize(final=True)  # must not raise or call the gateway
