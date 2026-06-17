"""`assembly eval --llm` / `--llm-reduce`: the per-item map, the across-items
reduce, and how both ride alongside the (unchanged) WER score.

The transcription boundary and the LLM Gateway chain helpers are mocked; the
dataset is a real temp manifest so the loader runs end to end.
"""

import contextlib
import json

import pytest
from typer.testing import CliRunner

from aai_cli.commands.evaluate import _exec as evaluate_exec
from aai_cli.commands.evaluate import _render as evaluate_render
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _auth():
    config.set_api_key("default", "sk_live")


class _Transcript:
    def __init__(self, text):
        self.text = text


def _write_manifest(tmp_path):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"fake-audio")
    (tmp_path / "manifest.csv").write_text(
        "audio,text\na.wav,hello there\nb.wav,goodbye now\n", encoding="utf-8"
    )


def _mock_transcribe(mocker, texts):
    return mocker.patch(
        "aai_cli.commands.evaluate._exec.client.transcribe",
        autospec=True,
        side_effect=[_Transcript(text) for text in texts],
    )


def _payload_of(result):
    return next(
        json.loads(line) for line in result.output.splitlines() if line.startswith('{"dataset"')
    )


def _mock_chain_steps(mocker):
    """Stand in for the per-item --llm chain: one step echoing the transcript text."""

    def fake(api_key, prompts, *, transcript_text=None, transcript_id=None, model, max_tokens):
        return [{"prompt": prompts[0], "output": f"MAP::{transcript_text}"}]

    return mocker.patch("aai_cli.commands.evaluate._exec.gateway.run_chain_steps", side_effect=fake)


# the per-item map step


def test_llm_map_attaches_steps_and_leaves_wer_on_raw_transcript(tmp_path, mocker):
    _auth()
    _write_manifest(tmp_path)
    # b.wav: 1 of 2 words wrong -> row 50%, pooled 1/4 -> 25%.
    _mock_transcribe(mocker, ["hello there", "goodbye cow"])
    steps = _mock_chain_steps(mocker)
    payload = _payload_of(runner.invoke(app, ["eval", "manifest.csv", "--llm", "fix", "--json"]))
    # WER still scores the raw transcript, not the --llm output.
    assert payload["wer"] == 0.25
    assert payload["rows"][0]["wer"] == 0.0
    # Each scored row carries its chain under "llm".
    assert payload["rows"][0]["llm"] == {
        "model": "claude-haiku-4-5-20251001",
        "steps": [{"prompt": "fix", "output": "MAP::hello there"}],
    }
    # The chain ran over the raw transcript text of each row.
    texts = {call.kwargs["transcript_text"] for call in steps.call_args_list}
    assert texts == {"hello there", "goodbye cow"}
    assert all(call.args[1] == ["fix"] for call in steps.call_args_list)


def test_llm_map_renders_in_human_mode(tmp_path, mocker):
    _auth()
    _write_manifest(tmp_path)
    _mock_transcribe(mocker, ["hello there", "goodbye now"])
    _mock_chain_steps(mocker)
    result = runner.invoke(app, ["eval", "manifest.csv", "--llm", "fix"])
    assert result.exit_code == 0
    assert "--llm" in result.output
    assert "a.wav: MAP::hello there" in result.output


def test_llm_map_skips_failed_rows(tmp_path, mocker):
    from aai_cli.core.errors import APIError

    _auth()
    _write_manifest(tmp_path)
    mocker.patch(
        "aai_cli.commands.evaluate._exec.client.transcribe",
        autospec=True,
        side_effect=[_Transcript("hello there"), APIError("boom")],
    )
    steps = _mock_chain_steps(mocker)
    # Whole run exits 1 (a row failed) but the map still ran on the one good row.
    result = runner.invoke(app, ["eval", "manifest.csv", "--llm", "fix", "--json"])
    assert result.exit_code == 1
    payload = _payload_of(result)
    good = next(row for row in payload["rows"] if "wer" in row)
    bad = next(row for row in payload["rows"] if "error" in row)
    assert "llm" in good and "llm" not in bad
    assert steps.call_count == 1


def test_model_and_max_tokens_reach_the_gateway(tmp_path, mocker):
    _auth()
    _write_manifest(tmp_path)
    _mock_transcribe(mocker, ["hello there", "goodbye now"])
    steps = _mock_chain_steps(mocker)
    argv = ["eval", "manifest.csv", "--llm", "x", "--model", "gpt-5", "--max-tokens", "42"]
    assert runner.invoke(app, argv).exit_code == 0
    assert steps.call_args.kwargs["model"] == "gpt-5"
    assert steps.call_args.kwargs["max_tokens"] == 42


# the across-items reduce step


def test_reduce_combines_map_outputs_and_lands_in_payload(tmp_path, mocker):
    _auth()
    _write_manifest(tmp_path)
    _mock_transcribe(mocker, ["hello there", "goodbye now"])
    _mock_chain_steps(mocker)
    captured = {}

    def fake_reduce(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        captured["prompts"] = prompts
        return "FINAL"

    mocker.patch("aai_cli.commands.evaluate._exec.gateway.run_chain", side_effect=fake_reduce)
    payload = _payload_of(
        runner.invoke(
            app, ["eval", "manifest.csv", "--llm", "judge", "--llm-reduce", "rank", "--json"]
        )
    )
    assert payload["reduce"] == {
        "model": "claude-haiku-4-5-20251001",
        "prompts": ["rank"],
        "output": "FINAL",
    }
    # The reduce saw each item's last --llm output under an item header.
    assert "### Item: a.wav" in captured["text"]
    assert "MAP::hello there" in captured["text"] and "MAP::goodbye now" in captured["text"]
    assert captured["prompts"] == ["rank"]


def test_reduce_falls_back_to_transcript_text(tmp_path, mocker):
    _auth()
    _write_manifest(tmp_path)
    _mock_transcribe(mocker, ["hello there", "goodbye now"])
    captured = {}

    def fake_reduce(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        return "FINAL"

    mocker.patch("aai_cli.commands.evaluate._exec.gateway.run_chain", side_effect=fake_reduce)
    # No --llm map ran, so the raw transcript text feeds the reduce.
    result = runner.invoke(app, ["eval", "manifest.csv", "--llm-reduce", "sum"])
    assert result.exit_code == 0
    assert "hello there" in captured["text"]
    assert "--llm-reduce" in result.output and "FINAL" in result.output


def test_reduce_skips_gateway_when_nothing_to_aggregate(tmp_path, mocker):
    from aai_cli.core.errors import APIError

    _auth()
    _write_manifest(tmp_path)
    mocker.patch(
        "aai_cli.commands.evaluate._exec.client.transcribe",
        autospec=True,
        side_effect=[APIError("boom"), APIError("boom")],
    )
    reduce = mocker.patch("aai_cli.commands.evaluate._exec.gateway.run_chain")
    result = runner.invoke(app, ["eval", "manifest.csv", "--llm-reduce", "sum", "--json"])
    reduce.assert_not_called()
    assert "Nothing to reduce" in result.output
    assert "reduce" not in _payload_of(result)


def test_llm_status_messages(tmp_path, mocker, monkeypatch):
    _auth()
    _write_manifest(tmp_path)
    _mock_transcribe(mocker, ["hello there", "goodbye now"])
    _mock_chain_steps(mocker)
    mocker.patch(
        "aai_cli.commands.evaluate._exec.gateway.run_chain",
        side_effect=lambda *a, **k: "FINAL",
    )
    seen = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        seen.append(message)
        yield

    monkeypatch.setattr("aai_cli.commands.evaluate._exec.output.status", fake_status)
    argv = ["eval", "manifest.csv", "--llm", "judge", "--llm-reduce", "rank"]
    assert runner.invoke(app, argv).exit_code == 0
    assert "Running --llm over 2 transcripts…" in seen
    assert "Running --llm-reduce over all items…" in seen


# ---------------------------------------------------------------- pure helpers


def test_llm_options_maps_fields():
    opts = evaluate_exec.EvalOptions(
        datasets=["d"], split=None, subset=None, limit=10, audio_column=None,
        text_column=None, speech_model=None, language_code=None, concurrency=1,
        llm_prompt=["a"], llm_reduce=["b", "c"], model="m", max_tokens=5,
    )  # fmt: skip
    llm_opts = opts.llm_options()
    assert llm_opts.prompts == ["a"]
    assert llm_opts.reduce_prompts == ["b", "c"]
    assert llm_opts.model == "m"
    assert llm_opts.max_tokens == 5


def test_llm_options_default_to_empty_chains():
    opts = evaluate_exec.EvalOptions(
        datasets=["d"], split=None, subset=None, limit=10, audio_column=None,
        text_column=None, speech_model=None, language_code=None, concurrency=1,
        llm_prompt=None, llm_reduce=None, model="m", max_tokens=5,
    )  # fmt: skip
    llm_opts = opts.llm_options()
    assert llm_opts.prompts == [] and llm_opts.reduce_prompts == []


def _result(row, hypothesis):
    return evaluate_exec._ItemResult(row=row, words=None, latency=0.0, hypothesis=hypothesis)


def test_reduce_input_prefers_last_llm_output_else_transcript():
    with_llm = _result(
        {"item": "a", "llm": {"steps": [{"output": "first"}, {"output": "last"}]}}, "raw"
    )
    assert evaluate_exec._reduce_input(with_llm) == "last"
    no_llm = _result({"item": "b"}, "raw text")
    assert evaluate_exec._reduce_input(no_llm) == "raw text"
    empty_steps = _result({"item": "c", "llm": {"steps": []}}, "fallback")
    assert evaluate_exec._reduce_input(empty_steps) == "fallback"


def test_gather_reduce_inputs_headers_and_skips_failures_and_blanks():
    results = [
        _result({"item": "a"}, "hello"),
        _result({"item": "fail", "error": "boom"}, None),  # failed: no hypothesis
        _result({"item": "blank"}, ""),  # empty transcript contributes nothing
    ]
    combined = evaluate_exec._gather_reduce_inputs(results)
    assert combined == "### Item: a\nhello"


def test_final_llm_output_distinguishes_absent_from_empty():
    assert evaluate_render._final_llm_output({"item": "a"}) is None
    assert evaluate_render._final_llm_output({"llm": {"steps": []}}) == ""
    assert evaluate_render._final_llm_output({"llm": {"steps": [{"output": "x"}]}}) == "x"


def test_render_blocks_drop_out_when_absent():
    assert evaluate_render._llm_block({"rows": [{"item": "a"}]}) is None
    assert evaluate_render._reduce_block({}) is None


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_llm_options_is_immutable():
    import dataclasses

    llm_opts = evaluate_exec._LlmOptions(prompts=[], reduce_prompts=[], model="m", max_tokens=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(llm_opts, "model", "other")
