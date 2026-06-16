"""`assembly transcribe --llm-reduce`: the map-reduce LLM step.

Task 1 covers the data plumbing (the flag and TransformOptions); later tasks add
the single-source chain and batch-reduce behavior tests to this file.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

from aai_cli.app.transcribe import batch as transcribe_batch
from aai_cli.app.transcribe import run as transcribe_run
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()

_TRANSCRIBE = "aai_cli.app.transcribe.run.client.transcribe"
_TRANSFORM = "aai_cli.core.llm.transform_transcript"


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    # Batch sources and sidecars resolve relative to cwd; isolate each test.
    monkeypatch.chdir(tmp_path)


def _auth() -> None:
    config.set_api_key("default", "sk_live")


def _fake_transcript(mocker, source):
    t = mocker.MagicMock()
    t.id = f"t_{source}"
    t.text = f"text of {source}"
    t.status = "completed"
    t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
    return t


def _ndjson(result):
    return [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]


_DEFAULT_OPTS = transcribe_run.TranscribeOptions(
    source=None,
    sample=False,
    from_stdin=False,
    concurrency=2,
    force=False,
    speech_model=None,
    language_code=None,
    language_detection=None,
    keyterms_prompt=None,
    temperature=None,
    prompt=None,
    punctuate=None,
    format_text=None,
    disfluencies=None,
    speaker_labels=False,
    speakers_expected=None,
    multichannel=None,
    redact_pii=None,
    redact_pii_policy=None,
    redact_pii_sub=None,
    redact_pii_audio=None,
    filter_profanity=None,
    content_safety=None,
    content_safety_confidence=None,
    speech_threshold=None,
    summarization=None,
    summary_model=None,
    summary_type=None,
    auto_chapters=None,
    sentiment_analysis=None,
    entity_detection=None,
    auto_highlights=None,
    topic_detection=None,
    word_boost=None,
    custom_spelling_file=None,
    audio_start=None,
    audio_end=None,
    download_sections=None,
    webhook_url=None,
    webhook_auth_header=None,
    translate_to=None,
    config_kv=None,
    config_file=None,
    llm_prompt=None,
    llm_reduce=None,
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    output_field=None,
    chars_per_caption=None,
    out=None,
    show_code=False,
)


def test_transform_options_carries_reduce_prompts() -> None:
    opts = dataclasses.replace(
        _DEFAULT_OPTS, llm_prompt=["judge"], llm_reduce=["rank", "summarize"]
    )
    transform = opts.transform_options()
    assert transform.prompts == ["judge"]
    assert transform.reduce_prompts == ["rank", "summarize"]


def test_chain_appends_reduce_to_map() -> None:
    transform = transcribe_run.TransformOptions(
        prompts=["a"], model="m", max_tokens=10, reduce_prompts=["b"]
    )
    assert transform.chain() == ["a", "b"]


def test_reduce_input_prefers_map_output_then_text_then_empty() -> None:
    # The last --llm step's output wins over the transcript text.
    assert (
        transcribe_batch._reduce_input(
            {"transform": {"steps": [{"output": "judged"}]}, "transcript": {"text": "raw"}}
        )
        == "judged"
    )
    # Falls back to transcript text when no --llm step ran.
    assert transcribe_batch._reduce_input({"transcript": {"text": "raw text"}}) == "raw text"
    # Empty when the record carries neither.
    assert transcribe_batch._reduce_input({}) == ""


def test_gather_reduce_inputs_skips_non_completed_items() -> None:
    done = transcribe_batch._Item("https://a", status="completed")
    failed = transcribe_batch._Item("https://b", status="failed")
    transcribe_batch._dump_sidecar(
        transcribe_batch.sidecar_path("https://a"),
        {"status": "completed", "transcript": {"text": "alpha text"}},
    )
    # `b` has a perfectly valid completed sidecar; only its item status excludes it,
    # so dropping the status guard would wrongly pull "beta text" into the reduce.
    transcribe_batch._dump_sidecar(
        transcribe_batch.sidecar_path("https://b"),
        {"status": "completed", "transcript": {"text": "beta text"}},
    )
    combined = transcribe_batch._gather_reduce_inputs([done, failed])
    assert "### Source: https://a" in combined
    assert "alpha text" in combined
    assert "beta text" not in combined


def test_single_source_runs_reduce_as_chain_step(mocker):
    _auth()
    mocker.patch(
        _TRANSCRIBE,
        return_value=mocker.MagicMock(
            id="t1",
            text="hello",
            status="completed",
            json_response={"id": "t1", "text": "hello", "status": "completed"},
        ),
    )
    transform = mocker.patch(_TRANSFORM, side_effect=["mapped", "reduced"])
    result = runner.invoke(app, ["transcribe", "--sample", "--llm", "map", "--llm-reduce", "red"])
    assert result.exit_code == 0, result.output
    # Two chain steps ran: --llm then --llm-reduce, over the one transcript.
    assert transform.call_count == 2
    assert "reduced" in result.output


def test_batch_reduce_feeds_map_outputs(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(
        _TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio)
    )
    mocker.patch(_TRANSFORM, side_effect=["JUDGED a", "JUDGED b", "FINAL"])
    captured = {}

    def spy(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        captured["prompts"] = prompts
        return "FINAL"

    monkeypatch.setattr(transcribe_batch.llm, "run_chain", spy)
    result = runner.invoke(
        app,
        ["transcribe", "--from-stdin", "--llm", "judge", "--llm-reduce", "rank"],
        input="https://a\nhttps://b\n",
    )
    assert result.exit_code == 0, result.output
    assert "### Source: https://a" in captured["text"]
    assert "JUDGED a" in captured["text"] and "JUDGED b" in captured["text"]
    assert captured["prompts"] == ["rank"]
    assert "FINAL" in result.output


def test_batch_reduce_falls_back_to_transcript_text(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(
        _TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio)
    )
    captured = {}

    def spy(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        return "FINAL"

    monkeypatch.setattr(transcribe_batch.llm, "run_chain", spy)
    result = runner.invoke(
        app,
        ["transcribe", "--from-stdin", "--llm-reduce", "summarize"],
        input="https://a\n",
    )
    assert result.exit_code == 0, result.output
    assert "text of https://a" in captured["text"]


def test_batch_reduce_emits_json_record(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(
        _TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio)
    )
    monkeypatch.setattr(
        transcribe_batch.llm,
        "run_chain",
        lambda api_key, prompts, *, transcript_text, model, max_tokens: "FINAL",
    )
    result = runner.invoke(
        app,
        ["transcribe", "--from-stdin", "--llm-reduce", "summarize", "--json"],
        input="https://a\n",
    )
    assert result.exit_code == 0, result.output
    records = _ndjson(result)
    reduce_records = [r for r in records if r.get("type") == "reduce"]
    assert len(reduce_records) == 1
    assert reduce_records[0]["output"] == "FINAL"
    assert reduce_records[0]["prompts"] == ["summarize"]


def test_batch_reduce_routes_table_to_stderr(mocker, monkeypatch, capsys):
    """run_batch sends the reduce result to stdout and the progress table to stderr."""
    import assemblyai as aai

    _auth()
    monkeypatch.setattr(
        _TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio)
    )
    monkeypatch.setattr(
        transcribe_batch.llm,
        "run_chain",
        lambda api_key, prompts, *, transcript_text, model, max_tokens: "AGGREGATE",
    )
    transform = transcribe_run.TransformOptions(
        prompts=[], model="m", max_tokens=10, reduce_prompts=["summarize"]
    )
    transcribe_batch.run_batch(
        "sk_live",
        ["https://a"],
        transcription_config=aai.TranscriptionConfig(),
        concurrency=1,
        force=False,
        transform=transform,
        json_mode=False,
        quiet=False,
    )
    out, err = capsys.readouterr()
    assert "AGGREGATE" in out
    assert "AGGREGATE" not in err
    assert "https://a" in err


def test_batch_reduce_skips_when_nothing_to_reduce(mocker, monkeypatch, capsys):
    """An all-empty batch result must not fire a (billable) reduce call."""
    import assemblyai as aai

    _auth()

    def _empty(api_key, audio, *, config):
        t = mocker.MagicMock()
        t.id = "t_a"
        t.text = ""
        t.status = "completed"
        t.json_response = {"id": "t_a", "text": "", "status": "completed"}
        return t

    monkeypatch.setattr(_TRANSCRIBE, _empty)
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return "SHOULD NOT RUN"

    monkeypatch.setattr(transcribe_batch.llm, "run_chain", _spy)
    transform = transcribe_run.TransformOptions(
        prompts=[], model="m", max_tokens=10, reduce_prompts=["summarize"]
    )
    transcribe_batch.run_batch(
        "sk_live",
        ["https://a"],
        transcription_config=aai.TranscriptionConfig(),
        concurrency=1,
        force=False,
        transform=transform,
        json_mode=False,
        quiet=False,
    )
    out, err = capsys.readouterr()
    assert calls["n"] == 0
    assert "SHOULD NOT RUN" not in out
    assert "Nothing to reduce" in err
