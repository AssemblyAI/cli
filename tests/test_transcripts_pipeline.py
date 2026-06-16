"""`assembly transcripts get` stdin batching and the --llm/--llm-reduce map-reduce.

Split out of test_transcripts.py (which holds the single-fetch and `list` tests) to
keep each file under the 500-line gate.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from aai_cli.core import client, config
from aai_cli.main import app

runner = CliRunner()


def _fake_transcript(mocker, *, id_, text):
    fake = mocker.MagicMock()
    fake.id = id_
    fake.text = text
    fake.status = "completed"
    fake.json_response = None
    return fake


def _dispatch_by_id(mocker, mapping):
    def fetch(_api_key, transcript_id):
        return _fake_transcript(mocker, id_=transcript_id, text=mapping[transcript_id])

    return mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, side_effect=fetch
    )


def _patch_chain_steps(mocker, fn):
    return mocker.patch("aai_cli.commands.transcripts.llm.run_chain_steps", side_effect=fn)


def _patch_reduce(mocker, fn):
    return mocker.patch("aai_cli.commands.transcripts.llm.run_chain", side_effect=fn)


def test_parse_transcript_ids_reads_list_json_array():
    # The array `transcripts list --json` prints: ids pulled out, order preserved, deduped.
    text = json.dumps([{"id": "t1", "status": "completed"}, {"id": "t2"}, {"id": "t1"}, {"id": ""}])
    assert client.parse_transcript_ids(text) == ["t1", "t2"]


def test_parse_transcript_ids_reads_single_object_and_string_array():
    assert client.parse_transcript_ids('{"id": "t9", "text": "hi"}') == ["t9"]
    assert client.parse_transcript_ids('["t1", "t2"]') == ["t1", "t2"]


def test_parse_transcript_ids_falls_back_to_lines():
    # Plain text (e.g. `jq -r '.[].id'`): one id per line, blanks dropped, deduped in order.
    assert client.parse_transcript_ids("t1\n\n t2 \nt1\n") == ["t1", "t2"]


def test_parse_transcript_ids_empty_input_yields_no_ids():
    assert client.parse_transcript_ids("   \n  ") == []


def test_get_reads_ids_from_piped_list_json(mocker):
    # The headline pipeline: `transcripts list --json | transcripts get -o text`, jq-free.
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "first text", "t2": "second text"})
    piped = json.dumps([{"id": "t1", "status": "completed"}, {"id": "t2", "status": "completed"}])
    result = runner.invoke(app, ["transcripts", "get", "-o", "text"], input=piped)
    assert result.exit_code == 0
    # One transcript's text per line, in the piped order.
    assert result.output.splitlines() == ["first text", "second text"]


def test_get_batch_json_emits_one_ndjson_record_per_id(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "first", "t2": "second"})
    result = runner.invoke(app, ["transcripts", "get", "--json"], input="t1\nt2\n")
    assert result.exit_code == 0
    records = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    # NDJSON stream: one record per id, each tagged with the CLI-wide "type" discriminator.
    assert [r["type"] for r in records] == ["transcript", "transcript"]
    assert [r["id"] for r in records] == ["t1", "t2"]


def test_get_batch_human_prints_plain_text_not_json(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "alpha", "t2": "beta"})
    result = runner.invoke(app, ["transcripts", "get"], input="t1\nt2\n")
    assert result.exit_code == 0
    # Human batch stays plain text — no NDJSON "type" wrapper leaks in.
    assert "alpha" in result.output and "beta" in result.output
    assert "type" not in result.output


def test_get_no_id_and_no_stdin_is_usage_error(mocker):
    config.set_api_key("default", "sk_live")
    get = mocker.patch("aai_cli.commands.transcripts.client.get_transcript", autospec=True)
    result = runner.invoke(app, ["transcripts", "get"])
    assert result.exit_code == 2
    assert "Give a transcript id" in result.output
    get.assert_not_called()


def test_get_stdin_without_ids_is_usage_error(mocker):
    config.set_api_key("default", "sk_live")
    get = mocker.patch("aai_cli.commands.transcripts.client.get_transcript", autospec=True)
    result = runner.invoke(app, ["transcripts", "get"], input="[]")
    assert result.exit_code == 2
    assert "No transcript ids found on stdin" in result.output
    get.assert_not_called()


def test_get_single_llm_runs_chain_over_transcript_id(mocker):
    config.set_api_key("default", "sk_live")
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript",
        autospec=True,
        return_value=_fake_transcript(mocker, id_="t_42", text="hello"),
    )
    seen = {}

    def steps(api_key, prompts, *, transcript_id, model, max_tokens):
        seen["transcript_id"] = transcript_id
        seen["prompts"] = list(prompts)
        return [{"prompt": prompts[-1], "output": "SUMMARY"}]

    _patch_chain_steps(mocker, steps)
    reduce = _patch_reduce(mocker, lambda *a, **k: "")
    result = runner.invoke(app, ["transcripts", "get", "t_42", "--llm", "summarize"])
    assert result.exit_code == 0, result.output
    # The map injects the transcript server-side by id, and only the --llm prompt runs.
    assert seen == {"transcript_id": "t_42", "prompts": ["summarize"]}
    assert "SUMMARY" in result.output
    reduce.assert_not_called()


def test_get_single_reduce_extends_the_chain(mocker):
    # One id: nothing to aggregate, so --llm-reduce becomes extra chain steps (no reduce call).
    config.set_api_key("default", "sk_live")
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript",
        autospec=True,
        return_value=_fake_transcript(mocker, id_="t1", text="hi"),
    )
    seen = {}

    def steps(api_key, prompts, *, transcript_id, model, max_tokens):
        seen["prompts"] = list(prompts)
        return [{"prompt": prompts[-1], "output": "FINAL"}]

    _patch_chain_steps(mocker, steps)
    reduce = _patch_reduce(mocker, lambda *a, **k: "")
    result = runner.invoke(app, ["transcripts", "get", "t1", "--llm", "map", "--llm-reduce", "red"])
    assert result.exit_code == 0, result.output
    assert seen["prompts"] == ["map", "red"]
    reduce.assert_not_called()
    assert "FINAL" in result.output


def test_get_batch_llm_maps_each_transcript(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "a", "t2": "b"})

    def steps(api_key, prompts, *, transcript_id, model, max_tokens):
        return [{"prompt": prompts[-1], "output": f"SUM-{transcript_id}"}]

    _patch_chain_steps(mocker, steps)
    reduce = _patch_reduce(mocker, lambda *a, **k: "")
    result = runner.invoke(app, ["transcripts", "get", "--llm", "summarize"], input="t1\nt2\n")
    assert result.exit_code == 0, result.output
    # One map per transcript, rendered as plain text — not the JSON NDJSON wrapper.
    assert "SUM-t1" in result.output and "SUM-t2" in result.output
    assert "type" not in result.output
    reduce.assert_not_called()


def test_get_batch_llm_json_emits_transform_records(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "a", "t2": "b"})

    def steps(api_key, prompts, *, transcript_id, model, max_tokens):
        return [{"prompt": "summarize", "output": f"S-{transcript_id}"}]

    _patch_chain_steps(mocker, steps)
    result = runner.invoke(
        app, ["transcripts", "get", "--llm", "summarize", "--json"], input="t1\nt2\n"
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert [r["type"] for r in records] == ["transcript", "transcript"]
    outputs = [r["transform"]["steps"][-1]["output"] for r in records]
    assert outputs == ["S-t1", "S-t2"]


def test_get_batch_reduce_over_transcript_texts(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "alpha", "t2": "beta"})
    steps = mocker.patch("aai_cli.commands.transcripts.llm.run_chain_steps", autospec=True)
    captured = {}

    def reduce(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        captured["prompts"] = list(prompts)
        return "RANKED"

    _patch_reduce(mocker, reduce)
    result = runner.invoke(app, ["transcripts", "get", "--llm-reduce", "rank"], input="t1\nt2\n")
    assert result.exit_code == 0, result.output
    # No --llm, so each transcript contributes its text under a per-id header.
    assert "### Transcript: t1" in captured["text"]
    assert "alpha" in captured["text"] and "beta" in captured["text"]
    assert captured["prompts"] == ["rank"]
    steps.assert_not_called()
    # Human reduce keeps stdout clean: only the aggregate prints, not each transcript.
    assert "RANKED" in result.output
    assert "alpha" not in result.output


def test_get_batch_reduce_feeds_map_outputs(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "a", "t2": "b"})

    def steps(api_key, prompts, *, transcript_id, model, max_tokens):
        return [{"prompt": prompts[-1], "output": f"JUDGED-{transcript_id}"}]

    _patch_chain_steps(mocker, steps)
    captured = {}

    def reduce(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        return "FINAL"

    _patch_reduce(mocker, reduce)
    result = runner.invoke(
        app, ["transcripts", "get", "--llm", "judge", "--llm-reduce", "rank"], input="t1\nt2\n"
    )
    assert result.exit_code == 0, result.output
    # The reduce sees the --llm output of each transcript, not the raw text.
    assert "JUDGED-t1" in captured["text"] and "JUDGED-t2" in captured["text"]
    assert "FINAL" in result.output


def test_get_batch_reduce_json_emits_reduce_record_and_per_id_records(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": "alpha"})
    _patch_reduce(mocker, lambda *a, **k: "RANKED")
    result = runner.invoke(
        app, ["transcripts", "get", "--llm-reduce", "rank", "--json"], input="t1\n"
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    reduce_records = [r for r in records if r.get("type") == "reduce"]
    assert len(reduce_records) == 1
    assert reduce_records[0]["output"] == "RANKED"
    assert reduce_records[0]["prompts"] == ["rank"]
    # JSON keeps the per-id stream too (unlike human reduce, which suppresses it).
    assert any(r.get("type") == "transcript" for r in records)


def test_get_batch_reduce_skips_when_nothing_to_reduce(mocker):
    config.set_api_key("default", "sk_live")
    _dispatch_by_id(mocker, {"t1": ""})  # empty transcript text
    reduce = mocker.patch("aai_cli.commands.transcripts.llm.run_chain", autospec=True)
    result = runner.invoke(app, ["transcripts", "get", "--llm-reduce", "rank"], input="t1\n")
    assert result.exit_code == 0, result.output
    reduce.assert_not_called()  # never fire a billable call over empty input
    assert "Nothing to reduce" in result.output
