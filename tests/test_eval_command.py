"""`assembly eval` behavior: WER scoring, rendering, flags, error paths.

The transcription boundary (`client.transcribe`) is mocked; datasets are real
temp manifests so the command exercises the loader end to end.
"""

import contextlib
import dataclasses
import json
import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aai_cli import config, eval_data
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _auth():
    config.set_api_key("default", "sk_live")


def _transcript(text):
    return SimpleNamespace(text=text)


def _write_wer_manifest(tmp_path):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"fake-audio")
    (tmp_path / "manifest.csv").write_text(
        "audio,text\na.wav,hello there\nb.wav,goodbye now\n", encoding="utf-8"
    )


def _mock_transcribe(mocker, results):
    return mocker.patch(
        "aai_cli.evaluate_exec.client.transcribe",
        autospec=True,
        side_effect=list(results),
    )


def _payload_of(result):
    return next(
        json.loads(line) for line in result.output.splitlines() if line.startswith('{"dataset"')
    )


def test_wer_table_with_per_file_and_pooled_scores(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    # File 1 is perfect; file 2 gets 1 of its 2 words wrong -> 50%; pooled 1/4 -> 25%.
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye cow")])
    result = runner.invoke(app, ["eval", "manifest.csv"])
    assert result.exit_code == 0
    assert "WER" in result.output
    assert "0.00%" in result.output
    assert "50.00%" in result.output
    assert "25.00%" in result.output
    assert "1 error / 4 words" in result.output  # pooled errors (and singular grammar)
    assert "1 errors" not in result.output
    assert "manifest.csv" in result.output
    assert "default model" in result.output


def test_summary_pluralizes_errors(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    # "bye cow" vs "goodbye now": two substitutions.
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("bye cow")])
    result = runner.invoke(app, ["eval", "manifest.csv"])
    assert "2 errors / 4 words" in result.output


def test_json_payload_shape(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye cow")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--json"])
    assert result.exit_code == 0
    payload = _payload_of(result)
    assert payload["dataset"] == "manifest.csv"
    assert payload["speech_model"] is None
    assert payload["items"] == 2
    assert payload["words"] == 4
    assert payload["errors"] == 1
    assert payload["wer"] == 0.25
    assert payload["rows"][0] == {"item": "a.wav", "words": 2, "errors": 0, "wer": 0.0}
    assert payload["rows"][1] == {"item": "b.wav", "words": 2, "errors": 1, "wer": 0.5}
    assert "failed" not in payload  # only present when a row failed


@pytest.mark.parametrize("model", ["universal-3-pro", "universal-2"])
def test_speech_model_flag_reaches_config_and_output(tmp_path, mocker, model):
    _auth()
    _write_wer_manifest(tmp_path)
    tx = _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--speech-model", model, "--json"])
    assert result.exit_code == 0
    assert _payload_of(result)["speech_model"] == model
    assert tx.call_args.kwargs["config"].speech_models == [model]


def test_no_speech_model_leaves_speech_models_unset(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    tx = _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    assert runner.invoke(app, ["eval", "manifest.csv"]).exit_code == 0
    assert tx.call_args.kwargs["config"].speech_models is None


@pytest.mark.parametrize("model", ["best", "nano", "slam-1", "universal"])
def test_legacy_models_are_a_usage_error(model):
    result = runner.invoke(app, ["eval", "manifest.csv", "--speech-model", model])
    assert result.exit_code == 2
    # CI forces color on (Rich under GITHUB_ACTIONS), interleaving style codes
    # mid-message, so assert on the color-free render (see test_help_rendering.py).
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "Invalid value for '--speech-model'" in plain


def test_speech_model_named_in_human_header(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--speech-model", "universal-3-pro"])
    assert "universal-3-pro" in result.output
    assert "default model" not in result.output


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_item_results_are_immutable():
    from aai_cli.evaluate_exec import _ItemResult

    result = _ItemResult(row={}, words=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(result, "words", None)


def test_missing_transcript_text_scores_as_all_deletions(tmp_path, mocker):
    _auth()
    (tmp_path / "a.wav").write_bytes(b"fake-audio")
    (tmp_path / "m.csv").write_text("audio,text\na.wav,hello there\n", encoding="utf-8")
    _mock_transcribe(mocker, [_transcript(None)])
    result = runner.invoke(app, ["eval", "m.csv", "--json"])
    assert _payload_of(result)["wer"] == 1.0


def _loaded_dataset():
    item = eval_data.EvalItem(item_id="a", audio="https://x/a.wav", reference="hello")
    return eval_data.EvalDataset(label="x", items=[item])


def test_loader_defaults(tmp_path, mocker):
    _auth()
    load = mocker.patch(
        "aai_cli.evaluate_exec.eval_data.load", autospec=True, return_value=_loaded_dataset()
    )
    _mock_transcribe(mocker, [_transcript("hello")])
    result = runner.invoke(app, ["eval", "org/ds"])
    assert result.exit_code == 0
    kwargs = load.call_args.kwargs
    assert kwargs["limit"] == 10
    assert kwargs["split"] is None and kwargs["subset"] is None
    assert kwargs["audio_column"] is None and kwargs["text_column"] is None


def test_explicit_loader_flags_pass_through(tmp_path, mocker):
    _auth()
    load = mocker.patch(
        "aai_cli.evaluate_exec.eval_data.load", autospec=True, return_value=_loaded_dataset()
    )
    _mock_transcribe(mocker, [_transcript("hello")])
    argv = [
        "eval", "org/ds", "--limit", "7", "--split", "validation", "--subset", "clean",
        "--audio-column", "wav", "--text-column", "ref",
    ]  # fmt: skip
    assert runner.invoke(app, argv).exit_code == 0
    kwargs = load.call_args.kwargs
    assert kwargs["limit"] == 7
    assert kwargs["split"] == "validation"
    assert kwargs["subset"] == "clean"
    assert kwargs["audio_column"] == "wav"
    assert kwargs["text_column"] == "ref"


@pytest.mark.parametrize("limit", ["0", "101"])
def test_limit_out_of_range_is_a_usage_error(limit):
    result = runner.invoke(app, ["eval", "org/ds", "--limit", limit])
    assert result.exit_code == 2
    assert "1<=x<=100" in result.output.replace(" ", "")


@pytest.mark.parametrize("limit", ["1", "100"])
def test_limit_bounds_are_inclusive(tmp_path, mocker, limit):
    _auth()
    mocker.patch(
        "aai_cli.evaluate_exec.eval_data.load", autospec=True, return_value=_loaded_dataset()
    )
    _mock_transcribe(mocker, [_transcript("hello")])
    assert runner.invoke(app, ["eval", "org/ds", "--limit", limit]).exit_code == 0


def test_progress_status_counts_items(tmp_path, mocker, monkeypatch):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    seen = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        seen.append(message)
        yield

    monkeypatch.setattr("aai_cli.evaluate_exec.output.status", fake_status)
    assert runner.invoke(app, ["eval", "manifest.csv"]).exit_code == 0
    assert seen == ["[1/2] Transcribing a.wav…", "[2/2] Transcribing b.wav…"]


def test_missing_manifest_is_a_usage_failure(tmp_path):
    _auth()
    result = runner.invoke(app, ["eval", "nope.csv"])
    assert result.exit_code == 2
    assert "Manifest not found" in result.output


def test_unauthenticated_exits_with_auth_code(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"fake-audio")
    (tmp_path / "m.csv").write_text("audio,text\na.wav,hello\n", encoding="utf-8")
    result = runner.invoke(app, ["eval", "m.csv"])
    assert result.exit_code == 4
