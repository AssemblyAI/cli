"""`assembly eval` behavior: WER/DER scoring, rendering, flags, error paths.

The transcription boundary (`client.transcribe`) is mocked; datasets are real
temp manifests so the command exercises the loader end to end.
"""

import contextlib
import dataclasses
import json
from types import SimpleNamespace

import assemblyai as aai
import pytest
from typer.testing import CliRunner

from aai_cli import config, eval_data
from aai_cli.errors import APIError
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _auth():
    config.set_api_key("default", "sk_live")


def _transcript(text, utterances=None):
    return SimpleNamespace(text=text, utterances=utterances)


def _write_wer_manifest(tmp_path):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"fake-audio")
    (tmp_path / "manifest.csv").write_text(
        "audio,text\na.wav,hello there\nb.wav,goodbye now\n", encoding="utf-8"
    )


def _write_speaker_manifest(tmp_path, *, with_text=False):
    (tmp_path / "a.wav").write_bytes(b"fake-audio")
    row = {
        "audio": "a.wav",
        "speakers": ["alice", "bob"],
        "timestamps_start": [0.0, 10.0],
        "timestamps_end": [10.0, 20.0],
    }
    if with_text:
        row["text"] = "hello there"
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row), encoding="utf-8")


def _diarized_utterances():
    # Against the manifest's alice 0-10s / bob 10-20s reference: 2s of confusion
    # around the 10s boundary. The default 1s collar forgives 10-10.5s and trims
    # 2s of boundary speech from the 20s total -> DER 1.5/18 (see test_der.py).
    return [
        SimpleNamespace(speaker="A", start=0, end=12000),
        SimpleNamespace(speaker="B", start=12000, end=20000),
    ]


def _mock_transcribe(mocker, results):
    return mocker.patch(
        "aai_cli.commands.evaluate.client.transcribe",
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
    assert "DER" not in result.output


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
    assert "der" not in payload
    assert payload["rows"][0] == {"item": "a.wav", "words": 2, "errors": 0, "wer": 0.0}
    assert payload["rows"][1] == {"item": "b.wav", "words": 2, "errors": 1, "wer": 0.5}


def test_speech_model_flag_reaches_config_and_output(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    tx = _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--speech-model", "universal", "--json"])
    assert result.exit_code == 0
    assert _payload_of(result)["speech_model"] == "universal"
    tx_config = tx.call_args.kwargs["config"]
    assert tx_config.speech_model == aai.SpeechModel.universal
    assert tx_config.speaker_labels is None  # not requested -> omitted, not False


def test_speech_model_named_in_human_header(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--speech-model", "universal"])
    assert "universal" in result.output
    assert "default model" not in result.output


def test_speaker_labels_scores_der_only_when_dataset_has_no_text(tmp_path, mocker):
    _auth()
    _write_speaker_manifest(tmp_path)
    tx = _mock_transcribe(mocker, [_transcript("ignored", _diarized_utterances())])
    result = runner.invoke(app, ["eval", "manifest.jsonl", "--speaker-labels"])
    assert result.exit_code == 0
    assert tx.call_args.kwargs["config"].speaker_labels is True
    assert "DER" in result.output
    assert "8.33%" in result.output  # 1.5/18 under the default 1s collar
    assert "WER" not in result.output
    assert "WORDS" not in result.output


def test_speaker_labels_with_text_scores_both_metrics(tmp_path, mocker):
    _auth()
    _write_speaker_manifest(tmp_path, with_text=True)
    _mock_transcribe(mocker, [_transcript("hello there", _diarized_utterances())])
    result = runner.invoke(app, ["eval", "manifest.jsonl", "--speaker-labels", "--json"])
    assert result.exit_code == 0
    payload = _payload_of(result)
    assert payload["wer"] == 0.0
    assert payload["der"] == 1.5 / 18  # default 1s collar
    assert payload["rows"][0]["wer"] == 0.0
    assert payload["rows"][0]["der"] == 1.5 / 18


def test_both_metrics_render_in_one_table(tmp_path, mocker):
    _auth()
    _write_speaker_manifest(tmp_path, with_text=True)
    _mock_transcribe(mocker, [_transcript("hello there", _diarized_utterances())])
    result = runner.invoke(app, ["eval", "manifest.jsonl", "--speaker-labels"])
    assert "WER" in result.output
    assert "DER" in result.output
    assert "8.33%" in result.output


def test_der_counts_leading_speech_the_model_missed(tmp_path, mocker):
    _auth()
    (tmp_path / "a.wav").write_bytes(b"fake-audio")
    row = {
        "audio": "a.wav",
        "speakers": ["alice"],
        "timestamps_start": [0.0],
        "timestamps_end": [10.0],
    }
    (tmp_path / "m.jsonl").write_text(json.dumps(row), encoding="utf-8")
    # The hypothesis starts 1000ms in (1.0s exactly after ms→s conversion):
    # 1s missed over 10s of reference speech, scored without collar forgiveness.
    utterances = [SimpleNamespace(speaker="A", start=1000, end=10000)]
    _mock_transcribe(mocker, [_transcript("x", utterances)])
    result = runner.invoke(app, ["eval", "m.jsonl", "--speaker-labels", "--collar", "0", "--json"])
    assert _payload_of(result)["der"] == 0.1


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_item_results_are_immutable():
    from aai_cli.commands.evaluate import _ItemResult

    result = _ItemResult(row={}, words=None, speakers=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(result, "words", None)


def test_collar_flag_loosens_der(tmp_path, mocker):
    _auth()
    _write_speaker_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("ignored", _diarized_utterances())])
    result = runner.invoke(
        app, ["eval", "manifest.jsonl", "--speaker-labels", "--collar", "4.0", "--json"]
    )
    assert _payload_of(result)["der"] == 0.0


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
        "aai_cli.commands.evaluate.eval_data.load", autospec=True, return_value=_loaded_dataset()
    )
    _mock_transcribe(mocker, [_transcript("hello")])
    result = runner.invoke(app, ["eval", "org/ds"])
    assert result.exit_code == 0
    kwargs = load.call_args.kwargs
    assert kwargs["limit"] == 10
    assert kwargs["with_speakers"] is False
    assert kwargs["split"] is None and kwargs["subset"] is None
    assert kwargs["audio_column"] is None and kwargs["text_column"] is None


def test_explicit_loader_flags_pass_through(tmp_path, mocker):
    _auth()
    load = mocker.patch(
        "aai_cli.commands.evaluate.eval_data.load", autospec=True, return_value=_loaded_dataset()
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
        "aai_cli.commands.evaluate.eval_data.load", autospec=True, return_value=_loaded_dataset()
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

    monkeypatch.setattr("aai_cli.commands.evaluate.output.status", fake_status)
    assert runner.invoke(app, ["eval", "manifest.csv"]).exit_code == 0
    assert seen == ["[1/2] Transcribing a.wav…", "[2/2] Transcribing b.wav…"]


def test_api_error_mid_run_fails_cleanly(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), APIError("rate limited")])
    result = runner.invoke(app, ["eval", "manifest.csv"])
    assert result.exit_code == 1
    assert "rate limited" in result.output


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
