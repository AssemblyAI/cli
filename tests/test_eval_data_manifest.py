"""Local-manifest loading for `assembly eval` (`aai_cli.eval_data`).

Runs against real temp files; the Hugging Face paths live in
test_eval_data_hf.py.
"""

import dataclasses
import json

import pytest

from aai_cli import eval_data
from aai_cli.errors import CLIError, UsageError

# ---------------------------------------------------------------- local manifests


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_loaded_dataset_and_items_are_immutable(tmp_path):
    _write_audio(tmp_path, "a.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\na.wav,hello\n", encoding="utf-8")
    data = eval_data.load(str(manifest), limit=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(data, "label", "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(data.items[0], "reference", "x")


def _write_audio(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"fake-audio")


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_csv_manifest_loads_items(tmp_path):
    _write_audio(tmp_path, "a.wav", "b.wav")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("audio,text\na.wav,hello there\nb.wav,goodbye now\n", encoding="utf-8")
    data = eval_data.load(str(manifest), limit=10)
    assert data.label == "manifest.csv"
    assert [item.item_id for item in data.items] == ["a.wav", "b.wav"]
    assert data.items[0].audio == str(tmp_path / "a.wav")
    assert data.items[0].reference == "hello there"


def test_manifest_respects_limit(tmp_path):
    _write_audio(tmp_path, "a.wav", "b.wav", "c.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\na.wav,x\nb.wav,y\nc.wav,z\n", encoding="utf-8")
    data = eval_data.load(str(manifest), limit=2)
    assert [item.item_id for item in data.items] == ["a.wav", "b.wav"]


def test_jsonl_manifest_with_url_audio_and_nemo_columns(tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(
        manifest,
        [{"audio_filepath": "https://cdn.example/a.mp3", "transcript": "hello world"}],
    )
    data = eval_data.load(str(manifest), limit=10)
    # URLs pass through untouched (no local-file check) and NeMo-style column
    # names auto-detect.
    assert data.items[0].audio == "https://cdn.example/a.mp3"
    assert data.items[0].item_id == "a.mp3"
    assert data.items[0].reference == "hello world"


def test_jsonl_manifest_skips_blank_lines(tmp_path):
    _write_audio(tmp_path, "a.wav", "b.wav")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"audio": "a.wav", "text": "x"}\n\n{"audio": "b.wav", "text": "y"}\n', encoding="utf-8"
    )
    assert len(eval_data.load(str(manifest), limit=10).items) == 2


def test_manifest_missing_file_errors_before_any_network(tmp_path):
    with pytest.raises(CLIError) as exc:
        eval_data.load(str(tmp_path / "missing.csv"), limit=10)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    assert "missing.csv" in exc.value.message


def test_manifest_audio_file_missing_names_resolved_path(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\nnope.wav,hello\n", encoding="utf-8")
    with pytest.raises(CLIError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    assert str(tmp_path / "nope.wav") in exc.value.message
    assert exc.value.suggestion is not None and str(tmp_path) in exc.value.suggestion


def test_manifest_with_unsupported_suffix_rejected(tmp_path):
    # A .parquet (or any non-.csv/.jsonl file) must name the real constraint, not
    # fail as "line 1 is not valid JSON" from the JSONL fallback parser.
    manifest = tmp_path / "data.parquet"
    manifest.write_bytes(b"PAR1\x00not-json")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert "Manifests must be .csv or .jsonl" in exc.value.message
    assert "data.parquet" in exc.value.message
    assert "not valid JSON" not in exc.value.message


def test_manifest_with_no_recognized_audio_column_uses_an_article(tmp_path):
    # Grammar: "an audio column", not "a audio column".
    manifest = tmp_path / "m.csv"
    manifest.write_text("wav,text\na.wav,hello\n", encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert "Could not find an audio column" in exc.value.message
    assert exc.value.suggestion is not None and "--audio-column" in exc.value.suggestion


def test_manifest_row_without_audio_value_reports_row_number(tmp_path):
    _write_audio(tmp_path, "a.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\na.wav,hello\n,world\n", encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert "row 2" in exc.value.message


def test_manifest_with_no_recognized_text_column_suggests_flag(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,ref\na.wav,hello\n", encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    # The message names the conventional column (the first auto-detect candidate).
    assert "Could not find a text column" in exc.value.message
    assert "audio, ref" in exc.value.message
    assert exc.value.suggestion is not None and "--text-column" in exc.value.suggestion


def test_manifest_explicit_text_column_is_used(tmp_path):
    _write_audio(tmp_path, "a.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,ref\na.wav,hello there\n", encoding="utf-8")
    data = eval_data.load(str(manifest), text_column="ref", limit=10)
    assert data.items[0].reference == "hello there"


def test_manifest_explicit_text_column_missing_errors(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\na.wav,hello\n", encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), text_column="ref", limit=10)
    assert "'ref'" in exc.value.message


def test_manifest_empty_reference_rejected(tmp_path):
    _write_audio(tmp_path, "a.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\na.wav,...\n", encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert "empty reference" in exc.value.message


def test_manifest_with_no_rows_errors(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\n", encoding="utf-8")
    with pytest.raises(UsageError, match="no rows"):
        eval_data.load(str(manifest), limit=10)


def test_jsonl_invalid_json_line_reports_line_number(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"audio": "a.wav", "text": "x"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert "line 2" in exc.value.message


def test_jsonl_non_object_line_rejected(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('["audio", "text"]\n', encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        eval_data.load(str(manifest), limit=10)
    assert "line 1 is not a JSON object" in exc.value.message


def test_split_and_subset_rejected_for_manifests(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio,text\na.wav,hello\n", encoding="utf-8")
    with pytest.raises(UsageError, match="local manifests"):
        eval_data.load(str(manifest), split="test", limit=10)
    with pytest.raises(UsageError, match="local manifests"):
        eval_data.load(str(manifest), subset="default", limit=10)
