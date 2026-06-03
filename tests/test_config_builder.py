import json

import pytest

from assemblyai_cli import config_builder as cb
from assemblyai_cli.errors import UsageError


def test_coerce_bool_int_float_list():
    assert cb.coerce_value("speaker_labels", "true") is True
    assert cb.coerce_value("speaker_labels", "false") is False
    assert cb.coerce_value("speakers_expected", "2") == 2
    assert cb.coerce_value("speech_threshold", "0.5") == 0.5
    assert cb.coerce_value("redact_pii_policies", "person_name, phone_number") == [
        "person_name",
        "phone_number",
    ]


def test_coerce_str_passthrough_and_json():
    assert cb.coerce_value("language_code", "en_us") == "en_us"
    assert cb.coerce_value("custom_spelling", '{"AssemblyAI": ["assembly ai"]}') == {
        "AssemblyAI": ["assembly ai"]
    }


def test_coerce_bad_bool_and_int_raise_usageerror():
    with pytest.raises(UsageError):
        cb.coerce_value("speaker_labels", "maybe")
    with pytest.raises(UsageError):
        cb.coerce_value("speakers_expected", "two")


def test_parse_config_overrides_unknown_key_lists_valid():
    with pytest.raises(UsageError) as exc:
        cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, ["not_a_field=1"])
    assert "not_a_field" in str(exc.value)
    assert "speaker_labels" in str(exc.value)  # error lists valid fields


def test_parse_config_overrides_requires_equals():
    with pytest.raises(UsageError):
        cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, ["speaker_labels"])


def test_build_transcription_config_layer_precedence(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"speaker_labels": False, "speakers_expected": 5}))
    tc = cb.build_transcription_config(
        flags={"speaker_labels": True},  # flag beats file
        overrides=["speakers_expected=3"],  # --config beats file
        config_file=str(cfg),
    )
    assert tc.speaker_labels is True
    assert tc.raw.speakers_expected == 3


def test_build_transcription_config_ignores_unset_flags():
    tc = cb.build_transcription_config(
        flags={"speaker_labels": None}, overrides=[], config_file=None
    )
    assert tc.speaker_labels is None  # None means "not set", does not override


def test_load_config_file_rejects_non_object(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(UsageError):
        cb.load_config_file(bad, cb.TRANSCRIBE_FIELDS)
