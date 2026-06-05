import json

import pytest

from aai_cli import config_builder as cb
from aai_cli.errors import UsageError


def _param_names(model_cls) -> set[str]:
    # assemblyai's transcription models are pydantic v1 (__fields__); the streaming.v3
    # models are pydantic v2 (model_fields). Accept either so tests track the SDK.
    return set(getattr(model_cls, "model_fields", None) or model_cls.__fields__)


def _dump(model) -> dict[str, object]:
    dumped = (
        model.model_dump(exclude_none=True)
        if hasattr(model, "model_dump")
        else model.dict(exclude_none=True)  # pydantic v1 fallback
    )
    return dict(dumped)


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


def test_parse_config_overrides_accepts_none():
    assert cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, None) == {}


def test_transcribe_config_layer_precedence(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"speaker_labels": False, "speakers_expected": 5}))
    tc = cb.construct_transcription_config(
        cb.merge_transcribe_config(
            flags={"speaker_labels": True},  # flag beats file
            overrides=["speakers_expected=3"],  # --config beats file
            config_file=str(cfg),
        )
    )
    assert tc.speaker_labels is True
    assert tc.raw.speakers_expected == 3


def test_transcribe_config_ignores_unset_flags():
    tc = cb.construct_transcription_config(
        cb.merge_transcribe_config(flags={"speaker_labels": None})
    )
    assert tc.speaker_labels is None  # None means "not set", does not override


def test_load_config_file_rejects_non_object(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(UsageError):
        cb.load_config_file(bad, cb.TRANSCRIBE_FIELDS)


def test_split_csv():
    assert cb.split_csv("a, b ,c") == ["a", "b", "c"]
    assert cb.split_csv(None) is None
    assert cb.split_csv("") is None


def test_parse_auth_header():
    assert cb.parse_auth_header("Authorization:Bearer x") == ("Authorization", "Bearer x")
    assert cb.parse_auth_header(None) is None
    with pytest.raises(UsageError):
        cb.parse_auth_header("no-colon")


def test_load_custom_spelling(tmp_path):
    p = tmp_path / "spell.json"
    p.write_text('{"AssemblyAI": ["assembly ai", "assemblyai"]}')
    assert cb.load_custom_spelling(str(p)) == {"AssemblyAI": ["assembly ai", "assemblyai"]}


def test_translation_request_shape():
    su = cb.translation_request(["es", "fr"])
    # target languages must be reachable from the payload regardless of dict/obj form.
    assert "es" in json.dumps(su, default=lambda o: getattr(o, "__dict__", str(o)))


def test_transcribe_config_with_translate_payload():
    # The SDK must accept the translation payload for speech_understanding without raising.
    tc = cb.construct_transcription_config(
        cb.merge_transcribe_config(
            flags={"speech_understanding": cb.translation_request(["es", "fr"])},
            overrides=[],
            config_file=None,
        )
    )
    assert "es" in json.dumps(
        tc.raw.speech_understanding, default=lambda o: getattr(o, "__dict__", str(o))
    )


def test_streaming_params_minimal():
    sp = cb.construct_streaming_params(
        cb.merge_streaming_params(
            flags={"sample_rate": 16000, "speech_model": "universal_streaming_multilingual"},
            overrides=["max_turn_silence=400"],
            config_file=None,
        )
    )
    assert sp.sample_rate == 16000
    assert sp.max_turn_silence == 400


@pytest.mark.parametrize(
    ("field", "raw", "expected", "extra"),
    [
        ("punctuate", "false", False, []),
        ("multichannel", "true", True, []),
        ("audio_start_from", "1500", 1500, []),
        ("temperature", "0.2", 0.2, []),
        # summary_type is only applied by the SDK when summarization is enabled.
        ("summary_type", "bullets", "bullets", ["summarization=true"]),
        ("keyterms_prompt", "a,b", ["a", "b"], []),
    ],
)
def test_transcribe_field_coercion_matrix(field, raw, expected, extra):
    tc = cb.construct_transcription_config(
        cb.merge_transcribe_config(flags={}, overrides=[f"{field}={raw}", *extra], config_file=None)
    )
    assert getattr(tc.raw, field) == expected


@pytest.mark.parametrize("field", sorted(cb.STREAM_FIELDS))
def test_every_stream_field_is_a_valid_param(field):
    # Each declared field must be a real StreamingParameters attribute.
    from assemblyai.streaming.v3 import StreamingParameters

    assert field in _param_names(StreamingParameters)


@pytest.mark.parametrize("field", sorted(cb.TRANSCRIBE_FIELDS))
def test_every_transcribe_field_is_a_valid_param(field):
    # Each declared field must be a real TranscriptionConfig request attribute.
    import assemblyai as aai

    raw_cls = type(aai.TranscriptionConfig().raw)
    assert field in _param_names(raw_cls)


def test_merge_transcribe_config_returns_kwargs_dict():
    from aai_cli import config_builder

    merged = config_builder.merge_transcribe_config(
        flags={"speaker_labels": True, "language_code": None},
        overrides=["sentiment_analysis=true"],
        config_file=None,
    )
    assert merged == {"speaker_labels": True, "sentiment_analysis": True}


def test_construct_transcribe_config_from_merged():
    import assemblyai as aai

    from aai_cli import config_builder

    tc = config_builder.construct_transcription_config({"speaker_labels": True})
    assert isinstance(tc, aai.TranscriptionConfig)
    # _dump may include SDK-internal keys; assert the field we set is present and on.
    assert _dump(tc.raw)["speaker_labels"] is True


def test_merge_streaming_params_coerces_speech_model_enum():
    from assemblyai.streaming.v3 import SpeechModel

    from aai_cli import config_builder

    merged = config_builder.merge_streaming_params(
        flags={"speech_model": "universal-streaming-multilingual", "sample_rate": 16000},
        overrides=[],
        config_file=None,
    )
    assert merged["speech_model"] is SpeechModel.universal_streaming_multilingual
    assert merged["sample_rate"] == 16000


# The full field -> coercion-kind mapping is frozen here. Kinds are derived from the
# SDK model annotations, but the curated field set and every resulting kind must stay
# exactly as below; this guards the whole table, not just the fields sampled above.
_EXPECTED_TRANSCRIBE_COERCE = {
    "language_code": "str",
    "language_codes": "list",
    "punctuate": "bool",
    "format_text": "bool",
    "dual_channel": "bool",
    "multichannel": "bool",
    "webhook_url": "str",
    "webhook_auth_header_name": "str",
    "webhook_auth_header_value": "str",
    "audio_start_from": "int",
    "audio_end_at": "int",
    "word_boost": "list",
    "boost_param": "str",
    "filter_profanity": "bool",
    "redact_pii": "bool",
    "redact_pii_audio": "bool",
    "redact_pii_audio_quality": "str",
    "redact_pii_audio_options": "json",
    "redact_pii_policies": "list",
    "redact_pii_sub": "str",
    "redact_pii_return_unredacted": "bool",
    "speaker_labels": "bool",
    "speakers_expected": "int",
    "speaker_options": "json",
    "content_safety": "bool",
    "content_safety_confidence": "int",
    "iab_categories": "bool",
    "custom_spelling": "json",
    "disfluencies": "bool",
    "sentiment_analysis": "bool",
    "auto_chapters": "bool",
    "entity_detection": "bool",
    "summarization": "bool",
    "summary_model": "str",
    "summary_type": "str",
    "auto_highlights": "bool",
    "language_detection": "bool",
    "language_confidence_threshold": "float",
    "language_detection_options": "json",
    "speech_threshold": "float",
    "speech_model": "str",
    "speech_models": "list",
    "prompt": "str",
    "temperature": "float",
    "remove_audio_tags": "str",
    "keyterms_prompt": "list",
    "keyterms_prompt_options": "json",
    "speech_understanding": "json",
    "domain": "str",
}

_EXPECTED_STREAM_COERCE = {
    "end_of_turn_confidence_threshold": "float",
    "min_end_of_turn_silence_when_confident": "int",
    "min_turn_silence": "int",
    "max_turn_silence": "int",
    "vad_threshold": "float",
    "format_turns": "bool",
    "keyterms_prompt": "list",
    "filter_profanity": "bool",
    "prompt": "str",
    "sample_rate": "int",
    "encoding": "str",
    "speech_model": "str",
    "language_detection": "bool",
    "domain": "str",
    "inactivity_timeout": "int",
    "webhook_url": "str",
    "webhook_auth_header_name": "str",
    "webhook_auth_header_value": "str",
    "llm_gateway": "json",
    "speaker_labels": "bool",
    "max_speakers": "int",
    "voice_focus": "str",
    "voice_focus_threshold": "float",
    "noise_suppression_model": "str",
    "noise_suppression_threshold": "float",
    "continuous_partials": "bool",
    "customer_support_audio_capture": "bool",
    "include_partial_turns": "bool",
    "redact_pii": "bool",
    "redact_pii_policies": "list",
    "redact_pii_sub": "str",
}


def test_transcribe_coerce_table_matches_frozen_mapping():
    assert cb.TRANSCRIBE_COERCE == _EXPECTED_TRANSCRIBE_COERCE


def test_stream_coerce_table_matches_frozen_mapping():
    assert cb.STREAM_COERCE == _EXPECTED_STREAM_COERCE


def test_coerce_bad_float_and_json_raise_usageerror():
    with pytest.raises(UsageError) as exc:
        cb.coerce_value("temperature", "hot")
    assert "number" in str(exc.value)
    with pytest.raises(UsageError) as exc:
        cb.coerce_value("custom_spelling", "{not json")
    assert "JSON" in str(exc.value)


def test_load_config_file_missing_file_raises_usageerror(tmp_path):
    with pytest.raises(UsageError) as exc:
        cb.load_config_file(tmp_path / "nope.json", cb.TRANSCRIBE_FIELDS)
    assert "not found" in str(exc.value)


def test_load_config_file_invalid_json_raises_usageerror(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid")
    with pytest.raises(UsageError) as exc:
        cb.load_config_file(bad, cb.TRANSCRIBE_FIELDS)
    assert "not valid JSON" in str(exc.value)


def test_load_config_file_unknown_field_lists_valid(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"bogus_field": 1}))
    with pytest.raises(UsageError) as exc:
        cb.load_config_file(cfg, cb.TRANSCRIBE_FIELDS)
    assert "bogus_field" in str(exc.value)
    assert "speaker_labels" in str(exc.value)  # error lists valid fields


def test_construct_transcription_config_wraps_sdk_error_as_usageerror():
    # An unknown kwarg makes the SDK constructor raise; it must surface as a usage error.
    with pytest.raises(UsageError) as exc:
        cb.construct_transcription_config({"totally_unknown_kwarg": 1})
    assert "Invalid transcription config" in str(exc.value)


def test_merge_streaming_params_invalid_speech_model_raises_usageerror():
    with pytest.raises(UsageError) as exc:
        cb.merge_streaming_params(
            flags={"speech_model": "not-a-real-model"}, overrides=[], config_file=None
        )
    assert "Invalid streaming config" in str(exc.value)


def test_construct_streaming_params_wraps_sdk_error_as_usageerror():
    with pytest.raises(UsageError) as exc:
        cb.construct_streaming_params({"sample_rate": "not-an-int"})
    assert "Invalid streaming config" in str(exc.value)


def test_load_custom_spelling_missing_file_raises_usageerror(tmp_path):
    with pytest.raises(UsageError) as exc:
        cb.load_custom_spelling(str(tmp_path / "nope.json"))
    assert "not found" in str(exc.value)


def test_load_custom_spelling_invalid_json_raises_usageerror(tmp_path):
    bad = tmp_path / "spell.json"
    bad.write_text("{not json")
    with pytest.raises(UsageError) as exc:
        cb.load_custom_spelling(str(bad))
    assert "not valid JSON" in str(exc.value)


def test_load_custom_spelling_rejects_non_object(tmp_path):
    p = tmp_path / "spell.json"
    p.write_text('["assembly ai"]')
    with pytest.raises(UsageError) as exc:
        cb.load_custom_spelling(str(p))
    assert "JSON object" in str(exc.value)


def test_derive_kind_multi_type_union_without_str_falls_back_to_json():
    import typing

    # A genuine multi-type union with no str-like member -> a raw JSON value is accepted.
    # typing.Union (not int|float) is required: the code matches `get_origin is typing.Union`.
    assert cb._derive_kind(typing.Union[int, float]) == "json"  # noqa: UP007


def test_derive_kind_dict_origin_is_json():
    assert cb._derive_kind(dict[str, int]) == "json"


def test_coerce_table_unknown_field_defaults_to_str():
    # A curated name the SDK model doesn't expose passes through as a string
    # rather than crashing at import time.
    class Empty:  # no model_fields / __fields__ -> no known annotations
        pass

    table = cb._coerce_table(Empty, ("phantom_field",))
    assert table == {"phantom_field": "str"}
