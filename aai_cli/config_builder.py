from __future__ import annotations

import enum
import json
import typing
from pathlib import Path

import assemblyai as aai
from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

from aai_cli.errors import UsageError

# The curated set of user-settable config fields per command. This is the authoritative
# allow-list (deliberately a subset of the SDK models — e.g. output-only and internal
# fields are excluded). The coercion KIND for each field is derived from the SDK model
# annotation in `_coerce_table`, so only the names are maintained here, not their types.
TRANSCRIBE_FIELD_NAMES: tuple[str, ...] = (
    "language_code",
    "language_codes",
    "punctuate",
    "format_text",
    "dual_channel",
    "multichannel",
    "webhook_url",
    "webhook_auth_header_name",
    "webhook_auth_header_value",
    "audio_start_from",
    "audio_end_at",
    "word_boost",
    "boost_param",
    "filter_profanity",
    "redact_pii",
    "redact_pii_audio",
    "redact_pii_audio_quality",
    "redact_pii_audio_options",
    "redact_pii_policies",
    "redact_pii_sub",
    "redact_pii_return_unredacted",
    "speaker_labels",
    "speakers_expected",
    "speaker_options",
    "content_safety",
    "content_safety_confidence",
    "iab_categories",
    "custom_spelling",
    "disfluencies",
    "sentiment_analysis",
    "auto_chapters",
    "entity_detection",
    "summarization",
    "summary_model",
    "summary_type",
    "auto_highlights",
    "language_detection",
    "language_confidence_threshold",
    "language_detection_options",
    "speech_threshold",
    "speech_model",
    "speech_models",
    "prompt",
    "temperature",
    "remove_audio_tags",
    "keyterms_prompt",
    "keyterms_prompt_options",
    "speech_understanding",
    "domain",
)

STREAM_FIELD_NAMES: tuple[str, ...] = (
    "end_of_turn_confidence_threshold",
    "min_end_of_turn_silence_when_confident",
    "min_turn_silence",
    "max_turn_silence",
    "vad_threshold",
    "format_turns",
    "keyterms_prompt",
    "filter_profanity",
    "prompt",
    "sample_rate",
    "encoding",
    "speech_model",
    "language_detection",
    "domain",
    "inactivity_timeout",
    "webhook_url",
    "webhook_auth_header_name",
    "webhook_auth_header_value",
    "llm_gateway",
    "speaker_labels",
    "max_speakers",
    "voice_focus",
    "voice_focus_threshold",
    "noise_suppression_model",
    "noise_suppression_threshold",
    "continuous_partials",
    "customer_support_audio_capture",
    "include_partial_turns",
    "redact_pii",
    "redact_pii_policies",
    "redact_pii_sub",
)

# Fields whose CLI input differs from the SDK annotation. `custom_spelling` is typed as a
# list-of-dicts on the model, but the CLI accepts the JSON object form directly.
_KIND_OVERRIDES: dict[str, str] = {"custom_spelling": "json"}


def _field_annotations(model_cls: type) -> dict[str, object]:
    """Map field name -> type annotation for a pydantic model (v2 or v1)."""
    model_fields = getattr(model_cls, "model_fields", None)
    if model_fields:  # pydantic v2
        return {name: field.annotation for name, field in model_fields.items()}
    legacy_fields = getattr(model_cls, "__fields__", {})  # pydantic v1
    return {name: field.outer_type_ for name, field in legacy_fields.items()}


def _derive_kind(annotation: object) -> str:
    """Map an SDK field annotation to a coercion kind for `coerce_value`."""
    if typing.get_origin(annotation) is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]  # unwrap Optional[X]
        # A genuine multi-type union (e.g. Union[str, LanguageCode]): a string is
        # always an acceptable input, otherwise fall back to a raw JSON value.
        elif any(_is_str_like(a) for a in non_none):
            return "str"
        else:
            return "json"
    origin = typing.get_origin(annotation)
    if origin in (list, set, tuple):
        return "list"
    if origin is dict:
        return "json"
    if isinstance(annotation, type):
        if issubclass(annotation, bool):  # before int: bool is an int subclass
            return "bool"
        if issubclass(annotation, enum.Enum):
            return "str"
        if issubclass(annotation, int):
            return "int"
        if issubclass(annotation, float):
            return "float"
        if issubclass(annotation, str):
            return "str"
    return "json"  # pydantic submodels and anything else: accept a raw JSON value


def _is_str_like(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, str | enum.Enum)


def _coerce_table(model_cls: type, names: tuple[str, ...]) -> dict[str, str]:
    """Build a field -> coercion-kind table for `names`, deriving kinds from the model."""
    annotations = _field_annotations(model_cls)
    table: dict[str, str] = {}
    for name in names:
        if name in _KIND_OVERRIDES:
            table[name] = _KIND_OVERRIDES[name]
        elif name in annotations:
            table[name] = _derive_kind(annotations[name])
        else:
            # A curated name the SDK no longer exposes: pass through as a string and let
            # the model constructor reject it, rather than crashing at import.
            table[name] = "str"
    return table


# field name -> coercion kind for --config/--config-file string values. The transcribe
# fields live on the request model behind TranscriptionConfig.raw.
TRANSCRIBE_COERCE: dict[str, str] = _coerce_table(
    type(aai.TranscriptionConfig().raw), TRANSCRIBE_FIELD_NAMES
)
STREAM_COERCE: dict[str, str] = _coerce_table(StreamingParameters, STREAM_FIELD_NAMES)

TRANSCRIBE_FIELDS = TRANSCRIBE_COERCE
STREAM_FIELDS = STREAM_COERCE

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def coerce_value(field: str, raw: str) -> object:
    """Coerce a string --config value to the type expected by `field`."""
    kind = TRANSCRIBE_COERCE.get(field) or STREAM_COERCE.get(field, "str")
    if kind == "bool":
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise UsageError(f"{field} expects a boolean (true/false), got {raw!r}.")
    if kind == "int":
        try:
            return int(raw)
        except ValueError as exc:
            raise UsageError(f"{field} expects an integer, got {raw!r}.") from exc
    if kind == "float":
        try:
            return float(raw)
        except ValueError as exc:
            raise UsageError(f"{field} expects a number, got {raw!r}.") from exc
    if kind == "list":
        return [part.strip() for part in raw.split(",") if part.strip()]
    if kind == "json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{field} expects a JSON value, got {raw!r}.") from exc
    return raw


def parse_config_overrides(fields: dict[str, str], pairs: list[str]) -> dict[str, object]:
    """Parse repeated KEY=VALUE strings into a coerced, validated dict."""
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise UsageError(f"--config expects KEY=VALUE, got {pair!r}.")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key not in fields:
            valid = ", ".join(sorted(fields))
            raise UsageError(f"Unknown config field {key!r}. Valid fields: {valid}.")
        out[key] = coerce_value(key, raw)
    return out


def load_config_file(path: str | Path, fields: dict[str, str]) -> dict[str, object]:
    """Load a JSON config file and validate its keys against `fields`."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise UsageError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"Config file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("Config file must contain a JSON object.")
    unknown = [k for k in data if k not in fields]
    if unknown:
        valid = ", ".join(sorted(fields))
        raise UsageError(f"Unknown config field(s) {unknown}. Valid fields: {valid}.")
    return data


def _merge(
    fields: dict[str, str],
    flags: dict[str, object],
    overrides: list[str],
    config_file: str | None,
) -> dict[str, object]:
    data: dict[str, object] = {}
    if config_file:
        data.update(load_config_file(config_file, fields))
    data.update(parse_config_overrides(fields, overrides))
    data.update({k: v for k, v in flags.items() if v is not None})
    return data


def merge_transcribe_config(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> dict[str, object]:
    """Merge config-file + --config overrides + curated flags into a kwargs dict."""
    return _merge(TRANSCRIBE_FIELDS, flags, overrides, config_file)


def construct_transcription_config(merged: dict[str, typing.Any]) -> aai.TranscriptionConfig:
    """Build a TranscriptionConfig from a merged kwargs dict, surfacing errors as usage."""
    try:
        return aai.TranscriptionConfig(**merged)
    except UsageError:
        raise
    except Exception as exc:  # surface SDK validation as a usage error
        raise UsageError(f"Invalid transcription config: {exc}") from exc


def build_transcription_config(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> aai.TranscriptionConfig:
    return construct_transcription_config(
        merge_transcribe_config(flags=flags, overrides=overrides, config_file=config_file)
    )


def merge_streaming_params(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> dict[str, object]:
    """Merge streaming config into a kwargs dict, coercing speech_model to a SpeechModel."""
    merged = _merge(STREAM_FIELDS, flags, overrides, config_file)
    raw_model = merged.get("speech_model")
    if isinstance(raw_model, str):
        try:
            merged["speech_model"] = SpeechModel[raw_model]
        except KeyError:
            try:
                merged["speech_model"] = SpeechModel(raw_model)
            except ValueError as exc:
                raise UsageError(f"Invalid streaming config: {exc}") from exc
    return merged


def construct_streaming_params(merged: dict[str, typing.Any]) -> StreamingParameters:
    """Build StreamingParameters from a merged kwargs dict, surfacing errors as usage."""
    try:
        return StreamingParameters(**merged)
    except UsageError:
        raise
    except Exception as exc:
        raise UsageError(f"Invalid streaming config: {exc}") from exc


def build_streaming_params(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> StreamingParameters:
    return construct_streaming_params(
        merge_streaming_params(flags=flags, overrides=overrides, config_file=config_file)
    )


def split_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated flag value into a list, or None if empty."""
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def parse_auth_header(value: str | None) -> tuple[str, str] | None:
    """Parse a `NAME:VALUE` webhook auth header flag."""
    if value is None:
        return None
    if ":" not in value:
        raise UsageError("--webhook-auth-header expects NAME:VALUE.")
    name, header_value = value.split(":", 1)
    return name.strip(), header_value.strip()


def load_custom_spelling(path: str) -> dict[str, object]:
    """Load a custom-spelling JSON map (e.g. {"AssemblyAI": ["assembly ai"]})."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise UsageError(f"Custom spelling file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"Custom spelling file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("Custom spelling file must contain a JSON object.")
    return data


def translation_request(languages: list[str]) -> dict[str, object]:
    """Build a Speech-Understanding translation payload for `speech_understanding`."""
    return {"request": {"translation": {"target_languages": list(languages)}}}
