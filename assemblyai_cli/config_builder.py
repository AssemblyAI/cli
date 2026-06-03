from __future__ import annotations

import json
from pathlib import Path

import assemblyai as aai
from assemblyai.streaming.v3 import StreamingParameters

from assemblyai_cli.errors import UsageError

# field name -> coercion kind for --config/--config-file string values.
# The KEYS are the authoritative set of valid config fields per command.
TRANSCRIBE_COERCE: dict[str, str] = {
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

STREAM_COERCE: dict[str, str] = {
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


def build_transcription_config(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> aai.TranscriptionConfig:
    merged = _merge(TRANSCRIBE_FIELDS, flags, overrides, config_file)
    try:
        return aai.TranscriptionConfig(**merged)
    except UsageError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface SDK validation as a usage error
        raise UsageError(f"Invalid transcription config: {exc}") from exc


def build_streaming_params(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> StreamingParameters:
    merged = _merge(STREAM_FIELDS, flags, overrides, config_file)
    try:
        return StreamingParameters(**merged)
    except UsageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise UsageError(f"Invalid streaming config: {exc}") from exc
