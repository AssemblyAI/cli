"""Transcription execution and non-Rich output shaping for the ``transcribe`` command.

Kept out of ``commands/transcribe.py`` so the command stays a thin option surface, and
so ``run_transcription`` lives in a core module that ``onboard`` can import directly
(rather than reaching into a command module's internals).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import assemblyai as aai

from aai_cli import choices, client, stdio, youtube
from aai_cli.errors import UsageError

# The PII policy strings the SDK accepts, validated client-side so a typo'd
# --redact-pii-policy fails before any upload — mirroring how an unknown --config
# key is rejected with the valid field list.
PII_POLICY_VALUES = frozenset(policy.value for policy in aai.PIIRedactionPolicy)


def validate_pii_policies(policies: list[str] | None) -> None:
    unknown = [p for p in policies or [] if p not in PII_POLICY_VALUES]
    if unknown:
        valid = ", ".join(sorted(PII_POLICY_VALUES))
        raise UsageError(f"Unknown PII policy(s) {unknown}. Valid policies: {valid}.")


def validate_language_flags(language_code: str | None, *, language_detection: bool | None) -> None:
    if language_code and language_detection:
        raise UsageError(
            "--language-code and --language-detection can't be combined.",
            suggestion="Force a language or auto-detect it, not both.",
        )


def validate_speakers_expected(merged: dict[str, object]) -> None:
    # Checked on the merged dict so `--config speaker_labels=true` also counts.
    if merged.get("speakers_expected") and not merged.get("speaker_labels"):
        raise UsageError(
            "--speakers-expected only applies when diarization is enabled.",
            suggestion="Add --speaker-labels.",
        )


def render_transform_steps(d: dict[str, Any]) -> str:
    """Human view of chained LLM-Gateway steps: the lone output, or each step labeled."""
    steps = d["transform"]["steps"]
    if len(steps) == 1:
        return str(steps[0]["output"])
    return "\n\n".join(f"Step {i} — {s['prompt']}:\n{s['output']}" for i, s in enumerate(steps, 1))


def out_payload(
    transcript: aai.Transcript,
    output_field: choices.TranscriptOutput | None,
    *,
    json_mode: bool,
) -> str:
    """The text to write for ``--out``: the chosen ``-o`` field, the ``--json`` payload,
    or the plain transcript text — the same content stdout would get, as a file artifact."""
    if output_field is not None:
        return client.select_transcript_field(transcript, output_field)
    if json_mode:
        return json.dumps(client.transcript_json_payload(transcript), default=str)
    return client.select_transcript_field(transcript, choices.TranscriptOutput.text)


def check_source_exists(source: str | None, *, sample: bool) -> None:
    """Resolve (and existence-check) the audio reference before credential resolution.

    Stdin (``-``) is exempt: its bytes are buffered at transcription time.
    """
    if source != "-":
        client.resolve_audio_source(source, sample=sample)


def run_transcription(
    api_key: str,
    source: str | None,
    *,
    sample: bool,
    transcription_config: aai.TranscriptionConfig,
    download_sections: list[str] | None = None,
) -> aai.Transcript:
    if source == "-":
        # Audio piped on stdin (e.g. `ffmpeg -i v.mp4 -f wav - | assembly transcribe -`).
        # The SDK uploads a path, so buffer the bytes to a temp file first.
        data = stdio.read_binary_stdin()
        if not data:
            raise UsageError("No audio received on stdin.")
        with tempfile.TemporaryDirectory(prefix="aai-stdin-") as td:
            local = Path(td) / "audio"
            local.write_bytes(data)
            return client.transcribe(api_key, str(local), config=transcription_config)

    audio = client.resolve_audio_source(source, sample=sample)
    if youtube.is_downloadable_url(audio):
        # Fetch first; AssemblyAI can't read a YouTube/podcast page URL itself.
        with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
            local = youtube.download_audio(audio, Path(td), download_sections=download_sections)
            return client.transcribe(api_key, str(local), config=transcription_config)
    return client.transcribe(api_key, audio, config=transcription_config)
