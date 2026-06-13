"""Argument validation and warnings for ``assembly transcribe``.

These checks run before any billed work — a typo'd ``--redact-pii-policy``, a
contradictory flag pair, or an unwritable ``--out`` directory should fail (or
warn) up front rather than after a long upload. They're pure functions of the
parsed flags with no transcription state, so they live apart from the execution
body in ``transcribe_exec`` and the onboarding wizard can reuse the same surface.
"""

from __future__ import annotations

import os
from pathlib import Path

import assemblyai as aai

from aai_cli.app.transcribe import sources as transcribe_sources
from aai_cli.core import choices
from aai_cli.core.errors import UsageError, mutually_exclusive
from aai_cli.ui import output

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
    mutually_exclusive(
        ("--language-code", language_code),
        ("--language-detection", language_detection),
        suggestion="Force a language or auto-detect it, not both.",
    )


def validate_speakers_expected(merged: dict[str, object]) -> None:
    # Checked on the merged dict so `--config speaker_labels=true` also counts.
    if merged.get("speakers_expected") and not merged.get("speaker_labels"):
        raise UsageError(
            "--speakers-expected only applies when diarization is enabled.",
            suggestion="Add --speaker-labels.",
        )


def validate_out_with_llm(out: Path | None, llm_prompts: list[str] | None) -> None:
    # --out captures the transcript itself; an LLM transform is a separate step.
    mutually_exclusive(
        ("--out", out),
        ("--llm", llm_prompts),
        suggestion='Pipe the transform instead, e.g. -o text | assembly llm -f "…".',
    )


def validate_out_path(out: Path | None) -> None:
    """Reject an unusable ``--out`` up front, before the (billed, possibly long)
    transcription runs — not after it finishes."""
    if out is None:
        return
    if ".." in out.parts:  # reject path-traversal segments in --out
        raise UsageError(f"--out path can't contain '..': {out}")
    parent = out.parent
    if not parent.is_dir():
        raise UsageError(
            f"--out directory doesn't exist: {parent}",
            suggestion="Create it first, or point --out at an existing directory.",
        )
    if not os.access(parent, os.W_OK):
        raise UsageError(f"--out directory isn't writable: {parent}")


def validate_json_with_output(
    output_field: choices.TranscriptOutput | None, *, json_mode: bool
) -> None:
    """``--json`` promises the full JSON payload (same as ``-o json``); any other
    ``-o`` field contradicts it rather than silently winning."""
    if output_field is None or output_field is choices.TranscriptOutput.json:
        return
    mutually_exclusive(
        ("--json", json_mode),
        (f"-o {output_field.value}", output_field),
        suggestion="Drop --json, or use -o json for the full JSON payload.",
    )


def warn_unrecognized_extension(source: str | None, *, json_mode: bool, quiet: bool) -> None:
    """Warn when a single local source doesn't carry a known audio extension.

    Directory batch mode filters by ``AUDIO_EXTENSIONS``; single-file mode uploads
    anything, so a likely-non-audio file (e.g. ``.txt``) gets a stderr heads-up —
    never an error, since the server is the truth about what it can transcribe.
    """
    if quiet or not source or source.startswith(("http://", "https://")):
        return
    suffix = Path(source).suffix.lower()
    if not suffix or suffix in transcribe_sources.AUDIO_EXTENSIONS:
        return
    output.emit_warning(
        f"'{source}' has extension '{suffix}', which doesn't look like audio; "
        "the API decides what it can transcribe.",
        json_mode=json_mode,
    )
