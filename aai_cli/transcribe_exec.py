"""Transcription execution and result delivery for the ``transcribe`` command.

Kept out of ``commands/transcribe.py`` so the command stays a thin option surface, and
so ``run_transcription`` lives in a core module that ``onboard`` can import directly
(rather than reaching into a command module's internals).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import assemblyai as aai
from rich.markup import escape

from aai_cli import choices, client, llm, output, stdio, transcribe_render, youtube
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


def validate_out_with_llm(out: Path | None, llm_prompts: list[str] | None) -> None:
    if out is not None and llm_prompts:
        # --out captures the transcript itself; an LLM transform is a separate step.
        raise UsageError(
            "--out can't be combined with --llm.",
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
    if json_mode and output_field is not None and output_field is not choices.TranscriptOutput.json:
        raise UsageError(
            f"--json conflicts with -o {output_field.value}.",
            suggestion="Drop --json, or use -o json for the full JSON payload.",
        )


def warn_unrecognized_extension(source: str | None, *, json_mode: bool, quiet: bool) -> None:
    """Warn when a single local source doesn't carry a known audio extension.

    Directory batch mode filters by ``AUDIO_EXTENSIONS``; single-file mode uploads
    anything, so a likely-non-audio file (e.g. ``.txt``) gets a stderr heads-up —
    never an error, since the server is the truth about what it can transcribe.
    """
    from aai_cli.transcribe_batch import AUDIO_EXTENSIONS  # avoid a module-load cycle

    if quiet or not source or source.startswith(("http://", "https://")):
        return
    suffix = Path(source).suffix.lower()
    if not suffix or suffix in AUDIO_EXTENSIONS:
        return
    output.emit_warning(
        f"'{source}' has extension '{suffix}', which doesn't look like audio; "
        "the API decides what it can transcribe.",
        json_mode=json_mode,
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


class TransformOptions(NamedTuple):
    """The ``--llm`` chain options: the prompts plus the gateway model settings."""

    prompts: list[str]
    model: str
    max_tokens: int


def deliver_result(
    transcript: aai.Transcript,
    *,
    api_key: str,
    out: Path | None,
    output_field: choices.TranscriptOutput | None,
    transform: TransformOptions,
    json_mode: bool,
    quiet: bool,
) -> None:
    """Route the finished transcript: ``--out`` file, single ``-o`` field, ``--llm``
    transform chain, or the default JSON/human render — first match wins."""
    if out is not None:
        # Write a clean file artifact and confirm on stderr; stdout stays empty.
        # The path itself was validated up front by validate_out_path.
        out.write_text(out_payload(transcript, output_field, json_mode=json_mode) + "\n")
        if not quiet:
            output.error_console.print(output.success(f"Saved to {escape(str(out))}"))
        return

    if output_field is not None:
        # Raw single-field output for pipelines (overrides --json and analysis render).
        output.emit_text(client.select_transcript_field(transcript, output_field))
        return

    if transform.prompts:
        # Chain the prompts: the first runs over the transcript (injected server-side
        # via transcript_id); each subsequent prompt runs over the prior response.
        steps = llm.run_chain_steps(
            api_key,
            transform.prompts,
            transcript_id=transcript.id,
            model=transform.model,
            max_tokens=transform.max_tokens,
        )
        output.emit(
            client.transcript_summary(transcript)
            | {"transform": {"model": transform.model, "steps": steps}},
            render_transform_steps,
            json_mode=json_mode,
        )
        return

    if json_mode:
        output.emit(client.transcript_json_payload(transcript), lambda d: d, json_mode=True)
    else:
        transcribe_render.render_transcript_result(transcript, output.console)
