"""Batch transcription: directories, globs, and stdin lists with sidecar resume.

``assembly transcribe`` switches to batch mode when the source is a directory or a
glob pattern — local, or on fsspec-addressable remote storage (an ``s3://…/*.mp3``
glob, or a trailing-slash folder like ``s3://bucket/calls/``) — or when
``--from-stdin`` supplies one path/URL per line (the source-list expansion itself
lives in ``transcribe_sources``). Sources run concurrently behind a live progress
table; each finished source gets a ``<source>.aai.json`` sidecar holding the full
transcript. The sidecar doubles as the resume marker — a re-run skips any source
whose sidecar records a completed transcription of the same bytes — so retrying a
partly-failed batch only pays for what's missing (``--force`` re-transcribes
everything).

``--llm`` prompts run per source once its transcription is recorded, landing under
the sidecar's ``transform`` key. The chain is resumable on its own: a re-run with
missing or changed prompts replays just the LLM step against the recorded
transcript id, never a second transcription.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from rich.live import Live
from rich.markup import escape

from aai_cli.app.transcribe import run as transcribe_exec
from aai_cli.app.transcribe.sources import SIDECAR_SUFFIX, URL_PREFIXES
from aai_cli.core import client, jsonshape, llm, remotefs
from aai_cli.core.errors import CLIError, NotAuthenticated
from aai_cli.ui import output, theme

if TYPE_CHECKING:
    import assemblyai as aai
    from rich.table import Table


def sidecar_path(source: str) -> Path:
    """Where ``source``'s sidecar lives: ``<file>.aai.json`` next to a local file, or
    a slug + URL-hash name in the working directory for a URL (web or bucket)."""
    if source.startswith(URL_PREFIXES) or remotefs.is_remote_url(source):
        digest = hashlib.sha256(source.encode()).hexdigest()[:8]
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source.partition("://")[2]).strip("-.")[:64]
        return Path(f"{slug}-{digest}{SIDECAR_SUFFIX}")
    return Path(source + SIDECAR_SUFFIX)


def _source_digest(source: str) -> str | None:
    """SHA-256 of a local file's bytes; ``None`` for URLs (and paths that aren't files)."""
    if source.startswith(URL_PREFIXES) or not Path(source).is_file():
        return None
    with Path(source).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resumable_record(sidecar: Path, *, digest: str | None) -> dict[str, object] | None:
    """The sidecar's record when it marks a completed transcription of the same bytes.

    ``None`` (transcribe again) when the sidecar is missing or corrupt, the run
    didn't complete, or a local file's hash no longer matches the recorded one.
    """
    try:
        record = jsonshape.as_mapping(json.loads(sidecar.read_text()))
    except (OSError, ValueError):
        return None
    if record is None or record.get("status") != "completed":
        return None
    if digest is not None and record.get("source_sha256") != digest:
        return None
    return record


def _dump_sidecar(sidecar: Path, record: dict[str, object]) -> None:
    sidecar.write_text(json.dumps(record, indent=2, default=str) + "\n")


def _write_sidecar(
    sidecar: Path, *, source: str, transcript: aai.Transcript, digest: str | None
) -> dict[str, object]:
    record: dict[str, object] = {
        "source": source,
        "id": transcript.id,
        "status": client.status_str(transcript),
        "transcript": client.transcript_json_payload(transcript),
    }
    if digest is not None:
        record["source_sha256"] = digest
    _dump_sidecar(sidecar, record)
    return record


def _transform_record(
    api_key: str, transform: transcribe_exec.TransformOptions, *, transcript_id: str
) -> dict[str, object]:
    """Run the ``--llm`` chain server-side over the transcript; the sidecar entry."""
    steps = llm.run_chain_steps(
        api_key,
        transform.prompts,
        transcript_id=transcript_id,
        model=transform.model,
        max_tokens=transform.max_tokens,
    )
    return {"model": transform.model, "prompts": transform.prompts, "steps": steps}


def _transform_satisfied(
    record: dict[str, object], transform: transcribe_exec.TransformOptions
) -> bool:
    """True when no ``--llm`` chain was requested, or the sidecar already records this
    exact chain (same prompts against the same gateway model)."""
    if not transform.prompts:
        return True
    existing = jsonshape.as_mapping(record.get("transform"))
    if existing is None:
        return False
    return existing.get("prompts") == transform.prompts and existing.get("model") == transform.model


@dataclasses.dataclass
class _Item:
    source: str
    status: str = "queued"  # queued → processing → completed | skipped | failed
    transcript_id: str = ""
    detail: str = ""  # sidecar path when completed/skipped; the error message when failed

    def record(self) -> dict[str, str]:
        """The NDJSON record emitted for this source under ``--json``."""
        # "type" discriminates NDJSON lines CLI-wide (see docs/cli-reference.md).
        rec = {"type": "result", "source": self.source, "status": self.status}
        if self.transcript_id:
            rec["id"] = self.transcript_id
        if self.status == "failed":
            rec["error"] = self.detail
        elif self.detail:
            rec["sidecar"] = self.detail
        return rec


def _resume_one(
    api_key: str,
    item: _Item,
    record: dict[str, object],
    sidecar: Path,
    *,
    transform: transcribe_exec.TransformOptions,
) -> bool:
    """Finish a source whose completed transcription the sidecar already holds.

    Skips outright when the recorded ``transform`` satisfies the requested chain;
    otherwise replays just the chain against the recorded transcript id. Returns
    False (transcribe again) when the record has no id to anchor the chain on.
    """
    item.transcript_id = str(record.get("id") or "")
    if _transform_satisfied(record, transform):
        item.status, item.detail = "skipped", str(sidecar)
        return True
    if not item.transcript_id:
        return False
    item.status = "processing"
    transformed = _transform_record(api_key, transform, transcript_id=item.transcript_id)
    _dump_sidecar(sidecar, dict(record) | {"transform": transformed})
    item.status, item.detail = "completed", str(sidecar)
    return True


def _transcribe_one(
    api_key: str,
    item: _Item,
    *,
    transcription_config: aai.TranscriptionConfig,
    force: bool,
    transform: transcribe_exec.TransformOptions,
) -> None:
    """Worker body: resume from the sidecar, or transcribe and write one.

    The ``--llm`` chain runs only after the sidecar records the completed
    transcription, so a failed chain leaves a resumable transcription and the
    retry pays only for the LLM step.

    A per-source failure is recorded on the item and the batch carries on — except
    NotAuthenticated, which re-raises so ``_drain`` aborts the batch (one rejected
    key fails every source identically, and auto-login should trigger once).
    """
    try:
        sidecar = sidecar_path(item.source)
        digest = _source_digest(item.source)
        record = None if force else resumable_record(sidecar, digest=digest)
        if record is not None and _resume_one(api_key, item, record, sidecar, transform=transform):
            return
        item.status = "processing"
        transcript = transcribe_exec.run_transcription(
            api_key, item.source, sample=False, transcription_config=transcription_config
        )
        fresh = _write_sidecar(sidecar, source=item.source, transcript=transcript, digest=digest)
        item.transcript_id = transcript.id or ""
        if transform.prompts:
            transformed = _transform_record(api_key, transform, transcript_id=item.transcript_id)
            _dump_sidecar(sidecar, fresh | {"transform": transformed})
        item.status, item.detail = "completed", str(sidecar)
    except CLIError as err:
        item.status, item.detail = "failed", err.message
        if isinstance(err, NotAuthenticated):
            raise


def _render_table(items: list[_Item]) -> Table:
    table = output.data_table("Source", "Status", "Transcript", "Result")
    for item in items:
        table.add_row(
            escape(item.source),
            theme.status_text(item.status),
            item.transcript_id,
            escape(item.detail),
        )
    return table


@contextmanager
def _progress_table(items: list[_Item], *, json_mode: bool) -> Generator[None]:
    """Render the batch as a live-updating table (human mode).

    Rich renders nothing while running on a non-interactive console and prints the
    final frame once on stop, so piped/agent runs still get the result table. JSON
    mode skips Rich entirely — NDJSON per source is the output.
    """
    if json_mode:
        yield
        return
    with Live(
        get_renderable=lambda: _render_table(items),
        console=output.console,
        refresh_per_second=4,  # pragma: no mutate (cosmetic refresh cadence)
    ):
        yield


def _drain(
    api_key: str,
    items: list[_Item],
    *,
    transcription_config: aai.TranscriptionConfig,
    concurrency: int,
    force: bool,
    transform: transcribe_exec.TransformOptions,
    json_mode: bool,
) -> None:
    """Run the workers, emitting one NDJSON record per finished source under ``--json``.

    The first exception that escapes a worker (NotAuthenticated, or a bug) drops the
    not-yet-started sources and re-raises once the in-flight ones drain.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _transcribe_one,
                api_key,
                item,
                transcription_config=transcription_config,
                force=force,
                transform=transform,
            ): item
            for item in items
        }
        for future in as_completed(futures):
            if (exc := future.exception()) is not None:
                pool.shutdown(cancel_futures=True)
                raise exc
            if json_mode:
                output.emit_ndjson(futures[future].record())


def _summarize(items: list[_Item], *, json_mode: bool, quiet: bool) -> None:
    failed = sum(1 for item in items if item.status == "failed")
    if failed:
        raise CLIError(
            f"{failed} of {len(items)} sources failed.",
            error_type="batch_failed",
            suggestion="Re-run the same command to retry only the failures — "
            "completed sources are skipped via their sidecars.",
        )
    completed = sum(1 for item in items if item.status == "completed")
    skipped = len(items) - completed
    if not json_mode and not quiet:
        output.error_console.print(output.success(f"Transcribed {completed}, skipped {skipped}."))


def run_batch(
    api_key: str,
    sources: list[str],
    *,
    transcription_config: aai.TranscriptionConfig,
    concurrency: int,
    force: bool,
    transform: transcribe_exec.TransformOptions,
    json_mode: bool,
    quiet: bool,
) -> None:
    """Transcribe ``sources`` concurrently, writing one sidecar per source.

    Raises CLIError (exit 1) when any source failed so scripts can trust the exit
    code; a re-run resumes from the sidecars and retries only the failures.
    """
    items = [_Item(source) for source in sources]
    with _progress_table(items, json_mode=json_mode):
        _drain(
            api_key,
            items,
            transcription_config=transcription_config,
            concurrency=concurrency,
            force=force,
            transform=transform,
            json_mode=json_mode,
        )
    _summarize(items, json_mode=json_mode, quiet=quiet)
