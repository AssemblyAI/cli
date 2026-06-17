"""The `assembly stream --save-dir` capture lifecycle: auto-name, note, and sidecar.

``--save-dir`` already auto-names a transcript + WAV under ``DIR/YYYY-MM-DD/``
(see ``naming``). This module folds the steps a wrapper script used to bolt on
afterwards into capture time, so nothing downstream needs an index pass:

- ``--auto-name`` derives the filename slug from the transcript itself (via the
  LLM), so a recording is meaningfully named with no calendar or manual title.
- ``--llm`` alongside ``--save-dir`` writes its final answer as a ``.md`` note
  next to the transcript — a summary produced as the audio is captured.
- a ``.aai.json`` sidecar (title, date, duration, speakers, turns) lands beside
  every recording so a list/browse UI shows rich info without parsing transcripts.

``write_outputs`` is pure file I/O (no network), so the rename/note/sidecar
behavior is unit-tested without a gateway; the LLM title call lives in
``derive_title`` and is injected past it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aai_cli.core import jsonshape, llm
from aai_cli.core.errors import CLIError
from aai_cli.streaming import naming

# Asks for a short headline only — kept terse so a small/cheap model returns a clean
# line we can slug, not a paragraph. The transcript is appended by build_messages.
TITLE_PROMPT = (
    "Write a short, descriptive title (3 to 7 words) for this transcript. "
    "Reply with only the title — no quotes, no surrounding punctuation."
)


@dataclass(frozen=True)
class SaveDirPlan:
    """The resolved ``--save-dir`` intent, the data ``write_outputs`` finalizes from.

    ``now`` is captured once up front so the live transcript file, the post-stream
    rename, and the sidecar's ``date`` all agree on a single timestamp.
    """

    save_dir: Path
    now: datetime
    name: str | None
    auto_name: bool
    write_note: bool

    @property
    def paths(self) -> naming.SavePaths:
        """The provisional paths the live run writes to (before any ``--auto-name`` rename)."""
        return naming.resolve(self.save_dir, self.name, now=self.now)


def derive_title(api_key: str, transcript_text: str, *, model: str, max_tokens: int) -> str:
    """Ask the LLM for a short headline for ``transcript_text`` (the ``--auto-name`` title).

    Returns the raw title text; ``naming.resolve`` slugs it into the filename, so an
    unusable answer (all punctuation) simply collapses to the bare timestamp stem.
    """
    return llm.run_chain(
        api_key,
        [TITLE_PROMPT],
        transcript_text=transcript_text,
        model=model,
        max_tokens=max_tokens,
    ).strip()


def _rename(src: Path, dst: Path) -> None:
    """Move a provisional capture file to its final auto-named path, as a clean CLIError."""
    try:
        src.rename(dst)
    except OSError as exc:
        raise CLIError(
            f"Cannot rename {src} to {dst}: {exc}",
            error_type="save_dir_path",
            exit_code=2,
        ) from exc


def _restem(path: Path, old_stem: str, new_stem: str) -> Path:
    """Swap a capture file's stem prefix for the auto-name rename.

    Every capture file is named ``<stem><rest>`` (``.txt``, ``.wav``, ``-you.wav``, …),
    so re-stemming preserves the per-channel/extension suffix while adopting the new
    auto-named stem.
    """
    return path.with_name(new_stem + path.name[len(old_stem) :])


def _move_restem(src: Path, old_stem: str, new_stem: str) -> Path:
    """Re-stem ``src`` to the auto-named stem and move it there, returning the new path.

    A stream interrupted before any audio leaves no WAV to move, so a missing source is
    skipped (its intended new name is still reported in the sidecar).
    """
    dst = _restem(src, old_stem, new_stem)
    if src.exists():
        _rename(src, dst)
    return dst


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` (the note or sidecar), as a clean CLIError on failure."""
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CLIError(
            f"Cannot write {path}: {exc}",
            error_type="save_dir_path",
            exit_code=2,
        ) from exc


def _sidecar_record(
    paths: naming.SavePaths,
    *,
    plan: SaveDirPlan,
    title: str | None,
    note_written: bool,
    audio_names: list[str],
    speakers: list[str],
    duration_seconds: int,
    turns: int,
) -> dict[str, object]:
    """The ``.aai.json`` metadata: enough for a browse UI without parsing the transcript.

    ``audio`` is a list so the ``--system-audio`` two-WAV case (one per channel) reads the
    same shape as the single-WAV case; it is empty under ``--no-save-audio``.
    """
    return {
        "title": title,
        "date": plan.now.isoformat(),
        "duration_seconds": duration_seconds,
        "speakers": speakers,
        "turns": turns,
        "transcript": paths.transcript.name,
        "audio": audio_names,
        "note": paths.note.name if note_written else None,
    }


def write_outputs(
    plan: SaveDirPlan,
    *,
    title: str | None,
    note: str | None,
    audio: list[Path],
    speakers: list[str],
    duration_seconds: int,
    turns: int,
) -> naming.SavePaths:
    """Finalize a ``--save-dir`` capture: rename for ``--auto-name``, write note + sidecar.

    ``title`` is the ``--auto-name`` headline (None when not requested); when it slugs to a
    non-empty stem the provisional transcript and every WAV in ``audio`` are re-stemmed to
    carry it. ``note`` is the final ``--llm`` answer, written as ``<stem>.md`` when present.
    The sidecar is always written. Returns the final paths.
    """
    provisional = plan.paths
    final_name = title if (plan.auto_name and title) else plan.name
    final = naming.resolve(plan.save_dir, final_name, now=plan.now)
    final_audio = list(audio)
    if final.stem != provisional.stem:
        _rename(provisional.transcript, final.transcript)
        final_audio = [_move_restem(src, provisional.stem, final.stem) for src in audio]
    if note is not None:
        _write(final.note, note + "\n")
    record = _sidecar_record(
        final,
        plan=plan,
        title=final_name,
        note_written=note is not None,
        audio_names=[p.name for p in final_audio],
        speakers=speakers,
        duration_seconds=duration_seconds,
        turns=turns,
    )
    _write(final.sidecar, jsonshape.dumps_sidecar(record))
    return final
