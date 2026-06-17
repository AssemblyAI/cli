"""Save-destination resolution for `assembly stream`.

Split out of ``_exec`` so the save-flag matrix (`--save-dir` auto-naming vs the
explicit `--save-audio`/`--save-transcript` paths, the `--system-audio` per-channel
WAV split, and the flags that apply only under `--save-dir`) lives in one file. The
run path calls :func:`resolve_save_targets` and writes to the returned
``SaveTargets``; everything here is a pure function of the parsed options.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.core.errors import UsageError, mutually_exclusive
from aai_cli.streaming import naming, record, savedir, transcript
from aai_cli.streaming.validate import SourceOptions

if TYPE_CHECKING:
    from aai_cli.commands.stream._exec import StreamOptions


@dataclass(frozen=True)
class SaveTargets:
    """Resolved save destinations for one streaming run.

    ``audio`` tees a single source to one WAV; ``audio_by_label`` instead maps each
    parallel ``--system-audio`` channel ("you", "system") to its own WAV when the two
    streams can't share a file. At most one of the two is set; ``transcript`` is the
    single shared transcript either way. ``plan`` is set only under ``--save-dir`` and
    carries the post-stream finalization (auto-name rename, ``--llm`` note, sidecar).
    """

    transcript: Path | None = None
    audio: Path | None = None
    audio_by_label: dict[str, Path] | None = None
    plan: savedir.SaveDirPlan | None = None


def _save_dir_targets(opts: StreamOptions, sources: SourceOptions, save_dir: Path) -> SaveTargets:
    """Resolve ``--save-dir`` into auto-named targets plus the finalization plan.

    ``--save-dir`` owns filename assembly, so it rejects the explicit
    ``--save-audio``/``--save-transcript`` paths and the conflicting ``--name``/
    ``--auto-name`` title pair. Two parallel ``--system-audio`` streams can't tee to one
    WAV, so each channel gets its own ``<stem>-{you,system}.wav`` (one shared transcript);
    ``--no-save-audio`` drops the WAV(s) entirely.
    """
    mutually_exclusive(
        ("--save-dir", True),
        ("--save-audio", opts.save_audio is not None),
        ("--save-transcript", opts.save_transcript is not None),
        suggestion="--save-dir names the files for you; drop the explicit path.",
    )
    mutually_exclusive(
        ("--name", opts.name is not None),
        ("--auto-name", opts.auto_name),
        suggestion="Both set the title — pass --name for an explicit one or "
        "--auto-name to derive it from the transcript.",
    )
    # Local wall-clock time (what a meeting filename wants); the explicit utc-then-
    # astimezone keeps the now() call timezone-aware for the linter.
    now = datetime.now(UTC).astimezone()
    plan = savedir.SaveDirPlan(
        save_dir=save_dir,
        now=now,
        name=opts.name,
        auto_name=opts.auto_name,
        write_note=bool(opts.llm_prompt),
    )
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    if opts.no_save_audio:
        # Transcript + sidecar (+ note) only; no WAV teed for any source.
        return SaveTargets(transcript=paths.transcript, plan=plan)
    if sources.system_audio:
        # Parallel mic + system: one WAV per channel beside the shared transcript.
        return SaveTargets(
            transcript=paths.transcript,
            audio_by_label={
                "you": naming.channel_audio(paths.audio, "you"),
                "system": naming.channel_audio(paths.audio, "system"),
            },
            plan=plan,
        )
    if sources.system_audio_only:
        # A lone system-audio stream; label its single WAV so it reads like the pair.
        return SaveTargets(
            transcript=paths.transcript,
            audio=naming.channel_audio(paths.audio, "system"),
            plan=plan,
        )
    return SaveTargets(transcript=paths.transcript, audio=paths.audio, plan=plan)


def resolve_save_targets(opts: StreamOptions, sources: SourceOptions) -> SaveTargets:
    """Resolve the save flags into the destinations the session writes.

    ``--save-dir`` owns filename assembly (see ``_save_dir_targets``); the explicit
    ``--save-audio``/``--save-transcript`` paths are the fallback, with the save-dir-only
    ``--name``/``--auto-name``/``--no-save-audio`` flags rejected outside it.
    """
    if opts.save_dir is not None:
        return _save_dir_targets(opts, sources, opts.save_dir)
    if opts.name is not None:
        raise UsageError(
            "--name applies only with --save-dir.",
            suggestion="Pass --save-dir DIR to auto-name the files, "
            "or --save-transcript PATH for an explicit path.",
        )
    if opts.auto_name:
        raise UsageError(
            "--auto-name applies only with --save-dir.",
            suggestion="Pass --save-dir DIR so there's an auto-named file to title.",
        )
    if opts.no_save_audio:
        raise UsageError(
            "--no-save-audio applies only with --save-dir.",
            suggestion="Omit --save-audio to skip the WAV, or pass --save-dir DIR.",
        )
    if opts.save_audio is not None:
        if sources.system_audio:
            raise UsageError(
                "--save-audio cannot be combined with --system-audio; the mic and system "
                "streams can't share one file.",
                suggestion="Pass --save-dir DIR to save one WAV per channel, "
                "or record a single source.",
            )
        record.validate_target(opts.save_audio)
    if opts.save_transcript is not None:
        transcript.validate_target(opts.save_transcript)
    return SaveTargets(transcript=opts.save_transcript, audio=opts.save_audio)
