"""ffmpeg cutting for `assembly clip`: silence detection + per-segment re-encode.

The pure selection logic (range parsing, utterance filtering, merging) lives in
``_select``; this module is the final stage — it turns the merged ``Segment``
windows into output files with ffmpeg: one ``silencedetect`` pass to snap cuts
into nearby pauses, then a frame-accurate re-encode per segment. The
orchestration that ties selection and cutting together stays in ``_exec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from aai_cli.app import mediafile
from aai_cli.commands.clip import _select as clip_select
from aai_cli.commands.clip._select import Segment
from aai_cli.ui import output

# -30dB for at least 0.2s reads as a pause in normal speech recordings.
SILENCE_FILTER = "silencedetect=noise=-30dB:d=0.2"


def detect_silences(ffmpeg: str, media: Path) -> list[Segment]:
    """The silence intervals ffmpeg hears in ``media`` (one decode pass).

    Snapping is best-effort: a failed detection returns no silences (so the
    cut proceeds at the selected times) rather than failing the command.
    silencedetect logs at info level on stderr, so the usual ``-loglevel
    error`` would silence the very lines this parses.
    """
    result = mediafile.run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(media),
            "-af",
            SILENCE_FILTER,
            "-f",
            "null",
            "-",
        ]
    )
    if result.returncode != 0:
        return []
    return clip_select.parse_silences(result.stderr)


def cut_clip(ffmpeg: str, media: Path, segment: Segment, dest: Path) -> None:
    """Re-encode one segment of ``media`` into ``dest``.

    Re-encoding (no ``-c copy``) keeps cuts frame-accurate where stream copy
    would snap to the nearest keyframe; ``-y`` makes a re-run overwrite its own
    earlier output instead of stalling on ffmpeg's prompt.
    """
    result = mediafile.run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media),
            "-ss",
            f"{segment.start:.3f}",
            "-to",
            f"{segment.end:.3f}",
            mediafile.path_arg(dest),
        ]
    )
    if result.returncode != 0:
        raise mediafile.ffmpeg_failure(result, "cut", dest, error_type="clip_failed")


def clip_dest(media: Path, out_dir: Path | None, index: int) -> Path:
    directory = out_dir if out_dir is not None else media.parent
    return directory / f"{media.stem}.clip{index:02d}{media.suffix}"


@dataclass(frozen=True)
class WrittenClip:
    """One output file and the source window it was cut from."""

    path: Path
    segment: Segment

    def payload(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "start": round(self.segment.start, 3),
            "end": round(self.segment.end, 3),
            "duration": round(self.segment.end - self.segment.start, 3),
        }

    def human_line(self) -> str:
        start = clip_select.format_clock(self.segment.start)
        end = clip_select.format_clock(self.segment.end)
        duration = round(self.segment.end - self.segment.start, 3)
        return output.success(f"{escape(str(self.path))}  {start} - {end}  ({duration}s)")
