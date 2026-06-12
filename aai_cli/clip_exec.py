"""Run logic for `assembly clip`: cut a media file by transcript content.

The command module (aai_cli/commands/clip.py) only parses argv — it builds a
``ClipOptions`` and hands it to ``run_clip`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive transcript resolution and the
ffmpeg orchestration by constructing options directly. The pure selection logic
(range parsing, utterance filtering, LLM reply parsing, merging) lives in
``clip_select``.

Selection composes four sources: ``--speaker`` and ``--search`` filter the
diarized utterances of a transcript (made on the fly, reused via
``--transcript-id``, or piped on stdin with ``-t -``), ``--llm`` hands the
timestamped utterances to the LLM Gateway and lets the model pick the windows,
and ``--range`` adds explicit ones. The selected segments are padded, merged
where they touch, snapped into nearby silence (one ffmpeg ``silencedetect``
pass over the source, skipped with ``--no-snap``), and each surviving segment
is re-encoded into its own file with ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import assemblyai as aai
from rich.markup import escape

from aai_cli import client, clip_select, jsonshape, llm, output, stdio, youtube
from aai_cli.clip_select import Segment
from aai_cli.context import AppState
from aai_cli.errors import CLIError, UsageError


@dataclass(frozen=True)
class ClipOptions:
    """Every `assembly clip` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    # The raw source as typed: a local path, or a downloadable media-page URL
    # (a pathlib.Path would collapse the "//" in "https://").
    media: str
    transcript_id: str | None
    speakers: list[str]
    search: str | None
    llm_prompt: str | None
    model: str
    max_tokens: int
    ranges: list[str]
    padding: float
    snap: bool
    out_dir: Path | None


def _llm_segments(
    api_key: str,
    utterances: list[object],
    opts: ClipOptions,
    *,
    json_mode: bool,
    quiet: bool,
) -> list[Segment]:
    """Ask the LLM Gateway to pick the windows matching the --llm instruction."""
    selecting = f"Selecting segments with {opts.model}…"
    with output.status(selecting, json_mode=json_mode, quiet=quiet):
        reply = llm.transform_transcript(
            api_key,
            prompt=f"{clip_select.LLM_INSTRUCTIONS}{opts.llm_prompt}",
            transcript_text=clip_select.utterance_listing(utterances),
            model=opts.model,
            max_tokens=opts.max_tokens,
        )
    return clip_select.parse_llm_segments(reply)


def _needs_transcript(opts: ClipOptions) -> bool:
    """Whether any requested selector reads the transcript (vs pure --range)."""
    return bool(opts.speakers) or opts.search is not None or opts.llm_prompt is not None


@dataclass(frozen=True)
class _PipedTranscript:
    """A transcript reconstructed from ``-t -`` stdin JSON (no API round-trip)."""

    id: str
    utterances: list[object]


_PIPE_SUGGESTION = (
    "Pipe a transcript into clip: "
    "assembly transcribe <file> --speaker-labels --json | assembly clip <file> -t - …"
)


def _stdin_transcript_text() -> str:
    text = stdio.piped_stdin_text()
    if text is None:
        raise UsageError(
            "-t - expects a transcript id or transcript JSON on stdin.",
            suggestion=_PIPE_SUGGESTION,
        )
    return text.strip()


def _piped_transcript(text: str) -> _PipedTranscript:
    """The transcript object encoded in piped ``--json`` output."""
    try:
        loaded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"Couldn't parse the transcript JSON on stdin: {exc}.",
            suggestion=_PIPE_SUGGESTION,
        ) from exc
    payload = jsonshape.as_mapping(loaded) or {}
    utterances: list[object] = [
        SimpleNamespace(
            start=item.get("start"),
            end=item.get("end"),
            speaker=item.get("speaker"),
            text=item.get("text"),
        )
        for item in jsonshape.mapping_list(payload.get("utterances"))
    ]
    return _PipedTranscript(id=str(payload.get("id")), utterances=utterances)


def _resolve_transcript(
    opts: ClipOptions, media: Path, state: AppState, *, json_mode: bool
) -> object:
    """The transcript backing --speaker/--search/--llm: piped on stdin (``-t -``),
    fetched by id, or made fresh from the (already local) media file — always
    diarized, since speaker labels are what clip selects on."""
    transcript_id = opts.transcript_id
    if transcript_id == "-":
        text = _stdin_transcript_text()
        if text.startswith("{"):
            return _piped_transcript(text)
        transcript_id = text  # a bare id (e.g. from `assembly transcribe … -o id`)
    if transcript_id is not None:
        return client.get_transcript(state.resolve_api_key(), transcript_id)
    config = aai.TranscriptionConfig(speaker_labels=True)
    api_key = state.resolve_api_key()
    with output.status("Transcribing for clip selection…", json_mode=json_mode, quiet=state.quiet):
        return client.transcribe(api_key, str(media), config=config)


def _transcript_segments(
    opts: ClipOptions, media: Path, state: AppState, *, json_mode: bool
) -> tuple[list[Segment], str | None]:
    """Matched utterance segments plus the transcript id, or ``([], None)`` when
    no transcript-backed selector was requested.

    --speaker/--search narrow the utterances first; --llm then picks windows
    from whatever survived (or from the whole transcript when unfiltered).
    """
    if not _needs_transcript(opts):
        return [], None
    transcript = _resolve_transcript(opts, media, state, json_mode=json_mode)
    transcript_id = str(getattr(transcript, "id", ""))
    utterances = jsonshape.object_list(getattr(transcript, "utterances", None))
    if not utterances:
        raise CLIError(
            f"Transcript {transcript_id} has no utterances to select from.",
            error_type="no_utterances",
            exit_code=2,
            suggestion=(
                "--speaker/--search/--llm need a diarized transcript. Pass a --transcript-id "
                "created with --speaker-labels, or drop -t to let clip transcribe the file."
            ),
        )
    matched = clip_select.matching_utterances(utterances, opts.speakers, opts.search)
    if not matched:
        raise CLIError(
            "No transcript segments matched the selection.",
            error_type="no_match",
            suggestion=(
                "Inspect who said what with 'assembly transcribe <file> --speaker-labels "
                "-o utterances', then adjust --speaker/--search."
            ),
        )
    if opts.llm_prompt is not None:
        segments = _llm_segments(
            state.resolve_api_key(), matched, opts, json_mode=json_mode, quiet=state.quiet
        )
        return segments, transcript_id
    return [clip_select.segment_of(utterance) for utterance in matched], transcript_id


def _validate_media(media: Path) -> None:
    """Reject a missing local source before credential resolution, so a typo'd
    path reads as "file not found", never as a login prompt or an opaque
    ffmpeg error."""
    if not media.exists():
        raise CLIError(
            f"File not found: {media}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Check the path. assembly clip needs a local audio/video file.",
        )
    if not media.is_file():
        raise CLIError(
            f"Not a file: {media}",
            error_type="not_a_file",
            exit_code=2,
            suggestion="Pass a media file, not a directory.",
        )


def _validate_out_dir(out_dir: Path | None) -> None:
    if out_dir is not None and not out_dir.is_dir():
        raise UsageError(
            f"--out-dir doesn't exist: {out_dir}",
            suggestion="Create it first, or point --out-dir at an existing directory.",
        )


def _validate_selection(opts: ClipOptions) -> None:
    if _needs_transcript(opts):
        return
    if not opts.ranges:
        raise UsageError(
            "Nothing selects a segment to clip.",
            suggestion="Pass --speaker, --search, --llm, and/or --range.",
        )
    if opts.transcript_id is not None:
        # -t feeds the transcript-backed selectors; with only --range it would be
        # ignored, and a requested flag is never dropped silently.
        raise UsageError(
            "--transcript-id only applies with --speaker/--search/--llm.",
            suggestion="Add a --speaker/--search/--llm selector, or drop --transcript-id.",
        )


def _require_ffmpeg() -> str:
    """The ffmpeg executable; checked before any (billed) transcription work."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise CLIError(
            "ffmpeg is required to cut media, but it isn't on PATH.",
            error_type="missing_dependency",
            suggestion="Install it (brew install ffmpeg / apt install ffmpeg) and re-run.",
        )
    return path


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Boundary seam for tests: one ffmpeg invocation, output captured."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


# -30dB for at least 0.2s reads as a pause in normal speech recordings.
_SILENCE_FILTER = "silencedetect=noise=-30dB:d=0.2"


def _detect_silences(ffmpeg: str, media: Path) -> list[Segment]:
    """The silence intervals ffmpeg hears in ``media`` (one decode pass).

    Snapping is best-effort: a failed detection returns no silences (so the
    cut proceeds at the selected times) rather than failing the command.
    silencedetect logs at info level on stderr, so the usual ``-loglevel
    error`` would silence the very lines this parses.
    """
    result = _run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(media),
            "-af",
            _SILENCE_FILTER,
            "-f",
            "null",
            "-",
        ]
    )
    if result.returncode != 0:
        return []
    return clip_select.parse_silences(result.stderr)


def _cut_clip(ffmpeg: str, media: Path, segment: Segment, dest: Path) -> None:
    """Re-encode one segment of ``media`` into ``dest``.

    Re-encoding (no ``-c copy``) keeps cuts frame-accurate where stream copy
    would snap to the nearest keyframe; ``-y`` makes a re-run overwrite its own
    earlier output instead of stalling on ffmpeg's prompt.
    """
    result = _run_ffmpeg(
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
            str(dest),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"ffmpeg exited with code {result.returncode}"
        raise CLIError(
            f"Could not cut {dest.name}: {reason}",
            error_type="clip_failed",
            suggestion="Check that the input is a readable audio/video file.",
        )


def _clip_dest(media: Path, out_dir: Path | None, index: int) -> Path:
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


def run_clip(opts: ClipOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly clip` invocation from already-parsed flags."""
    _validate_out_dir(opts.out_dir)
    _validate_selection(opts)
    explicit = [clip_select.parse_range(value) for value in opts.ranges]
    ffmpeg = _require_ffmpeg()
    if youtube.is_downloadable_url(opts.media):
        # A media-page URL (YouTube, podcast page, …) is downloaded once and
        # clipped locally. The download dir is temporary, so the clips land in
        # --out-dir or the current directory — never next to the temp file.
        with tempfile.TemporaryDirectory(prefix="aai-clip-") as td:
            with output.status("Downloading audio…", json_mode=json_mode, quiet=state.quiet):
                local = youtube.download_audio(opts.media, Path(td))
            out_dir = opts.out_dir if opts.out_dir is not None else Path.cwd()
            _cut_and_emit(opts, local, out_dir, explicit, ffmpeg, state, json_mode=json_mode)
        return
    if opts.media.startswith(("http://", "https://")):
        raise UsageError(
            "assembly clip can't fetch this URL; it cuts a local file or a "
            "media-page URL yt-dlp can download (YouTube, podcasts, …).",
            suggestion="Download the media first, then clip the local copy.",
        )
    media = Path(opts.media)
    _validate_media(media)
    _cut_and_emit(opts, media, opts.out_dir, explicit, ffmpeg, state, json_mode=json_mode)


def _cut_and_emit(
    opts: ClipOptions,
    media: Path,
    out_dir: Path | None,
    explicit: list[Segment],
    ffmpeg: str,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Select, cut, and report the clips for an already-local media file."""
    matched, transcript_id = _transcript_segments(opts, media, state, json_mode=json_mode)
    segments = clip_select.merge_segments([*matched, *explicit], opts.padding)
    if opts.snap:
        with output.status("Detecting silence…", json_mode=json_mode, quiet=state.quiet):
            silences = _detect_silences(ffmpeg, media)
        segments = clip_select.snap_to_silences(segments, silences)
    written: list[WrittenClip] = []
    cutting = f"Cutting {len(segments)} clip(s)…"
    with output.status(cutting, json_mode=json_mode, quiet=state.quiet):
        for index, segment in enumerate(segments, 1):
            dest = _clip_dest(media, out_dir, index)
            _cut_clip(ffmpeg, media, segment, dest)
            written.append(WrittenClip(path=dest, segment=segment))
    payload: dict[str, object] = {
        "source": opts.media,
        "transcript_id": transcript_id,
        "clips": [clip.payload() for clip in written],
    }
    output.emit(
        payload,
        lambda _: "\n".join(clip.human_line() for clip in written),
        json_mode=json_mode,
    )
