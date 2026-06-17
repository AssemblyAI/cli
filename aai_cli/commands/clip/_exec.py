"""Run logic for `assembly clip`: cut a media file by transcript content.

The command module (aai_cli/commands/clip/__init__.py) only parses argv — it builds a
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

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rich.markup import escape

from aai_cli.app import batch, mediafile
from aai_cli.app.context import AppState
from aai_cli.commands.clip import _select as clip_select
from aai_cli.commands.clip._select import Segment
from aai_cli.core import jsonshape, llm, stdio, youtube
from aai_cli.core.errors import CLIError, UsageError, mutually_exclusive
from aai_cli.ui import output


@dataclass(frozen=True)
class ClipOptions:
    """Every `assembly clip` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    # The raw source as typed: a local path, or a downloadable media-page URL
    # (a pathlib.Path would collapse the "//" in "https://"). Empty in batch mode,
    # where the sources arrive on stdin (--from-stdin) instead of as the argument.
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
    video: bool
    from_stdin: bool
    concurrency: int
    force: bool


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
    return mediafile.resolve_diarized_transcript(
        state.resolve_api_key(),
        transcript_id,
        media,
        status_message="Transcribing for clip selection…",
        json_mode=json_mode,
        quiet=state.quiet,
    )


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


def _validate_out_dir(out_dir: Path | None) -> None:
    if out_dir is None or out_dir.is_dir():
        return
    # Distinguish a missing path from one that exists but isn't a directory: the old
    # "doesn't exist" wording was misleading when --out-dir pointed at a regular file.
    if out_dir.exists():
        raise UsageError(
            f"--out-dir is not a directory: {out_dir}",
            suggestion="Point --out-dir at a directory, not a file.",
        )
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


# -30dB for at least 0.2s reads as a pause in normal speech recordings.
_SILENCE_FILTER = "silencedetect=noise=-30dB:d=0.2"


def _detect_silences(ffmpeg: str, media: Path) -> list[Segment]:
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
    """Execute `assembly clip`: one source, or a stdin batch (`--from-stdin`)."""
    _validate_out_dir(opts.out_dir)
    _validate_selection(opts)
    explicit = [clip_select.parse_range(value) for value in opts.ranges]
    ffmpeg = mediafile.require_ffmpeg("cut media")
    sources = batch.stdin_sources(opts.media, from_stdin=opts.from_stdin)
    if sources is not None:
        _reject_batch_conflicts(opts)
        batch.run_batch(
            sources,
            worker=_clip_worker(
                opts, explicit, ffmpeg, state, force=opts.force, json_mode=json_mode
            ),
            concurrency=opts.concurrency,
            summary_verb="Clipped",
            json_mode=json_mode,
            quiet=state.quiet,
        )
        return
    if not opts.media:
        raise UsageError(
            "Pass a media file or URL to clip, or --from-stdin to read a list from stdin.",
            suggestion="e.g. assembly clip meeting.mp4 --speaker A",
        )
    payload, written = _clip_one(opts, explicit, ffmpeg, state, json_mode=json_mode)
    output.emit(
        payload,
        lambda _: "\n".join(clip.human_line() for clip in written),
        json_mode=json_mode,
    )


def _reject_batch_conflicts(opts: ClipOptions) -> None:
    """Flags that can't span a many-source batch (a single transcript id can't)."""
    mutually_exclusive(
        ("--transcript-id/-t", opts.transcript_id),
        ("--from-stdin", True),
        suggestion="A transcript id can't apply to many sources; drop -t in batch mode.",
    )


def _clip_worker(
    opts: ClipOptions,
    explicit: list[Segment],
    ffmpeg: str,
    state: AppState,
    *,
    force: bool,
    json_mode: bool,
) -> batch.Worker:
    """A per-source worker for the batch runner: skip a source whose clips already
    exist (unless ``--force``), else cut it with spinners silenced."""
    quiet_state = dataclasses.replace(state, quiet=True)

    def worker(source: str) -> batch.SourceResult:
        if not force and _clips_exist(source, opts.out_dir):
            return batch.SourceResult(
                payload={"source": source}, summary="clips exist", status="skipped"
            )
        payload, written = _clip_one(
            dataclasses.replace(opts, media=source),
            explicit,
            ffmpeg,
            quiet_state,
            json_mode=json_mode,
        )
        return batch.SourceResult(payload=payload, summary=f"{len(written)} clip(s)")

    return worker


def _clips_exist(source: str, out_dir: Path | None) -> bool:
    """Whether a local ``source`` already has clip files (so batch mode skips it).

    URLs never match (the clipped stem isn't known until download), so they're
    always processed.
    """
    if "://" in source:
        return False
    src = Path(source)
    directory = out_dir if out_dir is not None else src.parent
    return any(directory.glob(f"{src.stem}.clip*{src.suffix}"))


def _clip_one(
    opts: ClipOptions,
    explicit: list[Segment],
    ffmpeg: str,
    state: AppState,
    *,
    json_mode: bool,
) -> tuple[dict[str, object], list[WrittenClip]]:
    """Resolve ``opts.media`` to a local file and cut its clips; the payload + clips.

    A media-page URL is downloaded once — the audio track by default, the full
    video with --video so the clips carry video too — and clipped locally.
    """
    youtube.validate_video_flag(opts.media, video=opts.video)
    with mediafile.resolve_media_source(
        opts.media,
        "clip",
        fetch_clause="cuts a local file or a media-page URL yt-dlp can download (YouTube, podcasts, …)",
        download_suggestion="Download the media first, then clip the local copy.",
        video=opts.video,
        download_sections=None,
        json_mode=json_mode,
        quiet=state.quiet,
    ) as (media, downloaded):
        if not downloaded:
            mediafile.validate_local_media(media, "clip")
        # A downloaded source lives in a temp dir, so its clips land in --out-dir
        # or the current directory — never next to the vanishing temp file.
        out_dir: Path | None = opts.out_dir
        if downloaded and out_dir is None:
            out_dir = Path.cwd()
        return _cut(opts, media, out_dir, explicit, ffmpeg, state, json_mode=json_mode)


def _cut(
    opts: ClipOptions,
    media: Path,
    out_dir: Path | None,
    explicit: list[Segment],
    ffmpeg: str,
    state: AppState,
    *,
    json_mode: bool,
) -> tuple[dict[str, object], list[WrittenClip]]:
    """Select and cut the clips for an already-local media file; the payload + clips."""
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
    return payload, written
