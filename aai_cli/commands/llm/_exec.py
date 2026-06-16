"""Run logic for `assembly llm`: the options/run split (see AGENTS.md).

The command module (aai_cli/commands/llm/__init__.py) only parses argv — it builds an
``LlmOptions`` and hands it to ``run_llm`` via ``context.run_command``, so tests can
drive one-shot and --follow behavior by constructing options directly, with no
CliRunner argv round-trip. (``aai_cli/llm.py`` is the gateway client itself and is
rich-free by architecture contract, so the rendering-aware run path lives here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli.app.context import AppState
from aai_cli.core import choices, client, errors, stdio
from aai_cli.core import llm as gateway
from aai_cli.core.errors import UsageError
from aai_cli.ui import output
from aai_cli.ui.follow import FollowRenderer

_FOLLOW_STDIN_MESSAGE = (
    "--follow needs transcript text piped on stdin, e.g. "
    '`assembly stream -o text | assembly llm -f "summarize action items as I talk"`.'
)


@dataclass(frozen=True)
class LlmOptions:
    """Every `assembly llm` prompt flag as plain data.

    ``--list-models`` is excluded: it dispatches to its own auth-free body in the
    command module. ``--json`` is excluded: run_command resolves it into the
    ``json_mode`` argument.
    """

    prompt: str | None
    model: str
    transcript_id: str | None
    system: str | None
    follow: bool
    output_field: choices.TextOrJson | None
    max_tokens: int
    # Raw --config KEY=VALUE pairs; parsed (and validated) once in run_llm.
    config_kv: tuple[str, ...] = ()
    # Input files read as the prompt's context (header-prefixed, concatenated).
    files: tuple[Path, ...] = ()


def _validate_follow_args(
    prompt: str | None,
    output_field: str | None,
    transcript_id: str | None,
    files: tuple[Path, ...],
) -> str:
    """Reject flag combinations that don't apply to --follow's live-panel mode.

    Returns the validated (non-empty) prompt so the caller has a plain ``str``.
    """
    if not prompt:
        raise UsageError("Provide a prompt to run over the streamed transcript.")
    if output_field is not None:
        raise UsageError(
            "--output applies to one-shot mode; --follow renders a live panel "
            "(or NDJSON when piped)."
        )
    if transcript_id:
        raise UsageError(
            "--follow runs over live transcript text piped on stdin; it can't be "
            "combined with --transcript-id."
        )
    if files:
        raise UsageError(
            "--follow runs over live transcript text piped on stdin; it can't be "
            "combined with file arguments."
        )
    if not stdio.stdin_is_piped():
        raise UsageError(_FOLLOW_STDIN_MESSAGE)
    return prompt


def _read_files(files: tuple[Path, ...]) -> str:
    """Read each file and join them, each prefixed with a ``===== name =====`` header.

    The header names each source (the file's stem) so a multi-file prompt can cite
    which note an answer came from; it's applied uniformly, even for a single file,
    so the format the model sees is predictable. A missing or unreadable path is a
    usage error raised before any auth or network — the same fail-fast ordering as
    the --transcript-id check.
    """
    sections: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(
                f"Couldn't read {path}: {exc.strerror or exc}.",
                suggestion="Check the path points at a readable file.",
            ) from exc
        sections.append(f"===== {path.stem} =====\n{text}")
    return "\n\n".join(sections)


def _input_text(
    state: AppState, transcript_id: str | None, files: tuple[Path, ...], *, json_mode: bool
) -> str | None:
    """Resolve the inline text the prompt operates on for one-shot mode.

    Three possible sources, in priority order: an explicit --transcript-id (injected
    server-side, so this returns None), one or more file arguments (read and
    concatenated), or text piped on stdin. A higher-priority source present alongside
    a lower one ignores the lower with a visible warning (suppressed by --quiet,
    structured under --json).
    """
    if transcript_id is not None:
        # Same cheap local id check as `transcripts get`, before auth or network.
        client.validate_transcript_id(transcript_id)
        ignored = _ignored_sources(files, stdio.stdin_is_piped())
        if ignored and not state.quiet:
            output.emit_warning(
                f"Ignoring {ignored}; --transcript-id takes priority.", json_mode=json_mode
            )
        return None
    if files:
        if stdio.stdin_is_piped() and not state.quiet:
            output.emit_warning(
                "Ignoring piped stdin; file arguments take priority.", json_mode=json_mode
            )
        return _read_files(files)
    return stdio.piped_stdin_text()


def _ignored_sources(files: tuple[Path, ...], stdin_piped: bool) -> str | None:
    """Name the lower-priority input sources present alongside --transcript-id, for the
    warning — or None when there's nothing to ignore."""
    sources: list[str] = []
    if files:
        sources.append("file arguments")
    if stdin_piped:
        sources.append("piped stdin")
    return " and ".join(sources) or None


def _run_follow(
    opts: LlmOptions, state: AppState, extra: dict[str, object], *, json_mode: bool
) -> None:
    prompt_text = _validate_follow_args(
        opts.prompt, opts.output_field, opts.transcript_id, opts.files
    )
    api_key = state.resolve_api_key()

    def ask(transcript_text: str) -> str:
        messages = gateway.build_messages(
            prompt_text, system=opts.system, transcript_text=transcript_text
        )
        response = gateway.complete(
            api_key, model=opts.model, messages=messages, max_tokens=opts.max_tokens, extra=extra
        )
        return gateway.content_of(response)

    transcript: list[str] = []
    with FollowRenderer(json_mode=json_mode) as render:
        # Ctrl-C is the normal "stop watching" signal: exit 130 (cancel) rather than
        # masquerading as a clean finish — the renderer's panel closes on the way out.
        try:
            for turn in stdio.iter_piped_stdin_lines():
                transcript.append(turn)
                render(ask("\n".join(transcript)), len(transcript))
        except KeyboardInterrupt:
            raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
    if not transcript:
        # An empty pipe (`assembly llm -f "…" </dev/null`) would otherwise exit 0
        # silently, having asked nothing.
        raise UsageError(_FOLLOW_STDIN_MESSAGE)


def _run_oneshot(
    opts: LlmOptions, state: AppState, extra: dict[str, object], *, json_mode: bool
) -> None:
    if not opts.prompt:
        raise UsageError(
            "Provide a prompt.",
            suggestion="Or pass --list-models to see available models.",
        )
    prompt_text = opts.prompt
    input_text = _input_text(state, opts.transcript_id, opts.files, json_mode=json_mode)
    api_key = state.resolve_api_key()
    messages = gateway.build_messages(
        prompt_text,
        system=opts.system,
        transcript_id=opts.transcript_id,
        transcript_text=input_text,
    )
    response = gateway.complete(
        api_key,
        model=opts.model,
        messages=messages,
        max_tokens=opts.max_tokens,
        transcript_id=opts.transcript_id,
        extra=extra,
    )
    content = gateway.content_of(response)
    if opts.output_field == "text":
        # Just the answer, raw — so `… | assembly llm -o text "…" | next` composes cleanly.
        output.emit_text(content)
        return
    output.emit(
        {"model": opts.model, "output": content, "usage": gateway.usage_of(response)},
        lambda d: escape(str(d["output"])),
        json_mode=json_mode or opts.output_field == "json",
    )


def run_llm(opts: LlmOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly llm` invocation (one-shot or --follow) from parsed flags."""
    # Parsed before any stdin/network work so a malformed pair fails fast.
    extra = gateway.parse_gateway_overrides(opts.config_kv)
    if opts.follow:
        _run_follow(opts, state, extra, json_mode=json_mode)
    else:
        _run_oneshot(opts, state, extra, json_mode=json_mode)
