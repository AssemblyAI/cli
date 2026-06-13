"""Run logic for `assembly llm`: the options/run split (see AGENTS.md).

The command module (aai_cli/commands/llm/__init__.py) only parses argv — it builds an
``LlmOptions`` and hands it to ``run_llm`` via ``context.run_command``, so tests can
drive one-shot and --follow behavior by constructing options directly, with no
CliRunner argv round-trip. (``aai_cli/llm.py`` is the gateway client itself and is
rich-free by architecture contract, so the rendering-aware run path lives here.)
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.markup import escape

from aai_cli import choices, client, output, stdio
from aai_cli import llm as gateway
from aai_cli.context import AppState
from aai_cli.errors import UsageError
from aai_cli.follow import FollowRenderer

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


def _validate_follow_args(
    prompt: str | None, output_field: str | None, transcript_id: str | None
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
    if not stdio.stdin_is_piped():
        raise UsageError(_FOLLOW_STDIN_MESSAGE)
    return prompt


def _stdin_transcript_text(
    state: AppState, transcript_id: str | None, *, json_mode: bool
) -> str | None:
    """Resolve the inline transcript text for one-shot mode.

    Text piped on stdin becomes the content the prompt operates on, unless an
    explicit --transcript-id is given — that injects server-side and takes
    priority, so piped text is ignored with a visible warning (suppressed by
    --quiet, structured under --json).
    """
    if transcript_id is None:
        return stdio.piped_stdin_text()
    # Same cheap local id check as `transcripts get`, before auth or network.
    client.validate_transcript_id(transcript_id)
    if stdio.stdin_is_piped() and not state.quiet:
        output.emit_warning(
            "Ignoring piped stdin; --transcript-id takes priority.", json_mode=json_mode
        )
    return None


def _run_follow(
    opts: LlmOptions, state: AppState, extra: dict[str, object], *, json_mode: bool
) -> None:
    prompt_text = _validate_follow_args(opts.prompt, opts.output_field, opts.transcript_id)
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
    interrupted = False
    with FollowRenderer(json_mode=json_mode) as render:
        # Ctrl-C is the normal "stop watching" signal -> exit cleanly (code 0).
        try:
            for turn in stdio.iter_piped_stdin_lines():
                transcript.append(turn)
                render(ask("\n".join(transcript)), len(transcript))
        except KeyboardInterrupt:
            interrupted = True
    if not transcript and not interrupted:
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
    stdin_text = _stdin_transcript_text(state, opts.transcript_id, json_mode=json_mode)
    api_key = state.resolve_api_key()
    messages = gateway.build_messages(
        prompt_text,
        system=opts.system,
        transcript_id=opts.transcript_id,
        transcript_text=stdin_text,
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
