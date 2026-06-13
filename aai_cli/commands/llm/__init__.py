from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_command
from aai_cli.commands.llm import _exec as llm_exec
from aai_cli.core import choices
from aai_cli.core import llm as gateway
from aai_cli.core.errors import UsageError
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=60,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("llm",),
)


def _list_models(output_field: choices.TextOrJson | None, json_mode: bool) -> None:
    """The --list-models body, routed through run_command so --json yields a
    machine-readable array instead of the human list; needs no auth. Rejects -o
    (it only applies to one-shot mode, mirroring how --follow rejects it)."""
    if output_field is not None:
        raise UsageError(
            "--output applies to one-shot mode; --list-models prints the plain "
            "list (use --json for a machine-readable array)."
        )
    output.emit(list(gateway.KNOWN_MODELS), "\n".join, json_mode=json_mode)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            (
                "Ask about a past transcript",
                'assembly llm "summarize the key decisions" --transcript-id 5551234-abcd',
            ),
            ("Pipe any text in", 'echo "meeting notes" | assembly llm "turn into action items"'),
            (
                "Pick a model and add a system prompt",
                'assembly llm "draft a follow-up email" --model claude-opus-4-7 --system "Be concise."',
            ),
            ("See available models", "assembly llm --list-models"),
        ]
    ),
)
def llm(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(None, help="The prompt to send to the model"),
    # Note: text piped on stdin is injected into the prompt (e.g. `cat notes | assembly llm "summarize"`).
    model: str = typer.Option(
        gateway.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model",
        autocompletion=gateway.complete_model,
    ),
    transcript_id: str | None = typer.Option(
        None, "--transcript-id", help="Inject this transcript's text into the prompt"
    ),
    system: str | None = typer.Option(None, "--system", help="Optional system prompt"),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Re-run the prompt over a growing transcript piped on stdin, refreshing "
        "the answer in place on every finalized turn (e.g. assembly stream -o text | assembly "
        'llm -f "summarize action items as I talk"). Ctrl-C to stop.',
    ),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field of the result: text (just the answer, pipe-friendly) or json",
    ),
    max_tokens: int = typer.Option(
        gateway.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens to generate", min=1
    ),
    config_kv: list[str] | None = typer.Option(
        None,
        "--config",
        help="Set any extra gateway request field: KEY=VALUE, repeatable "
        "(e.g. --config temperature=0.2). Values parse as JSON, else literal text.",
    ),
    list_models: bool = typer.Option(False, "--list-models", help="Print known models and exit"),
    json_out: bool = options.json_option("Output raw JSON (one object per turn in --follow mode)"),
) -> None:
    """Send a prompt to AssemblyAI's LLM Gateway and print the reply

    With --transcript-id the transcript's text is injected server-side, so you
    can ask questions about a past transcription (e.g. assembly llm "summarize" --transcript-id ID).
    """

    if list_models:
        run_command(
            ctx, lambda _state, json_mode: _list_models(output_field, json_mode), json=json_out
        )
        return

    opts = llm_exec.LlmOptions(
        prompt=prompt,
        model=model,
        transcript_id=transcript_id,
        system=system,
        follow=follow,
        output_field=output_field,
        max_tokens=max_tokens,
        config_kv=tuple(config_kv or ()),
    )
    run_command(
        ctx,
        lambda state, json_mode: llm_exec.run_llm(opts, state, json_mode=json_mode),
        json=json_out,
    )
