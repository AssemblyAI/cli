"""`assembly eval` — transcribe an evaluation dataset and score it against references.

The module is named ``evaluate`` because importing a module named ``eval`` would
shadow the builtin; the command itself registers as ``eval``. The scoring/render
logic lives in ``aai_cli.commands.evaluate._exec`` (the options/run split, see AGENTS.md).
"""

from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.evaluate import _exec as evaluate_exec
from aai_cli.commands.evaluate._exec import EvalSpeechModel
from aai_cli.core import llm
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=100,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("eval",),
)


@app.command(
    name="eval",
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            (
                "Score a model on 10 rows of a benchmark",
                "assembly eval tedlium",
            ),
            (
                "Score several benchmarks in one run",
                "assembly eval tedlium librispeech earnings22",
            ),
            (
                "Compare models on your own audio",
                "assembly eval calls.csv --speech-model universal-3-pro",
            ),
            (
                "More rows, transcribed four at a time",
                "assembly eval librispeech --limit 50 --concurrency 4",
            ),
            (
                "Evaluate non-English audio",
                "assembly eval commonvoice --subset fr --language-code fr",
            ),
            (
                "Summarize error patterns across the set",
                'assembly eval tedlium --llm-reduce "Summarize the common error patterns"',
            ),
        ]
    ),
)
def evaluate(
    ctx: typer.Context,
    datasets: list[str] = typer.Argument(
        ...,
        help="Hugging Face dataset ids, or local .csv/.jsonl manifests with audio + text columns",
    ),
    split: str | None = typer.Option(
        None, "--split", help="Hugging Face split to score (default: test)"
    ),
    subset: str | None = typer.Option(
        None, "--subset", help="Hugging Face config/subset name (e.g. a language)"
    ),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Rows to evaluate (1-100)"),
    audio_column: str | None = typer.Option(
        None, "--audio-column", help="Audio column name (default: auto-detect)"
    ),
    text_column: str | None = typer.Option(
        None, "--text-column", help="Reference text column name (default: auto-detect)"
    ),
    speech_model: EvalSpeechModel | None = typer.Option(
        None, "--speech-model", help="Speech model to evaluate"
    ),
    language_code: str | None = typer.Option(
        None, "--language-code", help="Force a language (e.g. en_us)"
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="How many items to transcribe at once (sequential by default)",
    ),
    llm_prompt: list[str] | None = typer.Option(
        None,
        "--llm",
        help="Transform each transcript through LLM Gateway before reporting (the WER "
        "score still uses the raw transcript). Repeatable: each prompt runs on the "
        "previous one's response, the first on the transcript.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    llm_reduce: list[str] | None = typer.Option(
        None,
        "--llm-reduce",
        help="Run one LLM-Gateway prompt over every item's result (a reduce). "
        "Repeatable: each runs on the previous one's output.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    json_out: bool = options.json_option("Output the rows and summary as one JSON object"),
) -> None:
    """Transcribe one or more datasets and score WER against their reference texts

    Each row's audio is transcribed, then scored against the row's reference
    text; both are normalized first (lowercased, punctuation stripped) so style
    differences don't count as errors, and the summary pools total errors over
    total reference words. Handy for picking a model: run once per
    --speech-model and compare.

    Pass several datasets to score them in one run; each is loaded, scored, and
    reported separately (under --json, one JSON object per dataset). The
    --limit/--split/--subset/--column flags apply to every dataset.

    Datasets come from the Hugging Face Hub (any public dataset its viewer
    serves with audio + reference columns; gated ones need HF_TOKEN), a local
    .csv/.jsonl manifest with audio + text columns, or a built-in benchmark
    alias that fills in the right hub id, subset, split, and columns:
    librispeech / librispeech-other (read English), tedlium (TED talks),
    earnings22 (earnings calls), spgispeech (financial calls), ami / ami-sdm
    (meetings), gigaspeech, peoples (real-world US English), commonvoice
    (English; --subset fr etc. for its 98 other locales), voxpopuli
    (parliament speech), switchboard (phone calls), expresso (expressive
    speech), loquacious, and callhome (phone calls).

    --llm runs an LLM-Gateway chain over each transcript (the WER score still
    uses the raw transcript); --llm-reduce then runs one prompt over every
    item's result to summarize patterns across the run.
    """
    opts = evaluate_exec.EvalOptions(
        datasets=datasets,
        split=split,
        subset=subset,
        limit=limit,
        audio_column=audio_column,
        text_column=text_column,
        speech_model=speech_model,
        language_code=language_code,
        concurrency=concurrency,
        llm_prompt=llm_prompt,
        llm_reduce=llm_reduce,
        model=model,
        max_tokens=max_tokens,
    )
    run_with_options(ctx, evaluate_exec.run_evaluate, opts, json=json_out)
