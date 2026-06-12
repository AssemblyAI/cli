"""`assembly eval` — transcribe an evaluation dataset and score it against references.

WER (via jiwer) against the dataset's reference texts; with ``--speaker-labels``
also DER (via pyannote.metrics) against its reference speaker turns. The module
is named ``evaluate`` because importing a module named ``eval`` would shadow the
builtin; the command itself registers as ``eval``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum

import assemblyai as aai
import typer
from rich.console import RenderableType

from aai_cli import client, der, eval_data, help_panels, jsonshape, options, output, wer
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError, NotAuthenticated, UsageError
from aai_cli.help_text import examples_epilog

app = typer.Typer()


class EvalSpeechModel(StrEnum):
    """The current-generation models, requested via the SDK's ``speech_models``
    list parameter (its legacy ``SpeechModel`` enum predates them)."""

    universal_3_pro = "universal-3-pro"
    universal_2 = "universal-2"


def _pct(value: object) -> str:
    return f"{jsonshape.as_float(value):.2%}"


def _hypothesis_turns(transcript: aai.Transcript) -> list[der.Turn]:
    """The transcript's diarized utterances as DER hypothesis turns (ms → seconds)."""
    return [
        der.Turn(
            speaker=str(getattr(utterance, "speaker", "")),
            start=jsonshape.as_float(getattr(utterance, "start", None)) / 1000,
            end=jsonshape.as_float(getattr(utterance, "end", None)) / 1000,
        )
        for utterance in jsonshape.object_list(getattr(transcript, "utterances", None))
    ]


@dataclass(frozen=True)
class _ItemResult:
    """One scored row: the emitted dict plus the scores kept for pooling."""

    row: dict[str, object]
    words: wer.Score | None
    speakers: der.DerScore | None


def _failed_result(item: eval_data.EvalItem, err: CLIError) -> _ItemResult:
    """A row whose transcription failed: the error rides along, no scores pooled."""
    return _ItemResult(row={"item": item.item_id, "error": err.message}, words=None, speakers=None)


def _score_item(
    item: eval_data.EvalItem, transcript: aai.Transcript, *, collar: float
) -> _ItemResult:
    row: dict[str, object] = {"item": item.item_id}
    words: wer.Score | None = None
    speakers: der.DerScore | None = None
    if item.reference is not None:
        words = wer.score(item.reference, str(transcript.text or ""))
        row.update({"words": words.words, "errors": words.errors, "wer": words.wer})
    if item.turns is not None:
        speakers = der.score(item.turns, _hypothesis_turns(transcript), collar=collar)
        row["der"] = speakers.der
    return _ItemResult(row=row, words=words, speakers=speakers)


def _pooled_metrics(results: list[_ItemResult]) -> dict[str, object]:
    """The summary scores pooled over the scored rows (failed rows carry none)."""
    metrics: dict[str, object] = {}
    word_scores = [result.words for result in results if result.words is not None]
    if word_scores:
        total = wer.pooled(word_scores)
        metrics.update({"words": total.words, "errors": total.errors, "wer": total.wer})
    der_scores = [result.speakers for result in results if result.speakers is not None]
    if der_scores:
        pooled = der.pooled(der_scores)
        metrics["der"] = pooled.der
        metrics["der_breakdown"] = {
            "missed": pooled.missed / pooled.total,
            "false_alarm": pooled.false_alarm / pooled.total,
            "confusion": pooled.confusion / pooled.total,
        }
    return metrics


def _transcribe_one(
    api_key: str, item: eval_data.EvalItem, config: aai.TranscriptionConfig
) -> aai.Transcript | CLIError:
    """One item's outcome: its transcript, or the CLIError it failed with.

    A bad item must not discard the other (paid) items, so per-item failures
    are recorded rather than raised — except ``NotAuthenticated`` (one rejected
    key fails every row identically) and non-CLIError bugs, which propagate and
    abort the run.
    """
    try:
        return client.transcribe(api_key, item.audio, config=config)
    except NotAuthenticated:
        raise
    except CLIError as err:
        return err


def _concurrent_transcripts(
    api_key: str,
    items: list[eval_data.EvalItem],
    *,
    transcription_config: aai.TranscriptionConfig,
    concurrency: int,
) -> list[aai.Transcript | CLIError]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_transcribe_one, api_key, item, transcription_config) for item in items
        ]
        for future in as_completed(futures):
            if (exc := future.exception()) is not None:
                # Only aborting failures escape _transcribe_one: drop the
                # not-yet-started items rather than burn an API call on each.
                pool.shutdown(cancel_futures=True)
                raise exc
        return [future.result() for future in futures]


def _transcripts(
    api_key: str,
    items: list[eval_data.EvalItem],
    *,
    transcription_config: aai.TranscriptionConfig,
    concurrency: int,
    json_mode: bool,
    quiet: bool,
) -> list[aai.Transcript | CLIError]:
    """Each item's transcript — or the CLIError it failed with — in dataset order.

    Sequential by default, with a per-item spinner; ``--concurrency`` fans the
    API calls out across a thread pool (see ``_transcribe_one`` for which
    failures are per-item outcomes and which abort the run).
    """
    if concurrency == 1:
        outcomes: list[aai.Transcript | CLIError] = []
        for index, item in enumerate(items, start=1):
            with output.status(
                f"[{index}/{len(items)}] Transcribing {item.item_id}…",
                json_mode=json_mode,
                quiet=quiet,
            ):
                outcomes.append(_transcribe_one(api_key, item, transcription_config))
        return outcomes
    with output.status(
        f"Transcribing {len(items)} items (concurrency {concurrency})…",
        json_mode=json_mode,
        quiet=quiet,
    ):
        return _concurrent_transcripts(
            api_key, items, transcription_config=transcription_config, concurrency=concurrency
        )


def _payload(
    label: str, speech_model: EvalSpeechModel | None, results: list[_ItemResult]
) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset": label,
        "speech_model": speech_model.value if speech_model else None,
        "items": len(results),
        "rows": [result.row for result in results],
    }
    payload.update(_pooled_metrics(results))
    failed = sum(1 for result in results if "error" in result.row)
    if failed:
        payload["failed"] = failed
    return payload


def _summary(payload: dict[str, object]) -> str:
    parts: list[str] = []
    if "wer" in payload:
        errors = jsonshape.as_int(payload.get("errors"))
        noun = "error" if errors == 1 else "errors"
        parts.append(
            f"WER {_pct(payload.get('wer'))} ({errors} {noun} / {payload.get('words')} words)"
        )
    if "der" in payload:
        breakdown = jsonshape.as_mapping(payload.get("der_breakdown")) or {}
        parts.append(
            f"DER {_pct(payload.get('der'))} (missed {_pct(breakdown.get('missed'))} · "
            f"false alarm {_pct(breakdown.get('false_alarm'))} · "
            f"confusion {_pct(breakdown.get('confusion'))})"
        )
    return output.heading("   ".join(parts))


def _cell(row: dict[str, object], key: str) -> str:
    """The row's value as table text — blank when absent (e.g. a failed row's scores)."""
    return str(row[key]) if key in row else ""


def _pct_cell(row: dict[str, object], key: str) -> str:
    return _pct(row[key]) if key in row else ""


def _render(payload: dict[str, object]) -> RenderableType:
    has_wer = "wer" in payload
    has_der = "der" in payload
    has_failed = "failed" in payload
    columns = [
        "ITEM",
        *(["WORDS", "ERRORS", "WER"] if has_wer else []),
        *(["DER"] if has_der else []),
        *(["ERROR"] if has_failed else []),
    ]
    table = output.data_table(*columns)
    for row in jsonshape.mapping_list(payload.get("rows")):
        cells = [str(row.get("item"))]
        if has_wer:
            cells += [_cell(row, "words"), _cell(row, "errors"), _pct_cell(row, "wer")]
        if has_der:
            cells.append(_pct_cell(row, "der"))
        if has_failed:
            cells.append(_cell(row, "error"))
        table.add_row(*cells)
    model = payload.get("speech_model") or "default model"
    return output.stack(
        output.muted(f"{payload.get('dataset')} · {model}"), table, _summary(payload)
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
                "Compare models on your own audio",
                "assembly eval calls.csv --speech-model universal-3-pro",
            ),
            (
                "Score diarization too (WER + DER)",
                "assembly eval agent-calls.jsonl --speaker-labels",
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
                "DER on a diarization benchmark",
                "assembly eval callhome --speaker-labels",
            ),
        ]
    ),
)
def evaluate(
    ctx: typer.Context,
    dataset: str = typer.Argument(
        ...,
        help="Hugging Face dataset id, or a local .csv/.jsonl manifest with audio + text columns.",
    ),
    split: str | None = typer.Option(
        None, "--split", help="Hugging Face split to score (default: test)."
    ),
    subset: str | None = typer.Option(
        None, "--subset", help="Hugging Face config/subset name (e.g. a language)."
    ),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Rows to evaluate (1-100)."),
    audio_column: str | None = typer.Option(
        None, "--audio-column", help="Audio column name (default: auto-detect)."
    ),
    text_column: str | None = typer.Option(
        None, "--text-column", help="Reference text column name (default: auto-detect)."
    ),
    speech_model: EvalSpeechModel | None = typer.Option(
        None, "--speech-model", help="Speech model to evaluate."
    ),
    language_code: str | None = typer.Option(
        None, "--language-code", help="Force a language (e.g. en_us)."
    ),
    speaker_labels: bool = typer.Option(
        False,
        "--speaker-labels",
        help="Diarize and also score DER against the dataset's reference speaker turns (speakers/timestamps_start/timestamps_end columns, in seconds).",
    ),
    collar: float | None = typer.Option(
        None,
        "--collar",
        min=0.0,
        help="DER forgiveness (seconds) around each reference turn boundary "
        "(default: 1.0; needs --speaker-labels).",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="How many items to transcribe at once (sequential by default).",
    ),
    json_out: bool = options.json_option("Output the rows and summary as one JSON object."),
) -> None:
    """Transcribe an evaluation dataset and score WER against its reference texts.

    Each row's audio is transcribed, then scored against the row's reference
    text; both are normalized first (lowercased, punctuation stripped) so style
    differences don't count as errors, and the summary pools total errors over
    total reference words. Handy for picking a model: run once per
    --speech-model and compare. --speaker-labels also diarizes and scores DER
    against reference speaker turns.

    Datasets come from the Hugging Face Hub (any public dataset its viewer
    serves with audio + reference columns; gated ones need HF_TOKEN), a local
    .csv/.jsonl manifest with audio + text columns, or a built-in benchmark
    alias that fills in the right hub id, subset, split, and columns:
    librispeech / librispeech-other (read English), tedlium (TED talks),
    earnings22 (earnings calls), spgispeech (financial calls), ami / ami-sdm
    (meetings), gigaspeech, peoples (real-world US English), commonvoice
    (English; --subset fr etc. for its 98 other locales), voxpopuli
    (parliament speech), switchboard (phone calls), expresso (expressive
    speech), loquacious, and callhome (phone calls with speaker turns, for
    --speaker-labels).
    """

    def body(state: AppState, json_mode: bool) -> None:
        if collar is not None and not speaker_labels:
            raise UsageError(
                "--collar only applies when diarization is being scored.",
                suggestion="Add --speaker-labels.",
            )
        # Resolve credentials before any dataset download: a signed-out user must
        # not pull the whole dataset only to fail at the first transcription.
        api_key = state.resolve_api_key()
        data = eval_data.load(
            dataset,
            split=split,
            subset=subset,
            audio_column=audio_column,
            text_column=text_column,
            limit=limit,
            with_speakers=speaker_labels,
        )
        transcription_config = aai.TranscriptionConfig(
            speech_models=[speech_model.value] if speech_model else None,
            language_code=language_code,
            speaker_labels=speaker_labels or None,
        )
        outcomes = _transcripts(
            api_key,
            data.items,
            transcription_config=transcription_config,
            concurrency=concurrency,
            json_mode=json_mode,
            quiet=state.quiet,
        )
        results = [
            _failed_result(item, outcome)
            if isinstance(outcome, CLIError)
            else _score_item(item, outcome, collar=collar if collar is not None else 1.0)
            for item, outcome in zip(
                data.items,
                outcomes,
                strict=True,  # pragma: no mutate (defensive invariant; _transcripts returns one outcome per item)
            )
        ]
        payload = _payload(data.label, speech_model, results)
        output.emit(payload, _render, json_mode=json_mode)
        failed = jsonshape.as_int(payload.get("failed"))
        if failed:
            raise CLIError(
                f"{failed} of {len(results)} items failed to transcribe.",
                error_type="eval_failed",
                suggestion="The summary covers only the items that transcribed.",
            )

    run_command(ctx, body, json=json_out)
