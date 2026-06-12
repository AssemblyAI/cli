"""Run logic for `assembly eval`: transcribe a dataset and score WER against references.

The command module (aai_cli/commands/evaluate.py) only parses argv — it builds an
``EvalOptions`` and hands it to ``run_evaluate`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive the scoring and rendering by
constructing options directly instead of round-tripping argv.

WER (via jiwer) against the dataset's reference texts. The sibling module is named
``evaluate`` because importing a module named ``eval`` would shadow the builtin;
the command itself registers as ``eval``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum

import assemblyai as aai
from rich.console import RenderableType

from aai_cli import client, eval_data, jsonshape, output, wer
from aai_cli.context import AppState
from aai_cli.errors import CLIError, NotAuthenticated


class EvalSpeechModel(StrEnum):
    """The current-generation models, requested via the SDK's ``speech_models``
    list parameter (its legacy ``SpeechModel`` enum predates them)."""

    universal_3_pro = "universal-3-pro"
    universal_2 = "universal-2"


@dataclass(frozen=True)
class EvalOptions:
    """Every `assembly eval` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    dataset: str
    split: str | None
    subset: str | None
    limit: int
    audio_column: str | None
    text_column: str | None
    speech_model: EvalSpeechModel | None
    language_code: str | None
    concurrency: int


def _pct(value: object) -> str:
    return f"{jsonshape.as_float(value):.2%}"


@dataclass(frozen=True)
class _ItemResult:
    """One scored row: the emitted dict plus the score kept for pooling."""

    row: dict[str, object]
    words: wer.Score | None


def _failed_result(item: eval_data.EvalItem, err: CLIError) -> _ItemResult:
    """A row whose transcription failed: the error rides along, no scores pooled."""
    return _ItemResult(row={"item": item.item_id, "error": err.message}, words=None)


def _score_item(item: eval_data.EvalItem, transcript: aai.Transcript) -> _ItemResult:
    words = wer.score(item.reference, str(transcript.text or ""))
    row: dict[str, object] = {
        "item": item.item_id,
        "words": words.words,
        "errors": words.errors,
        "wer": words.wer,
    }
    return _ItemResult(row=row, words=words)


def _pooled_metrics(results: list[_ItemResult]) -> dict[str, object]:
    """The summary scores pooled over the scored rows (failed rows carry none)."""
    metrics: dict[str, object] = {}
    word_scores = [result.words for result in results if result.words is not None]
    if word_scores:
        total = wer.pooled(word_scores)
        metrics.update({"words": total.words, "errors": total.errors, "wer": total.wer})
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
    return output.heading("   ".join(parts))


def _cell(row: dict[str, object], key: str) -> str:
    """The row's value as table text — blank when absent (e.g. a failed row's scores)."""
    return str(row[key]) if key in row else ""


def _pct_cell(row: dict[str, object], key: str) -> str:
    return _pct(row[key]) if key in row else ""


def _render(payload: dict[str, object]) -> RenderableType:
    has_wer = "wer" in payload
    has_failed = "failed" in payload
    columns = [
        "ITEM",
        *(["WORDS", "ERRORS", "WER"] if has_wer else []),
        *(["ERROR"] if has_failed else []),
    ]
    table = output.data_table(*columns)
    for row in jsonshape.mapping_list(payload.get("rows")):
        cells = [str(row.get("item"))]
        if has_wer:
            cells += [_cell(row, "words"), _cell(row, "errors"), _pct_cell(row, "wer")]
        if has_failed:
            cells.append(_cell(row, "error"))
        table.add_row(*cells)
    model = payload.get("speech_model") or "default model"
    return output.stack(
        output.muted(f"{payload.get('dataset')} · {model}"), table, _summary(payload)
    )


def run_evaluate(opts: EvalOptions, state: AppState, *, json_mode: bool) -> None:
    """Transcribe an evaluation dataset and score WER against its reference texts."""
    # Resolve credentials before any dataset download: a signed-out user must
    # not pull the whole dataset only to fail at the first transcription.
    api_key = state.resolve_api_key()
    data = eval_data.load(
        opts.dataset,
        split=opts.split,
        subset=opts.subset,
        audio_column=opts.audio_column,
        text_column=opts.text_column,
        limit=opts.limit,
    )
    transcription_config = aai.TranscriptionConfig(
        speech_models=[opts.speech_model.value] if opts.speech_model else None,
        language_code=opts.language_code,
    )
    outcomes = _transcripts(
        api_key,
        data.items,
        transcription_config=transcription_config,
        concurrency=opts.concurrency,
        json_mode=json_mode,
        quiet=state.quiet,
    )
    results = [
        _failed_result(item, outcome)
        if isinstance(outcome, CLIError)
        else _score_item(item, outcome)
        for item, outcome in zip(
            data.items,
            outcomes,
            strict=True,  # pragma: no mutate (defensive invariant; _transcripts returns one outcome per item)
        )
    ]
    payload = _payload(data.label, opts.speech_model, results)
    output.emit(payload, _render, json_mode=json_mode)
    failed = jsonshape.as_int(payload.get("failed"))
    if failed:
        raise CLIError(
            f"{failed} of {len(results)} items failed to transcribe.",
            error_type="eval_failed",
            suggestion="The summary covers only the items that transcribed.",
        )
