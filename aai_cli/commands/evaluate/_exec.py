"""Run logic for `assembly eval`: transcribe a dataset and score WER against references.

The command module (aai_cli/commands/evaluate/__init__.py) only parses argv — it builds an
``EvalOptions`` and hands it to ``run_evaluate`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive the scoring and rendering by
constructing options directly instead of round-tripping argv.

WER (via jiwer) against the dataset's reference texts. The sibling module is named
``evaluate`` because importing a module named ``eval`` would shadow the builtin;
the command itself registers as ``eval``.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum

import assemblyai as aai
from rich.console import RenderableType
from rich.markup import escape

from aai_cli.app.context import AppState
from aai_cli.commands.evaluate import _data as eval_data
from aai_cli.core import client, jsonshape, wer
from aai_cli.core import llm as gateway
from aai_cli.core.errors import CLIError, NotAuthenticated
from aai_cli.ui import output


class EvalSpeechModel(StrEnum):
    """The current-generation models, requested via the SDK's ``speech_models``
    list parameter (its legacy ``SpeechModel`` enum predates them)."""

    universal_3_pro = "universal-3-pro"
    universal_2 = "universal-2"


@dataclass(frozen=True)
class EvalOptions:
    """Every `assembly eval` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    datasets: list[str]
    split: str | None
    subset: str | None
    limit: int
    audio_column: str | None
    text_column: str | None
    speech_model: EvalSpeechModel | None
    language_code: str | None
    concurrency: int
    llm_prompt: list[str] | None
    llm_reduce: list[str] | None
    model: str
    max_tokens: int

    def llm_options(self) -> _LlmOptions:
        """The ``--llm`` / ``--llm-reduce`` chain settings as plain data."""
        return _LlmOptions(
            prompts=list(self.llm_prompt or []),
            reduce_prompts=list(self.llm_reduce or []),
            model=self.model,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class _LlmOptions:
    """The post-transcription LLM-Gateway transform: the per-item ``--llm`` chain
    (a *map*) and the across-items ``--llm-reduce`` chain (a *reduce*), plus the
    gateway model + token budget both run under."""

    prompts: list[str]
    reduce_prompts: list[str]
    model: str
    max_tokens: int


def _pct(value: object) -> str:
    return f"{jsonshape.as_float(value):.2%}"


def _secs(value: object) -> str:
    """A latency in seconds, formatted for display."""
    return f"{jsonshape.as_float(value):.2f}s"


def _percentile(values: list[float], q: float) -> float:
    """The q-quantile (q in [0, 1]) of ``values``, linearly interpolated between
    the two closest ranks (numpy's default method). ``values`` must be non-empty."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass(frozen=True)
class _ItemResult:
    """One scored row: the emitted dict plus the score and latency kept for pooling.

    ``hypothesis`` is the transcript text (``None`` for a failed row) — kept so the
    optional ``--llm`` map / ``--llm-reduce`` reduce can run over it after scoring.
    """

    row: dict[str, object]
    words: wer.Score | None
    latency: float
    hypothesis: str | None = None


def _failed_result(item: eval_data.EvalItem, err: CLIError, latency: float) -> _ItemResult:
    """A row whose transcription failed: the error and latency ride along, no scores pooled."""
    return _ItemResult(
        row={"item": item.item_id, "error": err.message, "latency": latency},
        words=None,
        latency=latency,
    )


def _score_item(
    item: eval_data.EvalItem, transcript: aai.Transcript, latency: float
) -> _ItemResult:
    hypothesis = str(transcript.text or "")
    words = wer.score(item.reference, hypothesis)
    row: dict[str, object] = {
        "item": item.item_id,
        "words": words.words,
        "errors": words.errors,
        "wer": words.wer,
        "latency": latency,
    }
    return _ItemResult(row=row, words=words, latency=latency, hypothesis=hypothesis)


def _pooled_metrics(results: list[_ItemResult]) -> dict[str, object]:
    """The summary metrics: WER pooled over the scored rows (failed rows carry none),
    and the latency distribution over every row that ran a transcription."""
    metrics: dict[str, object] = {}
    word_scores = [result.words for result in results if result.words is not None]
    if word_scores:
        total = wer.pooled(word_scores)
        metrics.update({"words": total.words, "errors": total.errors, "wer": total.wer})
    latencies = [result.latency for result in results]
    if latencies:
        metrics["latency_p50"] = _percentile(latencies, 0.5)
        metrics["latency_p90"] = _percentile(latencies, 0.9)
    return metrics


@dataclass(frozen=True)
class _Timed:
    """One transcription's outcome paired with its wall-clock latency in seconds."""

    outcome: aai.Transcript | CLIError
    latency: float


def _transcribe_one(
    api_key: str, item: eval_data.EvalItem, config: aai.TranscriptionConfig
) -> _Timed:
    """One item's timed outcome: its transcript (or the CLIError it failed with) and
    the wall-clock latency of the request.

    A bad item must not discard the other (paid) items, so per-item failures
    are recorded rather than raised — except ``NotAuthenticated`` (one rejected
    key fails every row identically) and non-CLIError bugs, which propagate and
    abort the run.
    """
    start = time.perf_counter()
    try:
        outcome: aai.Transcript | CLIError = client.transcribe(api_key, item.audio, config=config)
    except NotAuthenticated:
        raise
    except CLIError as err:
        outcome = err
    return _Timed(outcome=outcome, latency=time.perf_counter() - start)


def _concurrent_transcripts(
    api_key: str,
    items: list[eval_data.EvalItem],
    *,
    transcription_config: aai.TranscriptionConfig,
    concurrency: int,
) -> list[_Timed]:
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
) -> list[_Timed]:
    """Each item's timed transcript — or the CLIError it failed with — in dataset order.

    Sequential by default, with a per-item spinner; ``--concurrency`` fans the
    API calls out across a thread pool (see ``_transcribe_one`` for which
    failures are per-item outcomes and which abort the run).
    """
    if concurrency == 1:
        outcomes: list[_Timed] = []
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


def _run_llm_map(
    api_key: str,
    results: list[_ItemResult],
    llm_opts: _LlmOptions,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    """Run the ``--llm`` chain over each transcribed row and attach it under ``llm``.

    A *map*: the chain runs over the row's transcript text (inline, like
    ``stream --llm``) and lands as ``{"model", "steps"}`` on the row — the WER score
    is untouched. Failed rows have no transcript, so they're skipped.
    """
    scored = [result for result in results if result.hypothesis is not None]
    with output.status(
        f"Running --llm over {len(scored)} transcripts…", json_mode=json_mode, quiet=quiet
    ):
        for result in scored:
            steps = gateway.run_chain_steps(
                api_key,
                llm_opts.prompts,
                transcript_text=result.hypothesis,
                model=llm_opts.model,
                max_tokens=llm_opts.max_tokens,
            )
            result.row["llm"] = {"model": llm_opts.model, "steps": steps}


def _reduce_input(result: _ItemResult) -> str:
    """A row's contribution to the reduce: its last ``--llm`` output, else its transcript."""
    llm_data = jsonshape.as_mapping(result.row.get("llm"))
    if llm_data is not None:
        steps = jsonshape.mapping_list(llm_data.get("steps"))
        if steps:
            return str(steps[-1].get("output", "") or "")
    return result.hypothesis or ""


def _gather_reduce_inputs(results: list[_ItemResult]) -> str:
    """Concatenate every transcribed row's reduce input under an item header."""
    blocks: list[str] = []
    for result in results:
        if result.hypothesis is None:
            continue
        text = _reduce_input(result)
        if text:
            blocks.append(f"### Item: {result.row.get('item')}\n{text}")
    return "\n\n".join(blocks)


def _run_reduce(
    api_key: str,
    results: list[_ItemResult],
    llm_opts: _LlmOptions,
    *,
    json_mode: bool,
    quiet: bool,
) -> dict[str, object] | None:
    """Run the ``--llm-reduce`` chain once over every row's result; the payload entry.

    ``None`` when there's nothing to aggregate (every row failed or transcribed to
    empty text) so the caller skips the (billable) gateway call and the payload key.
    """
    combined = _gather_reduce_inputs(results)
    if not combined:
        output.emit_warning(
            "Nothing to reduce: no transcript text across items.", json_mode=json_mode
        )
        return None
    with output.status("Running --llm-reduce over all items…", json_mode=json_mode, quiet=quiet):
        result = gateway.run_chain(
            api_key,
            llm_opts.reduce_prompts,
            transcript_text=combined,
            model=llm_opts.model,
            max_tokens=llm_opts.max_tokens,
        )
    return {"model": llm_opts.model, "prompts": llm_opts.reduce_prompts, "output": result}


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
    if "latency_p50" in payload:
        parts.append(
            f"latency p50 {_secs(payload.get('latency_p50'))}"
            f" · p90 {_secs(payload.get('latency_p90'))}"
        )
    return output.heading("   ".join(parts))


def _cell(row: dict[str, object], key: str) -> str:
    """The row's value as table text — blank when absent (e.g. a failed row's scores)."""
    return str(row[key]) if key in row else ""


def _pct_cell(row: dict[str, object], key: str) -> str:
    return _pct(row[key]) if key in row else ""


def _secs_cell(row: dict[str, object], key: str) -> str:
    return _secs(row[key]) if key in row else ""


def _final_llm_output(row: dict[str, object]) -> str | None:
    """A row's last ``--llm`` step output, or ``None`` when no chain ran on it."""
    llm_data = jsonshape.as_mapping(row.get("llm"))
    if llm_data is None:
        return None
    steps = jsonshape.mapping_list(llm_data.get("steps"))
    return str(steps[-1].get("output", "") or "") if steps else ""


def _llm_block(payload: dict[str, object]) -> str | None:
    """The per-item ``--llm`` outputs as a heading + one ``item: output`` line each,
    or ``None`` when no ``--llm`` chain ran."""
    lines: list[str] = []
    for row in jsonshape.mapping_list(payload.get("rows")):
        final = _final_llm_output(row)
        if final is not None:
            lines.append(f"{escape(str(row.get('item')))}: {escape(final)}")
    if not lines:
        return None
    return "\n".join([output.heading("--llm"), *lines])


def _reduce_block(payload: dict[str, object]) -> str | None:
    """The ``--llm-reduce`` aggregate as a heading + the output, or ``None`` when unset."""
    reduce = jsonshape.as_mapping(payload.get("reduce"))
    if reduce is None:
        return None
    return f"{output.heading('--llm-reduce')}\n{escape(str(reduce.get('output', '')))}"


def _render(payload: dict[str, object]) -> RenderableType:
    has_wer = "wer" in payload
    has_failed = "failed" in payload
    has_latency = "latency_p50" in payload
    columns = [
        "ITEM",
        *(["WORDS", "ERRORS", "WER"] if has_wer else []),
        *(["LATENCY"] if has_latency else []),
        *(["ERROR"] if has_failed else []),
    ]
    table = output.data_table(*columns)
    for row in jsonshape.mapping_list(payload.get("rows")):
        cells = [str(row.get("item"))]
        if has_wer:
            cells += [_cell(row, "words"), _cell(row, "errors"), _pct_cell(row, "wer")]
        if has_latency:
            cells.append(_secs_cell(row, "latency"))
        if has_failed:
            cells.append(_cell(row, "error"))
        table.add_row(*cells)
    model = payload.get("speech_model") or "default model"
    return output.stack(
        output.muted(f"{payload.get('dataset')} · {model}"),
        table,
        _summary(payload),
        _llm_block(payload),
        _reduce_block(payload),
    )


def _evaluate_one(
    dataset: str, api_key: str, opts: EvalOptions, state: AppState, *, json_mode: bool
) -> dict[str, object]:
    """Score one dataset end to end and return its emitted payload."""
    data = eval_data.load(
        dataset,
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
        _failed_result(item, timed.outcome, timed.latency)
        if isinstance(timed.outcome, CLIError)
        else _score_item(item, timed.outcome, timed.latency)
        for item, timed in zip(
            data.items,
            outcomes,
            strict=True,  # pragma: no mutate (defensive invariant; _transcripts returns one outcome per item)
        )
    ]
    llm_opts = opts.llm_options()
    if llm_opts.prompts:
        _run_llm_map(api_key, results, llm_opts, json_mode=json_mode, quiet=state.quiet)
    payload = _payload(data.label, opts.speech_model, results)
    if llm_opts.reduce_prompts:
        reduce = _run_reduce(api_key, results, llm_opts, json_mode=json_mode, quiet=state.quiet)
        if reduce is not None:
            payload["reduce"] = reduce
    return payload


def run_evaluate(opts: EvalOptions, state: AppState, *, json_mode: bool) -> None:
    """Transcribe one or more evaluation datasets and score WER against references."""
    # Resolve credentials before any dataset download: a signed-out user must
    # not pull the whole dataset only to fail at the first transcription.
    api_key = state.resolve_api_key()
    failed = 0
    total = 0
    for dataset in opts.datasets:
        payload = _evaluate_one(dataset, api_key, opts, state, json_mode=json_mode)
        output.emit(payload, _render, json_mode=json_mode)
        failed += jsonshape.as_int(payload.get("failed"))
        total += jsonshape.as_int(payload.get("items"))
    if failed:
        raise CLIError(
            f"{failed} of {total} items failed to transcribe.",
            error_type="eval_failed",
            suggestion="The summary covers only the items that transcribed.",
        )
