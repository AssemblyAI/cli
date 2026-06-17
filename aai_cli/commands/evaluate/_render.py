"""Human-mode rendering for `assembly eval`: turn an emitted payload into a table.

Split out of ``_exec`` so the scoring/orchestration path and the Rich rendering
stay in separate files. Every function here reads only the already-emitted payload
dict (plus its rows), so it shares no state with the run path — the `--json`
output is the payload verbatim, and this module is what `-o`/human mode renders.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.markup import escape

from aai_cli.core import jsonshape
from aai_cli.ui import output


def _pct(value: object) -> str:
    return f"{jsonshape.as_float(value):.2%}"


def _secs(value: object) -> str:
    """A latency in seconds, formatted for display."""
    return f"{jsonshape.as_float(value):.2f}s"


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


def render(payload: dict[str, object]) -> RenderableType:
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
