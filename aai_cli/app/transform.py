"""Shared LLM-Gateway map/reduce delivery for `transcribe` and `transcripts get`.

Both commands post-process transcripts through the LLM Gateway the same way: a
per-transcript ``--llm`` *map* chain, then an optional ``--llm-reduce`` chain run
*once* over every result. The map render (the ``transform`` record shape and the
human/NDJSON split) and the reduce policy (concatenate each contribution under a
header, skip the billable call when there's nothing to reduce, emit the additive
``{"type": "reduce", …}`` record) used to be duplicated across
``commands/transcripts.py`` and ``app/transcribe/{run,batch}.py``. Centralized here
so the two callers can't drift on the record shapes, the headers, or the empty-input
guard. The low-level chain primitives stay in ``core.llm``; this is only the
delivery/rendering policy that sits above them.
"""

from __future__ import annotations

from typing import Any

from aai_cli.core import client, llm
from aai_cli.ui import output


def render_transform_steps(d: dict[str, Any]) -> str:
    """Human view of chained LLM-Gateway steps: the lone output, or each step labeled."""
    steps = d["transform"]["steps"]
    if len(steps) == 1:
        return str(steps[0]["output"])
    return "\n\n".join(f"Step {i} — {s['prompt']}:\n{s['output']}" for i, s in enumerate(steps, 1))


def emit_transform(
    transcript: object,
    *,
    model: str,
    steps: list[dict[str, str]],
    json_mode: bool,
    batch: bool = False,
) -> None:
    """Emit a transcript's ``--llm`` chain result.

    One NDJSON ``{"type": "transcript", …}`` record per id under ``--json`` in a batch
    (so a downstream stage can map over the stream), otherwise the same JSON/human render
    `transcribe` gives a single source.
    """
    record = client.transcript_summary(transcript) | {"transform": {"model": model, "steps": steps}}
    if json_mode and batch:
        output.emit_ndjson({"type": "transcript", **record})
    else:
        output.emit(record, render_transform_steps, json_mode=json_mode)


def emit_reduce(
    api_key: str,
    contributions: list[tuple[str, str]],
    *,
    prompts: list[str],
    model: str,
    max_tokens: int,
    block_label: str,
    empty_noun: str,
    json_mode: bool,
) -> None:
    """Run the ``--llm-reduce`` chain once over every contribution; print to stdout.

    ``contributions`` is one ``(id, text)`` per source — each non-empty one becomes a
    ``### {block_label}: {id}`` block. When every contribution is empty the (billable)
    Gateway call is skipped with a ``Nothing to reduce … across {empty_noun}`` warning
    instead of prompting over nothing. Under ``--json`` the result is the additive
    ``{"type": "reduce", …}`` record; otherwise the plain aggregate text.
    """
    combined = "\n\n".join(
        f"### {block_label}: {cid}\n{text}" for cid, text in contributions if text
    )
    if not combined:
        output.emit_warning(
            f"Nothing to reduce: no transcript text across {empty_noun}.", json_mode=json_mode
        )
        return
    result = llm.run_chain(
        api_key, prompts, transcript_text=combined, model=model, max_tokens=max_tokens
    )
    if json_mode:
        output.emit_ndjson({"type": "reduce", "model": model, "prompts": prompts, "output": result})
    else:
        output.emit_text(result)
