"""Dataset loading for `assembly eval`: local manifests and Hugging Face datasets.

Two source shapes, one output: a list of (audio, reference-text) items.

* A **local manifest** — a ``.csv`` or ``.jsonl`` file with an audio column
  (path or URL) and a reference-text column. Relative audio paths resolve
  against the manifest's directory.
* A **Hugging Face dataset id** (e.g. ``hf-internal-testing/librispeech_asr_dummy``),
  fetched through the hub's datasets-server REST API — no heavyweight
  ``datasets`` dependency, and the audio arrives as hosted URLs the AssemblyAI
  API ingests directly. Gated/private datasets authenticate via ``HF_TOKEN``.

With ``with_speakers`` (the ``--speaker-labels`` flag), rows must also carry
diarization references as the parallel ``speakers`` / ``timestamps_start`` /
``timestamps_end`` arrays the Hugging Face diarization datasets use (seconds);
reference text then becomes optional, since diarization sets often have none.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

import httpx2 as httpx

from aai_cli import der, jsonshape, wer
from aai_cli.errors import APIError, CLIError, UsageError

_DATASETS_SERVER = "https://datasets-server.huggingface.co"
_TIMEOUT = 30.0  # pragma: no mutate (request timeout; nothing observable to assert)
_MANIFEST_SUFFIXES = (".csv", ".jsonl")
# Hub ids are `name` or `namespace/name`; rejecting anything else keeps a typo'd
# local path from being sent to the hub as if it were a dataset id.
_HF_ID_RE = re.compile(r"^[\w.-]+(?:/[\w.-]+)?$")

# Column auto-detection, in preference order: the names the common ASR datasets
# and manifest tools use (HF audio datasets, Common Voice, NeMo manifests).
_AUDIO_COLUMNS = ("audio", "audio_filepath", "audio_url", "path", "file")
_TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript", "normalized_text")
# Diarization references: the parallel-array convention the Hugging Face
# diarization datasets (diarizers-community/*) use, in seconds.
_SPEAKER_COLUMNS = ("speakers", "timestamps_start", "timestamps_end")


@dataclass(frozen=True)
class EvalItem:
    """One evaluation row: an audio source (path or URL) plus its references —
    text for WER, speaker turns for DER (each optional, never both absent)."""

    item_id: str
    audio: str
    reference: str | None
    turns: list[der.Turn] | None = None


@dataclass(frozen=True)
class EvalDataset:
    """The loaded items plus a human label naming what was resolved (file, or
    ``dataset · subset/split``) for the result header."""

    label: str
    items: list[EvalItem]


def load(
    dataset: str,
    *,
    split: str | None = None,
    subset: str | None = None,
    audio_column: str | None = None,
    text_column: str | None = None,
    limit: int,
    with_speakers: bool = False,
) -> EvalDataset:
    """Load evaluation items from a local manifest or a Hugging Face dataset id."""
    path = Path(dataset)
    if path.suffix in _MANIFEST_SUFFIXES or path.is_file():
        if split is not None or subset is not None:
            raise UsageError(
                "--split/--subset apply to Hugging Face datasets, not local manifests."
            )
        return _load_manifest(
            path,
            audio_column=audio_column,
            text_column=text_column,
            limit=limit,
            with_speakers=with_speakers,
        )
    return _load_hf(
        dataset,
        split=split,
        subset=subset,
        audio_column=audio_column,
        text_column=text_column,
        limit=limit,
        with_speakers=with_speakers,
    )


def _pick_column(
    available: list[str], requested: str | None, candidates: tuple[str, ...], flag: str
) -> str:
    """Resolve a column name: the explicit request, else the first known candidate."""
    if requested is not None:
        if requested in available:
            return requested
        raise UsageError(
            f"The dataset has no '{requested}' column (columns: {', '.join(available)}).",
            suggestion=f"Pass {flag} with one of the existing columns.",
        )
    for candidate in candidates:
        if candidate in available:
            return candidate
    noun = candidates[0]
    article = "an" if noun[0] in "aeiou" else "a"
    raise UsageError(
        f"Could not find {article} {noun} column (columns: {', '.join(available)}).",
        suggestion=f"Name it with {flag}.",
    )


def _resolve_columns(
    columns: list[str],
    *,
    audio_column: str | None,
    text_column: str | None,
    with_speakers: bool,
) -> tuple[str, str | None]:
    """The (audio, text) columns to read. Text is only optional when speaker
    turns are being scored instead — diarization datasets often carry no text."""
    audio_col = _pick_column(columns, audio_column, _AUDIO_COLUMNS, "--audio-column")
    if with_speakers:
        missing = [name for name in _SPEAKER_COLUMNS if name not in columns]
        if missing:
            raise UsageError(
                f"--speaker-labels needs speaker-turn columns; missing: {', '.join(missing)} "
                f"(columns: {', '.join(columns)}).",
                suggestion="Rows carry parallel speakers/timestamps_start/timestamps_end "
                "arrays in seconds (arrays need a .jsonl manifest, not .csv).",
            )
        if text_column is None and not any(name in columns for name in _TEXT_COLUMNS):
            return audio_col, None
    return audio_col, _pick_column(columns, text_column, _TEXT_COLUMNS, "--text-column")


def _checked_reference(item_id: str, reference: str) -> str:
    """Reject a reference that normalizes to no words — WER would be undefined."""
    if not wer.normalize_words(reference):
        raise UsageError(
            f"{item_id} has an empty reference text after normalization.",
            suggestion="Fix the row, or point --text-column at the right column.",
        )
    return reference


def _row_reference(cells: dict[str, object], text_col: str | None, item_id: str) -> str | None:
    if text_col is None:
        return None
    return _checked_reference(item_id, str(cells.get(text_col) or ""))


def _row_turns(cells: dict[str, object], item_id: str) -> list[der.Turn]:
    """The row's reference speaker turns from the parallel-array columns."""
    speakers = jsonshape.object_list(cells.get("speakers"))
    starts = jsonshape.object_list(cells.get("timestamps_start"))
    ends = jsonshape.object_list(cells.get("timestamps_end"))
    if not speakers or len(starts) != len(speakers) or len(ends) != len(speakers):
        raise UsageError(
            f"{item_id} needs non-empty, equal-length speakers/timestamps_start/"
            "timestamps_end arrays.",
            suggestion="Each row lists who spoke plus matching start/end seconds.",
        )
    return [
        der.Turn(
            speaker=str(speakers[i]),
            start=jsonshape.as_float(starts[i]),
            end=jsonshape.as_float(ends[i]),
        )
        for i in range(len(speakers))
    ]


# ---------------------------------------------------------------- local manifests


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{path.name} line {lineno} is not valid JSON: {exc}") from exc
        mapping = jsonshape.as_mapping(data)
        if mapping is None:
            raise UsageError(f"{path.name} line {lineno} is not a JSON object.")
        rows.append(mapping)
    return rows


def _load_manifest(
    path: Path,
    *,
    audio_column: str | None,
    text_column: str | None,
    limit: int,
    with_speakers: bool,
) -> EvalDataset:
    if not path.is_file():
        raise CLIError(
            f"Manifest not found: {path}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Pass a .csv/.jsonl manifest path, or a Hugging Face dataset id.",
        )
    if path.suffix not in _MANIFEST_SUFFIXES:
        # Anything else (.parquet, .txt, …) would be parsed as JSONL and fail with a
        # confusing "line 1 is not valid JSON" — name the real constraint instead.
        raise UsageError(
            f"Manifests must be .csv or .jsonl; got '{path.name}'.",
            suggestion="Convert the manifest, or pass a Hugging Face dataset id.",
        )
    rows = _manifest_rows(path)
    if not rows:
        raise UsageError(f"Manifest {path.name} has no rows.")
    audio_col, text_col = _resolve_columns(
        list(rows[0]),
        audio_column=audio_column,
        text_column=text_column,
        with_speakers=with_speakers,
    )
    items = [
        _manifest_item(
            path, index, row, audio_col=audio_col, text_col=text_col, with_speakers=with_speakers
        )
        for index, row in enumerate(rows[:limit], start=1)
    ]
    return EvalDataset(label=path.name, items=items)


def _manifest_item(
    path: Path,
    index: int,
    row: dict[str, object],
    *,
    audio_col: str,
    text_col: str | None,
    with_speakers: bool,
) -> EvalItem:
    audio = str(row.get(audio_col) or "")
    if not audio:
        raise UsageError(f"{path.name} row {index} has no '{audio_col}' value.")
    if not audio.startswith(("http://", "https://")):
        # pathlib drops the left side when the right side is absolute, so this
        # resolves relative paths against the manifest dir and keeps absolute ones.
        resolved = path.parent / audio
        if not resolved.is_file():
            raise CLIError(
                f"Audio file not found: {resolved}",
                error_type="file_not_found",
                exit_code=2,
                suggestion=f"Manifest audio paths resolve relative to {path.parent}.",
            )
        audio = str(resolved)
    item_id = Path(audio).name
    return EvalItem(
        item_id=item_id,
        audio=audio,
        reference=_row_reference(row, text_col, item_id),
        turns=_row_turns(row, item_id) if with_speakers else None,
    )


# ------------------------------------------------------- Hugging Face datasets


def _error_detail(resp: httpx.Response) -> str:
    try:
        body: object = resp.json()
    except ValueError:
        return resp.text
    mapping = jsonshape.as_mapping(body)
    if mapping is not None and "error" in mapping:
        return str(mapping["error"])
    return resp.text


# A 401/403 body that mentions one of these reads like HF auth/gating, where a token
# can actually help; anything else (e.g. a sandbox proxy's "Host not in allowlist")
# gets the body verbatim instead of a misleading HF_TOKEN hint.
_GATING_HINTS = ("gated", "private", "auth", "token")


def _looks_gating_related(detail: str) -> bool:
    lowered = detail.lower()
    return not detail or any(hint in lowered for hint in _GATING_HINTS)


def _denied_access_error(resp: httpx.Response, *, dataset: str) -> APIError:
    detail = _error_detail(resp)
    message = f"Hugging Face denied access to '{dataset}' (HTTP {resp.status_code})"
    if detail:
        message += f": {detail}"
    return APIError(
        message,
        suggestion=(
            "Gated or private dataset? Set HF_TOKEN to a token that has access."
            if _looks_gating_related(detail)
            else None
        ),
    )


def _checked_payload(resp: httpx.Response, *, dataset: str) -> dict[str, object]:
    if resp.status_code in (401, 403):
        raise _denied_access_error(resp, dataset=dataset)
    if resp.status_code == HTTPStatus.NOT_FOUND:
        raise UsageError(
            f"Hugging Face dataset '{dataset}' was not found: {_error_detail(resp)}",
            suggestion="Check the dataset id, e.g. 'distil-whisper/meanwhile'.",
        )
    if resp.status_code != HTTPStatus.OK:
        raise APIError(
            f"Hugging Face datasets server error (HTTP {resp.status_code}): {_error_detail(resp)}"
        )
    try:
        data: object = resp.json()
    except ValueError as exc:
        raise APIError("Hugging Face datasets server returned invalid JSON.") from exc
    mapping = jsonshape.as_mapping(data)
    if mapping is None:
        raise APIError(
            "Hugging Face datasets server returned unexpected JSON (expected an object)."
        )
    return mapping


def _fetch_json(endpoint: str, params: dict[str, str | int], *, dataset: str) -> dict[str, object]:
    token = os.environ.get("HF_TOKEN")
    headers = {"authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=_DATASETS_SERVER, timeout=_TIMEOUT, headers=headers) as client:
            resp = client.get(endpoint, params=params)
    except httpx.HTTPError as exc:
        raise APIError(f"Could not reach the Hugging Face datasets server: {exc}") from exc
    return _checked_payload(resp, dataset=dataset)


def _split_entries(dataset: str) -> list[dict[str, object]]:
    payload = _fetch_json("/splits", {"dataset": dataset}, dataset=dataset)
    entries = jsonshape.mapping_list(payload.get("splits"))
    if not entries:
        raise APIError(f"Hugging Face reports no splits for '{dataset}'.")
    return entries


def _pick_subset(entries: list[dict[str, object]], subset: str | None, dataset: str) -> str:
    configs = list(dict.fromkeys(str(entry.get("config")) for entry in entries))
    if subset is not None:
        if subset in configs:
            return subset
        raise UsageError(f"'{dataset}' has no subset '{subset}' (subsets: {', '.join(configs)}).")
    if len(configs) == 1:
        return configs[0]
    if "default" in configs:
        return "default"
    raise UsageError(
        f"'{dataset}' has multiple subsets: {', '.join(configs)}.",
        suggestion="Pick one with --subset.",
    )


def _pick_split(
    entries: list[dict[str, object]], config: str, split: str | None, dataset: str
) -> str:
    splits = [str(entry.get("split")) for entry in entries if str(entry.get("config")) == config]
    if split is not None:
        if split in splits:
            return split
        raise UsageError(
            f"'{dataset}' has no '{split}' split in subset '{config}' "
            f"(splits: {', '.join(splits)})."
        )
    if "test" in splits:
        return "test"
    if len(splits) == 1:
        return splits[0]
    raise UsageError(
        f"'{dataset}' has several splits in subset '{config}': {', '.join(splits)}.",
        suggestion="Pick one with --split (eval sets usually score 'test').",
    )


def _audio_source(cell: object, *, column: str, item_id: str) -> str:
    """The audio URL out of a datasets-server cell: a bare string, or the first
    ``src`` of the ``[{"src": …, "type": …}]`` shape audio columns render as."""
    if isinstance(cell, str):
        return cell
    for source in jsonshape.mapping_list(cell):
        src = source.get("src")
        if isinstance(src, str):
            return src
    raise APIError(
        f"{item_id}: column '{column}' carries no audio URL.",
        suggestion="Point --audio-column at the dataset's audio column.",
    )


def _hf_item(
    row: dict[str, object],
    split_name: str,
    *,
    audio_col: str,
    text_col: str | None,
    with_speakers: bool,
) -> EvalItem:
    cells = jsonshape.as_mapping(row.get("row")) or {}
    item_id = f"{split_name}[{row.get('row_idx')}]"
    return EvalItem(
        item_id=item_id,
        audio=_audio_source(cells.get(audio_col), column=audio_col, item_id=item_id),
        reference=_row_reference(cells, text_col, item_id),
        turns=_row_turns(cells, item_id) if with_speakers else None,
    )


def _load_hf(
    dataset: str,
    *,
    split: str | None,
    subset: str | None,
    audio_column: str | None,
    text_column: str | None,
    limit: int,
    with_speakers: bool,
) -> EvalDataset:
    if not _HF_ID_RE.match(dataset):
        raise UsageError(
            f"'{dataset}' is neither a local .csv/.jsonl manifest nor a Hugging Face dataset id.",
            suggestion="Pass a manifest path, or an id like 'distil-whisper/meanwhile'.",
        )
    entries = _split_entries(dataset)
    config = _pick_subset(entries, subset, dataset)
    split_name = _pick_split(entries, config, split, dataset)
    payload = _fetch_json(
        "/rows",
        {"dataset": dataset, "config": config, "split": split_name, "offset": 0, "length": limit},
        dataset=dataset,
    )
    rows = jsonshape.mapping_list(payload.get("rows"))
    if not rows:
        raise APIError(f"'{dataset}' ({config}/{split_name}) returned no rows.")
    audio_col, text_col = _resolve_columns(
        list(jsonshape.as_mapping(rows[0].get("row")) or {}),
        audio_column=audio_column,
        text_column=text_column,
        with_speakers=with_speakers,
    )
    return EvalDataset(
        label=f"{dataset} · {config}/{split_name}",
        items=[
            _hf_item(
                row, split_name, audio_col=audio_col, text_col=text_col, with_speakers=with_speakers
            )
            for row in rows
        ],
    )
