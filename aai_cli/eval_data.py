"""Dataset loading for `assembly eval`: local manifests and Hugging Face datasets.

Two source shapes, one output: a list of (audio, reference-text) items.

* A **local manifest** — a ``.csv`` or ``.jsonl`` file with an audio column
  (path or URL) and a reference-text column. Relative audio paths resolve
  against the manifest's directory.
* A **Hugging Face dataset id** (e.g. ``hf-internal-testing/librispeech_asr_dummy``),
  fetched through the hub's datasets-server REST API — no heavyweight
  ``datasets`` dependency, and the audio arrives as hosted URLs the AssemblyAI
  API ingests directly. Gated/private datasets authenticate via ``HF_TOKEN``.
  The common benchmarks also have short **aliases** (``ALIASES``) that fill in
  the hub id plus the subset/split/audio-column defaults each set needs.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from aai_cli import eval_hf_api, jsonshape, wer
from aai_cli.errors import APIError, CLIError, UsageError

_MANIFEST_SUFFIXES = (".csv", ".jsonl")
# Hub ids are `name` or `namespace/name`; rejecting anything else keeps a typo'd
# local path from being sent to the hub as if it were a dataset id.
_HF_ID_RE = re.compile(r"^[\w.-]+(?:/[\w.-]+)?$")

# Column auto-detection, in preference order: the names the common ASR datasets
# and manifest tools use (HF audio datasets, Common Voice, NeMo manifests).
_AUDIO_COLUMNS = ("audio", "audio_filepath", "audio_url", "path", "file")
_TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript", "normalized_text")


@dataclass(frozen=True)
class Alias:
    """A built-in benchmark alias: the hub dataset id plus the subset/split/
    audio-column defaults its layout needs (explicit flags still win)."""

    dataset: str
    subset: str | None = None
    split: str | None = None
    audio_column: str | None = None


# The benchmarks the `assembly eval` help recommends, under short memorable names
# — each pins the hub id and the fiddly defaults its layout needs, so
# `assembly eval tedlium` just works.
ALIASES: dict[str, Alias] = {
    "librispeech": Alias("openslr/librispeech_asr", subset="clean"),
    "librispeech-other": Alias("openslr/librispeech_asr", subset="other"),
    "tedlium": Alias("sanchit-gandhi/tedlium-data"),
    "earnings22": Alias("sanchit-gandhi/earnings22_robust_split"),
    "spgispeech": Alias("kensho/spgispeech", subset="test"),
    "ami": Alias("edinburghcstr/ami", subset="ihm"),
    "ami-sdm": Alias("edinburghcstr/ami", subset="sdm"),
    "gigaspeech": Alias("fixie-ai/gigaspeech", subset="dev", split="dev"),
    "peoples": Alias("fixie-ai/peoples_speech", subset="clean"),
    "commonvoice": Alias("fixie-ai/common_voice_17_0", subset="en"),
    "voxpopuli": Alias("facebook/voxpopuli", subset="en"),
    "switchboard": Alias("hhoangphuoc/switchboard", split="validation"),
    "expresso": Alias("ylacombe/expresso"),
    "loquacious": Alias("speechbrain/LoquaciousSet", subset="small", audio_column="wav"),
    "callhome": Alias("talkbank/callhome", subset="eng"),
}


@dataclass(frozen=True)
class EvalItem:
    """One evaluation row: an audio source (path or URL) plus its reference text."""

    item_id: str
    audio: str
    reference: str


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
        )
    alias = ALIASES.get(dataset)
    if alias is not None:
        dataset = alias.dataset
        subset = subset or alias.subset
        split = split or alias.split
        audio_column = audio_column or alias.audio_column
    return _load_hf(
        dataset,
        split=split,
        subset=subset,
        audio_column=audio_column,
        text_column=text_column,
        limit=limit,
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
) -> tuple[str, str]:
    """The (audio, text) columns to read."""
    audio_col = _pick_column(columns, audio_column, _AUDIO_COLUMNS, "--audio-column")
    return audio_col, _pick_column(columns, text_column, _TEXT_COLUMNS, "--text-column")


def _checked_reference(item_id: str, reference: str) -> str:
    """Reject a reference that normalizes to no words — WER would be undefined."""
    if not wer.normalize_words(reference):
        raise UsageError(
            f"{item_id} has an empty reference text after normalization.",
            suggestion="Fix the row, or point --text-column at the right column.",
        )
    return reference


def _row_reference(cells: dict[str, object], text_col: str, item_id: str) -> str:
    return _checked_reference(item_id, str(cells.get(text_col) or ""))


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
) -> EvalDataset:
    if not path.is_file():
        raise CLIError(
            f"Manifest not found: {path}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Pass a .csv/.jsonl manifest path, or a Hugging Face dataset id.",
        )
    if path.suffix not in _MANIFEST_SUFFIXES:
        # Other suffixes (.parquet, .txt, …) would be parsed as JSONL and fail with a
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
    )
    items = [
        _manifest_item(path, index, row, audio_col=audio_col, text_col=text_col)
        for index, row in enumerate(rows[:limit], start=1)
    ]
    return EvalDataset(label=path.name, items=items)


def _manifest_item(
    path: Path,
    index: int,
    row: dict[str, object],
    *,
    audio_col: str,
    text_col: str,
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
    )


# ------------------------------------------------------- Hugging Face datasets


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
    text_col: str,
) -> EvalItem:
    cells = jsonshape.as_mapping(row.get("row")) or {}
    item_id = f"{split_name}[{row.get('row_idx')}]"
    return EvalItem(
        item_id=item_id,
        audio=_audio_source(cells.get(audio_col), column=audio_col, item_id=item_id),
        reference=_row_reference(cells, text_col, item_id),
    )


def _load_hf(
    dataset: str,
    *,
    split: str | None,
    subset: str | None,
    audio_column: str | None,
    text_column: str | None,
    limit: int,
) -> EvalDataset:
    if not _HF_ID_RE.match(dataset):
        raise UsageError(
            f"'{dataset}' is neither a local .csv/.jsonl manifest nor a Hugging Face dataset id.",
            suggestion="Pass a manifest path, or an id like 'distil-whisper/meanwhile'.",
        )
    entries = eval_hf_api.split_entries(dataset)
    config = eval_hf_api.pick_subset(entries, subset, dataset)
    split_name = eval_hf_api.pick_split(entries, config, split, dataset)
    payload = eval_hf_api.fetch_json(
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
    )
    return EvalDataset(
        label=f"{dataset} · {config}/{split_name}",
        items=[_hf_item(row, split_name, audio_col=audio_col, text_col=text_col) for row in rows],
    )
