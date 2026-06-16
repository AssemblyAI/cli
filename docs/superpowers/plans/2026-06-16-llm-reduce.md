# `--llm-reduce` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable `--llm-reduce 'PROMPT'` flag to `assembly transcribe` that runs one LLM-Gateway call (a "reduce") over all of a batch's per-source results and prints the aggregate answer to stdout.

**Architecture:** `--llm` is the per-source *map* (existing); `--llm-reduce` is the *reduce*. In batch mode, after every source is transcribed, the reduce step concatenates each source's last `--llm` output (or its transcript text if no `--llm` ran) and runs the reduce prompt chain over the combined text via the existing `llm.run_chain(..., transcript_text=...)` path. When `--llm-reduce` is set, the progress table is routed to stderr so stdout carries only the reduce result. In single-source mode there is nothing to aggregate, so the reduce prompts are appended to the `--llm` chain over the one transcript.

**Tech Stack:** Python 3.12+, Typer, the `assemblyai` SDK, the OpenAI-compatible LLM Gateway, pytest + pytest-mock, syrupy snapshots. All tooling runs via `uv run`.

---

## File structure

- `aai_cli/app/transcribe/run.py` — add `reduce_prompts` to `TransformOptions`, a `chain()` helper, `llm_reduce` to `TranscribeOptions`, wire `transform_options()`, single-source delivery, show-code, and `--out` validation.
- `aai_cli/commands/transcribe.py` — add the `--llm-reduce` Typer option and pass it into `TranscribeOptions`.
- `aai_cli/app/transcribe/batch.py` — add `_reduce_input`, `_gather_reduce_inputs`, `_run_reduce`; route the progress table to stderr when reduce is active; call `_run_reduce` after a successful batch.
- `tests/test_transcribe_reduce.py` — new, self-contained test module for all `--llm-reduce` behavior (no edits to shared test files).
- `REFERENCE.md`, `README.md`, `tests/__snapshots__/` — docs + regenerated `transcribe --help` golden.

Run the targeted tests with `uv run pytest tests/test_transcribe_reduce.py -q` during development; run `./scripts/check.sh` once at the end.

---

### Task 1: Plumb the `--llm-reduce` flag as data

**Files:**
- Modify: `aai_cli/app/transcribe/run.py` (`TransformOptions`, `TranscribeOptions`, `transform_options`)
- Modify: `aai_cli/commands/transcribe.py` (option + construction)
- Test: `tests/test_transcribe_reduce.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe_reduce.py`:

```python
"""`assembly transcribe --llm-reduce`: the batch map-reduce step, single-source
chain behavior, and output routing."""

import json

import pytest
from typer.testing import CliRunner

from aai_cli.app.transcribe import batch as transcribe_batch
from aai_cli.app.transcribe import run as transcribe_run
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()

_TRANSCRIBE = "aai_cli.app.transcribe.run.client.transcribe"
_TRANSFORM = "aai_cli.core.llm.transform_transcript"


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _auth():
    config.set_api_key("default", "sk_live")


def _defaults(**overrides):
    """A minimal TranscribeOptions for seam tests; override only what matters."""
    base = dict(
        source=None, sample=False, from_stdin=False, concurrency=2, force=False,
        speech_model=None, language_code=None, language_detection=None,
        keyterms_prompt=None, temperature=None, prompt=None, punctuate=None,
        format_text=None, disfluencies=None, speaker_labels=False,
        speakers_expected=None, multichannel=None, redact_pii=None,
        redact_pii_policy=None, redact_pii_sub=None, redact_pii_audio=None,
        filter_profanity=None, content_safety=None, content_safety_confidence=None,
        speech_threshold=None, summarization=None, summary_model=None,
        summary_type=None, auto_chapters=None, sentiment_analysis=None,
        entity_detection=None, auto_highlights=None, topic_detection=None,
        word_boost=None, custom_spelling_file=None, audio_start=None,
        audio_end=None, download_sections=None, webhook_url=None,
        webhook_auth_header=None, translate_to=None, config_kv=None,
        config_file=None, llm_prompt=None, model="claude-haiku-4-5-20251001",
        max_tokens=1000, output_field=None, chars_per_caption=None, out=None,
        show_code=False, llm_reduce=None,
    )
    base.update(overrides)
    return transcribe_run.TranscribeOptions(**base)


def test_transform_options_carries_reduce_prompts():
    opts = _defaults(llm_prompt=["judge"], llm_reduce=["rank", "summarize"])
    transform = opts.transform_options()
    assert transform.prompts == ["judge"]
    assert transform.reduce_prompts == ["rank", "summarize"]


def test_chain_appends_reduce_to_map():
    transform = transcribe_run.TransformOptions(
        prompts=["a"], model="m", max_tokens=10, reduce_prompts=["b"]
    )
    assert transform.chain() == ["a", "b"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_transcribe_reduce.py -q`
Expected: FAIL — `TranscribeOptions` has no `llm_reduce`, `TransformOptions` has no `reduce_prompts`/`chain`.

- [ ] **Step 3: Extend `TransformOptions` in `aai_cli/app/transcribe/run.py`**

Replace the `TransformOptions` class (currently lines 97-102) with:

```python
class TransformOptions(NamedTuple):
    """The ``--llm`` chain options: the prompts plus the gateway model settings.

    ``reduce_prompts`` is the ``--llm-reduce`` chain — the aggregate step run over
    all batch results (or appended to the per-transcript chain for a single source).
    """

    prompts: list[str]
    model: str
    max_tokens: int
    reduce_prompts: list[str]

    def chain(self) -> list[str]:
        """The full single-source chain: the map prompts followed by the reduce ones.

        With one source there is nothing to aggregate, so the reduce prompts simply
        extend the ``--llm`` chain over that transcript.
        """
        return self.prompts + self.reduce_prompts
```

- [ ] **Step 4: Update `transform_options()` and add the `TranscribeOptions` field**

In `aai_cli/app/transcribe/run.py`, add the field to the `TranscribeOptions` dataclass right after `llm_prompt` (line 215):

```python
    llm_prompt: list[str] | None
    llm_reduce: list[str] | None
    model: str
    max_tokens: int
```

And update `transform_options()` (lines 272-276) to:

```python
    def transform_options(self) -> TransformOptions:
        """The post-transcription LLM transform spec built from the `--llm` flags."""
        return TransformOptions(
            prompts=list(self.llm_prompt or []),
            model=self.model,
            max_tokens=self.max_tokens,
            reduce_prompts=list(self.llm_reduce or []),
        )
```

- [ ] **Step 5: Check for other `TransformOptions(` constructors**

Run: `grep -rn "TransformOptions(" aai_cli tests`
Expected: only `transform_options()` (just updated) and the test from Step 1 (which already passes `reduce_prompts`). If any other source constructor exists, add `reduce_prompts=[]` to it.

- [ ] **Step 6: Add the Typer option in `aai_cli/commands/transcribe.py`**

Insert immediately after the `llm_prompt` option block (after line 312, before `model`):

```python
    llm_reduce: list[str] | None = typer.Option(
        None,
        "--llm-reduce",
        help="Run one LLM-Gateway prompt over all batch results (a reduce). "
        "Repeatable: each runs on the previous one's output. For a single source it "
        "extends the --llm chain over that transcript.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
```

And pass it into the `TranscribeOptions(...)` construction, right after `llm_prompt=llm_prompt,` (line 412):

```python
        llm_prompt=llm_prompt,
        llm_reduce=llm_reduce,
        model=model,
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transcribe_reduce.py -q`
Expected: PASS (2 tests).

- [ ] **Step 8: Commit**

```bash
git add aai_cli/app/transcribe/run.py aai_cli/commands/transcribe.py tests/test_transcribe_reduce.py
git commit -m "feat(transcribe): plumb --llm-reduce flag as data"
```

---

### Task 2: Single-source reduce (append to the `--llm` chain)

**Files:**
- Modify: `aai_cli/app/transcribe/run.py` (`deliver_result`, `_print_show_code`, `run_transcribe` validation)
- Test: `tests/test_transcribe_reduce.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transcribe_reduce.py`:

```python
def test_single_source_runs_reduce_as_chain_step(mocker):
    _auth()
    mocker.patch(_TRANSCRIBE, return_value=mocker.MagicMock(
        id="t1", text="hello", status="completed",
        json_response={"id": "t1", "text": "hello", "status": "completed"},
    ))
    transform = mocker.patch(_TRANSFORM, side_effect=["mapped", "reduced"])
    result = runner.invoke(app, ["transcribe", "--sample", "--llm", "map", "--llm-reduce", "red"])
    assert result.exit_code == 0, result.output
    # Two chain steps ran: --llm then --llm-reduce, over the one transcript.
    assert transform.call_count == 2
    assert "reduced" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_transcribe_reduce.py::test_single_source_runs_reduce_as_chain_step -q`
Expected: FAIL — only one chain step runs (`--llm-reduce` is ignored in single-source mode).

- [ ] **Step 3: Use the combined chain in `deliver_result`**

In `aai_cli/app/transcribe/run.py`, replace the `if transform.prompts:` branch (lines 140-156) with:

```python
    chain = transform.chain()
    if chain:
        # Chain the prompts: the first runs over the transcript (injected server-side
        # via transcript_id); each subsequent prompt runs over the prior response.
        # --llm-reduce prompts extend the chain here — a single source has nothing to
        # aggregate, so reduce is just more chain steps over this one transcript.
        steps = llm.run_chain_steps(
            api_key,
            chain,
            transcript_id=transcript.id,
            model=transform.model,
            max_tokens=transform.max_tokens,
        )
        output.emit(
            client.transcript_summary(transcript)
            | {"transform": {"model": transform.model, "steps": steps}},
            render_transform_steps,
            json_mode=json_mode,
        )
        return
```

- [ ] **Step 4: Include reduce prompts in `--out` validation and `--show-code`**

In `run_transcribe` (line 320), change the `--out`/`--llm` guard to also cover reduce:

```python
    transcribe_validate.validate_out_with_llm(
        opts.out, (opts.llm_prompt or []) + (opts.llm_reduce or []) or None
    )
```

In `_print_show_code` (line 295), include reduce prompts in the generated gateway chain:

```python
    gateway = code_gen.gateway_options(
        list(opts.llm_prompt or []) + list(opts.llm_reduce or []), opts.model, opts.max_tokens
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_transcribe_reduce.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add aai_cli/app/transcribe/run.py tests/test_transcribe_reduce.py
git commit -m "feat(transcribe): single-source --llm-reduce extends the chain"
```

---

### Task 3: Batch reduce (gather, run, route table to stderr)

**Files:**
- Modify: `aai_cli/app/transcribe/batch.py`
- Test: `tests/test_transcribe_reduce.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transcribe_reduce.py`:

```python
def _fake_transcript(mocker, source):
    t = mocker.MagicMock()
    t.id = f"t_{source}"
    t.text = f"text of {source}"
    t.status = "completed"
    t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
    return t


def _ndjson(result):
    return [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]


def test_batch_reduce_feeds_map_outputs(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(_TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio))
    # map step returns one output per source; reduce returns the final answer.
    mocker.patch(_TRANSFORM, side_effect=["JUDGED a", "JUDGED b", "FINAL"])
    captured = {}

    real_run_chain = transcribe_batch.llm.run_chain

    def spy(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        captured["prompts"] = prompts
        return "FINAL"

    monkeypatch.setattr(transcribe_batch.llm, "run_chain", spy)
    result = runner.invoke(
        app,
        ["transcribe", "--from-stdin", "--llm", "judge", "--llm-reduce", "rank"],
        input="https://a\nhttps://b\n",
    )
    assert result.exit_code == 0, result.output
    # Reduce saw both sources' map outputs, each under a source header.
    assert "### Source: https://a" in captured["text"]
    assert "JUDGED a" in captured["text"] and "JUDGED b" in captured["text"]
    assert captured["prompts"] == ["rank"]
    assert "FINAL" in result.output


def test_batch_reduce_falls_back_to_transcript_text(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(_TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio))
    captured = {}

    def spy(api_key, prompts, *, transcript_text, model, max_tokens):
        captured["text"] = transcript_text
        return "FINAL"

    monkeypatch.setattr(transcribe_batch.llm, "run_chain", spy)
    result = runner.invoke(
        app,
        ["transcribe", "--from-stdin", "--llm-reduce", "summarize"],
        input="https://a\n",
    )
    assert result.exit_code == 0, result.output
    # No --llm map ran, so the transcript text is fed to the reduce.
    assert "text of https://a" in captured["text"]


def test_batch_reduce_emits_json_record(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(_TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio))
    monkeypatch.setattr(
        transcribe_batch.llm, "run_chain",
        lambda api_key, prompts, *, transcript_text, model, max_tokens: "FINAL",
    )
    result = runner.invoke(
        app,
        ["transcribe", "--from-stdin", "--llm-reduce", "summarize", "--json"],
        input="https://a\n",
    )
    assert result.exit_code == 0, result.output
    records = _ndjson(result)
    reduce_records = [r for r in records if r.get("type") == "reduce"]
    assert len(reduce_records) == 1
    assert reduce_records[0]["output"] == "FINAL"
    assert reduce_records[0]["prompts"] == ["summarize"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_transcribe_reduce.py -q`
Expected: FAIL — `--llm-reduce` produces no reduce call / record in batch mode.

- [ ] **Step 3: Add the reduce helpers in `aai_cli/app/transcribe/batch.py`**

Add these functions above `run_batch` (after `_summarize`, line 296):

```python
def _reduce_input(record: dict[str, object]) -> str:
    """A source's contribution to the reduce: its last --llm output, else its text."""
    transform = jsonshape.as_mapping(record.get("transform"))
    if transform is not None:
        steps = transform.get("steps")
        if isinstance(steps, list) and steps:
            last = jsonshape.as_mapping(steps[-1])
            if last is not None:
                return str(last.get("output", "") or "")
    transcript = jsonshape.as_mapping(record.get("transcript"))
    if transcript is not None:
        return str(transcript.get("text", "") or "")
    return ""


def _gather_reduce_inputs(items: list[_Item]) -> str:
    """Concatenate each completed/skipped source's reduce input under a header."""
    blocks: list[str] = []
    for item in items:
        if item.status not in ("completed", "skipped"):
            continue
        record = resumable_record(sidecar_path(item.source), digest=None)
        text = _reduce_input(record) if record is not None else ""
        if text:
            blocks.append(f"### Source: {item.source}\n{text}")
    return "\n\n".join(blocks)


def _run_reduce(
    api_key: str,
    items: list[_Item],
    *,
    transform: transcribe_exec.TransformOptions,
    json_mode: bool,
) -> None:
    """Run the --llm-reduce chain once over every source's result; print to stdout."""
    combined = _gather_reduce_inputs(items)
    result = llm.run_chain(
        api_key,
        transform.reduce_prompts,
        transcript_text=combined,
        model=transform.model,
        max_tokens=transform.max_tokens,
    )
    if json_mode:
        # Additive NDJSON event after the per-source {"type":"result"} records.
        output.emit_ndjson(
            {
                "type": "reduce",
                "model": transform.model,
                "prompts": transform.reduce_prompts,
                "output": result,
            }
        )
    else:
        output.emit_text(result)
```

- [ ] **Step 4: Route the table to stderr and call the reduce in `run_batch`**

Change `_progress_table` (line 230) to accept a `reduce_active` flag and pick the console:

```python
@contextmanager
def _progress_table(items: list[_Item], *, json_mode: bool, reduce_active: bool = False) -> Generator[None]:
    """Render the batch as a live-updating table (human mode).

    Rich renders nothing while running on a non-interactive console and prints the
    final frame once on stop, so piped/agent runs still get the result table. JSON
    mode skips Rich entirely — NDJSON per source is the output. When a --llm-reduce
    step will print the aggregate to stdout, the table goes to stderr so stdout
    carries only the reduce result.
    """
    if json_mode:
        yield
        return
    console = output.error_console if reduce_active else output.console
    with Live(
        get_renderable=lambda: _render_table(items),
        console=console,
        refresh_per_second=4,  # pragma: no mutate (cosmetic refresh cadence)
    ):
        yield
```

Update `run_batch` (lines 314-325) to:

```python
    items = [_Item(source) for source in sources]
    reduce_active = bool(transform.reduce_prompts)
    with _progress_table(items, json_mode=json_mode, reduce_active=reduce_active):
        _drain(
            api_key,
            items,
            transcription_config=transcription_config,
            concurrency=concurrency,
            force=force,
            transform=transform,
            json_mode=json_mode,
        )
    _summarize(items, json_mode=json_mode, quiet=quiet)
    if reduce_active:
        _run_reduce(api_key, items, transform=transform, json_mode=json_mode)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transcribe_reduce.py -q`
Expected: PASS (6 tests).

- [ ] **Step 6: Add a routing test (table on stderr, result on stdout)**

Append to `tests/test_transcribe_reduce.py`:

```python
def test_batch_reduce_routes_table_to_stderr(mocker, monkeypatch, capsys):
    """run_batch sends the reduce result to stdout and the progress table to stderr."""
    import assemblyai as aai

    _auth()
    monkeypatch.setattr(_TRANSCRIBE, lambda api_key, audio, *, config: _fake_transcript(mocker, audio))
    monkeypatch.setattr(
        transcribe_batch.llm, "run_chain",
        lambda api_key, prompts, *, transcript_text, model, max_tokens: "AGGREGATE",
    )
    transform = transcribe_run.TransformOptions(
        prompts=[], model="m", max_tokens=10, reduce_prompts=["summarize"]
    )
    transcribe_batch.run_batch(
        "sk_live",
        ["https://a"],
        transcription_config=aai.TranscriptionConfig(),
        concurrency=1,
        force=False,
        transform=transform,
        json_mode=False,
        quiet=False,
    )
    out, err = capsys.readouterr()
    assert "AGGREGATE" in out          # reduce result → stdout
    assert "AGGREGATE" not in err
    assert "https://a" in err          # progress table → stderr
```

- [ ] **Step 7: Run it to verify it passes**

Run: `uv run pytest tests/test_transcribe_reduce.py::test_batch_reduce_routes_table_to_stderr -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aai_cli/app/transcribe/batch.py tests/test_transcribe_reduce.py
git commit -m "feat(transcribe): batch --llm-reduce aggregates results to stdout"
```

---

### Task 4: Docs + help snapshot

**Files:**
- Modify: `REFERENCE.md`, `README.md`
- Modify: `tests/__snapshots__/` (regenerated, not hand-edited)

- [ ] **Step 1: Locate the transcribe flag + NDJSON sections in REFERENCE.md**

Run: `grep -n "\-\-llm\b\|\"type\"\|result.*sidecar\|## .*transcribe" REFERENCE.md`
Read the surrounding lines to match the existing format.

- [ ] **Step 2: Document `--llm-reduce` and the `reduce` event in REFERENCE.md**

Add a `--llm-reduce` entry beside the existing `--llm` documentation, worded: "Run one LLM-Gateway prompt over all batch results (a reduce); repeatable. For a single source it extends the `--llm` chain." Where the NDJSON `type` values are listed, add `reduce` — `{"type":"reduce","model","prompts","output"}`, emitted once after the per-source `result` records when `--llm-reduce` is set.

- [ ] **Step 3: Update the README example**

In `README.md`, find the "Score diarization quality across several videos" example (added earlier) and extend it to show the reduce step, replacing the trailing `--llm` line so the block ends with:

```sh
| assembly transcribe --from-stdin --concurrency 3 --speaker-labels \
    --llm 'Judge diarization quality; output JSON {speaker_count, issues, score}' \
    --llm-reduce 'Rank these videos worst-to-best and summarize the failure modes'
```

Update the prose to mention that `--llm-reduce` runs one prompt over all results.

- [ ] **Step 4: Regenerate the `transcribe --help` snapshot**

Run: `uv run pytest tests/ -k "snapshot and transcribe" --snapshot-update -q`
Then inspect the diff: `git diff tests/__snapshots__/` — it must show only the new `--llm-reduce` help line. Never hand-edit `.ambr` files.

- [ ] **Step 5: Verify docs + snapshots**

Run: `uv run pytest tests/ -k "snapshot and transcribe" -q && uv run python scripts/docs_consistency_gate.py`
Expected: PASS / no consistency errors. (If `scripts/docs_consistency_gate.py` fails to start under the sandbox with an EPERM from safe-chain, re-run the single command with the sandbox disabled.)

- [ ] **Step 6: Commit**

```bash
git add REFERENCE.md README.md tests/__snapshots__
git commit -m "docs(transcribe): document --llm-reduce and the reduce NDJSON event"
```

---

### Task 5: Full gate

- [ ] **Step 1: Run the authoritative gate**

Run: `./scripts/check.sh`
Expected: it prints `All checks passed.` This enforces lint, types, vulture/deptry/import-linter, **100% patch coverage vs origin/main**, the **diff-scoped mutation gate**, the "no new escape hatches" gate, CodeQL, and the build. Do not claim done until it prints that line.

- [ ] **Step 2: Fix and re-run as needed**

If patch coverage flags an uncovered line, add an assertion that would fail if that line broke (not just a call). If a mutant survives on a changed line, add the assertion that kills it. Re-run `./scripts/check.sh` after any edit (the commit-gate hook requires a passing run for the current tree).

- [ ] **Step 3: Final commit (if the gate produced fixups)**

```bash
git add -A
git commit -m "test(transcribe): close coverage/mutation gaps for --llm-reduce"
```

---

## Self-review

- **Spec coverage:** flag + repeatable chain (Task 1); reduce-input = last map output else transcript text (Task 3 `_reduce_input`); concatenation with `### Source:` headers (Task 3 `_gather_reduce_inputs`); reduce via `llm.run_chain` inline text (Task 3 `_run_reduce`); table→stderr routing gated on the flag (Task 3 `_progress_table`); additive `{type:"reduce"}` NDJSON (Task 3); single-source = append to chain (Task 2); docs + snapshot (Task 4); gates (Task 5). All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; the only "find the spot" steps (Task 4 REFERENCE.md) give exact grep commands because the file's layout isn't quoted here.
- **Type consistency:** `reduce_prompts: list[str]` and `chain()` defined in Task 1 are the names used in Tasks 2-3; `_run_reduce`/`_gather_reduce_inputs`/`_reduce_input` signatures match their call sites; `run_chain(..., transcript_text=...)` matches `aai_cli/core/llm.py`.
