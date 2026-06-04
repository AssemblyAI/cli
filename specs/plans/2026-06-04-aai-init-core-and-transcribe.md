# `aai init` Core Machinery + `transcribe` Template — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `aai init` command that interactively picks a template, copies a real, Vercel-deployable Python web project, installs its dependencies, launches a local server, and opens the browser — implemented end-to-end for the `transcribe` template.

**Architecture:** A new `aai_cli/commands/init.py` Typer command drives three small, independently testable units under a new `aai_cli/init/` package: a template registry, a `scaffold` step (copy committed template files + write `.env`), and a `runner` step (detect `uv`/`venv`, install deps, pick a free port, launch `uvicorn`, open the browser). Templates are committed real projects under `aai_cli/init/templates/<id>/` and copied verbatim (Approach A from the spec) — the generated output *is* the files we hand-wrote and test. Progress is reported with the same "steps with status" pattern as `aai claude`.

**Tech Stack:** Python 3.10+, Typer/Click (vendored in typer), `questionary` (new dep, arrow-key picker), `importlib.resources` (locate packaged template files), `uv`/`venv`+`pip` (env setup for the generated app), FastAPI + uvicorn + python-dotenv (dependencies of the *generated* app only), pytest + Typer `CliRunner` + FastAPI `TestClient` (tests).

**Spec:** `specs/2026-06-04-aai-init-design.md`

---

## Scope & follow-on plans

This plan delivers the **full `aai init` machinery** plus **one template (`transcribe`)** as a complete, working vertical slice. The remaining three templates from the spec are **separate sibling plans**, each additive and of the same shape (add `aai_cli/init/templates/<id>/`, register the id, add an integrity test):

- `2026-06-XX-aai-init-stream-template.md` — live captions; backend `/api/token` only, browser holds the AssemblyAI Streaming WS directly.
- `2026-06-XX-aai-init-agent-template.md` — voice agent; same token-only backend shape as `stream` (`GET /v1/token` → `wss://agents.assemblyai.com/v1/ws`).
- `2026-06-XX-aai-init-llm-template.md` — chat with audio; transcribe create+poll, then `/api/chat` over the transcript via LLM Gateway.

Splitting this way keeps each plan focused and lets the template plans build on the *proven* machinery from this one. The picker and registry are written to make adding a template a one-line registration plus a template directory.

## File structure

**New files (CLI package):**
- `aai_cli/init/__init__.py` — package marker; re-exports the public surface (`TEMPLATES`, `TEMPLATE_ORDER`, `scaffold`, `resolve_optional_api_key`).
- `aai_cli/init/templates.py` — the template registry: `TEMPLATES` (id → human title), `TEMPLATE_ORDER` (display order), `title_for`, `is_template`.
- `aai_cli/init/keys.py` — `resolve_optional_api_key(profile)`: the CLI's key chain (env → keyring) but returning `None` instead of raising when absent.
- `aai_cli/init/scaffold.py` — `scaffold(template, target_dir, api_key)`: copy the packaged template tree (renaming dotfile templates) and write `.env`; plus `target_conflict(dir)` and `template_files(template)` helpers.
- `aai_cli/init/runner.py` — pure command builders + IO helpers: `has_uv`, `venv_python`, `env_setup_commands`, `serve_command`, `find_free_port`, `wait_for_port`, `launch_and_open`.
- `aai_cli/init/steps.py` — `Step` TypedDict + `render_steps(steps)` (same shape as `aai_cli/commands/claude.py`, kept local to `init`).
- `aai_cli/commands/init.py` — the Typer command wiring picker → scaffold → runner → step output.

**New files (the `transcribe` template, committed real project):**
- `aai_cli/init/templates/transcribe/api/index.py` — FastAPI app: serve `index.html`, `POST /api/transcribe` (submit), `GET /api/status/{id}` (poll).
- `aai_cli/init/templates/transcribe/index.html` — one vanilla-JS page: file picker → submit → poll → render transcript + audio-intelligence chips.
- `aai_cli/init/templates/transcribe/vercel.json` — route `/` to the page and `/api/*` to the function.
- `aai_cli/init/templates/transcribe/requirements.txt` — `fastapi`, `assemblyai`, `python-dotenv`, `uvicorn`, `python-multipart`.
- `aai_cli/init/templates/transcribe/README.md` — what it is, run locally, deploy to Vercel, "ideas to extend".
- `aai_cli/init/templates/transcribe/gitignore` — packaged WITHOUT a leading dot; scaffold renames to `.gitignore`.
- `aai_cli/init/templates/transcribe/env.example` — packaged without a leading dot; scaffold renames to `.env.example`.

**Modified files:**
- `pyproject.toml` — add `questionary` to `dependencies`; ensure the template tree (incl. non-`.py` files) is included in the wheel.
- `aai_cli/main.py` — import and register the `init` command; add `"init"` to `_COMMAND_ORDER`.
- `tests/` — new test modules per task.

**Design note on dotfiles:** template dotfiles are stored under plain names (`gitignore`, `env.example`) so (a) the repo's own `.gitignore` can't accidentally ignore them, and (b) they're reliably included in the wheel. `scaffold` maps them to their dotted names on copy. The real `.env` is never committed in a template — it's written at scaffold time from the resolved key (or a placeholder).

---

## Task 1: New dependency + package skeleton + packaging

**Files:**
- Modify: `pyproject.toml` (dependencies + wheel package data)
- Create: `aai_cli/init/__init__.py`
- Test: `tests/test_init_packaging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_packaging.py
import importlib


def test_init_package_imports():
    mod = importlib.import_module("aai_cli.init")
    assert mod is not None


def test_questionary_is_available():
    # questionary is a runtime dependency of the CLI (the init picker).
    assert importlib.import_module("questionary") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_packaging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.init'` (and possibly `questionary`).

- [ ] **Step 3: Add the dependency and create the package**

In `pyproject.toml`, add to the `dependencies` array (keep the existing list; add this line):

```toml
    "questionary>=2.0.1",
```

Then ensure the template tree ships in the wheel. The wheel target currently is:

```toml
[tool.hatch.build.targets.wheel]
packages = ["aai_cli"]
```

Add an explicit artifacts include so non-`.py` template files (and renamed dotfiles) are always packaged:

```toml
[tool.hatch.build.targets.wheel]
packages = ["aai_cli"]
artifacts = ["aai_cli/init/templates/**"]
```

Create the package marker:

```python
# aai_cli/init/__init__.py
from __future__ import annotations
```

- [ ] **Step 4: Sync the environment and run the test**

Run: `uv sync --extra dev && uv run pytest tests/test_init_packaging.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock aai_cli/init/__init__.py tests/test_init_packaging.py
git commit -m "feat(init): add questionary dep + init package skeleton"
```

---

## Task 2: Template registry

**Files:**
- Create: `aai_cli/init/templates.py`
- Test: `tests/test_init_templates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_templates.py
from aai_cli.init import templates


def test_all_four_template_ids_present():
    assert set(templates.TEMPLATES) == {"transcribe", "stream", "agent", "llm"}


def test_template_order_is_complete_and_stable():
    # Display order mirrors the CLI's command order; every id appears exactly once.
    assert templates.TEMPLATE_ORDER == ("transcribe", "stream", "agent", "llm")
    assert set(templates.TEMPLATE_ORDER) == set(templates.TEMPLATES)


def test_title_for_known_and_unknown():
    assert "Transcribe" in templates.title_for("transcribe")
    assert templates.title_for("nope") == "nope"  # falls back to the raw id


def test_is_template():
    assert templates.is_template("agent") is True
    assert templates.is_template("nope") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.init.templates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aai_cli/init/templates.py
from __future__ import annotations

# id -> human-facing title shown in the picker. Ids mirror the CLI's commands.
TEMPLATES: dict[str, str] = {
    "transcribe": "Transcribe & explore a file",
    "stream": "Live captions (mic → browser)",
    "agent": "Talk to a voice agent",
    "llm": "Chat with your audio (LLM)",
}

# Display order for the picker and `--help`, matching the CLI's command order.
TEMPLATE_ORDER: tuple[str, ...] = ("transcribe", "stream", "agent", "llm")


def is_template(name: str) -> bool:
    return name in TEMPLATES


def title_for(name: str) -> str:
    """The human title for a template id, or the raw id if unknown."""
    return TEMPLATES.get(name, name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_init_templates.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/templates.py tests/test_init_templates.py
git commit -m "feat(init): template registry"
```

---

## Task 3: Optional API-key resolution

**Files:**
- Create: `aai_cli/init/keys.py`
- Test: `tests/test_init_keys.py`

The CLI's `config.resolve_api_key` raises `NotAuthenticated` when no key exists. `aai init` must still scaffold without a key (writing a placeholder), so it needs a non-raising variant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_keys.py
from aai_cli import config
from aai_cli.init import keys


def test_resolves_from_env(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "env-key-123")
    assert keys.resolve_optional_api_key(profile=None) == "env-key-123"


def test_resolves_from_keyring(memory_keyring):
    config.set_api_key("default", "stored-key-456")
    assert keys.resolve_optional_api_key(profile=None) == "stored-key-456"


def test_returns_none_when_absent():
    # isolate_env strips the env var and memory_keyring starts empty.
    assert keys.resolve_optional_api_key(profile=None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_keys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.init.keys'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aai_cli/init/keys.py
from __future__ import annotations

from aai_cli import config
from aai_cli.errors import NotAuthenticated


def resolve_optional_api_key(*, profile: str | None) -> str | None:
    """The CLI's key chain (env -> keyring), but None instead of raising when absent.

    `aai init` scaffolds even without a key (writing a placeholder), so it must not
    fail the way run commands do.
    """
    try:
        return config.resolve_api_key(profile=profile)
    except NotAuthenticated:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_init_keys.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/keys.py tests/test_init_keys.py
git commit -m "feat(init): optional API-key resolution (no raise when absent)"
```

---

## Task 4: The `transcribe` template files

**Files:**
- Create: `aai_cli/init/templates/transcribe/api/index.py`
- Create: `aai_cli/init/templates/transcribe/index.html`
- Create: `aai_cli/init/templates/transcribe/vercel.json`
- Create: `aai_cli/init/templates/transcribe/requirements.txt`
- Create: `aai_cli/init/templates/transcribe/README.md`
- Create: `aai_cli/init/templates/transcribe/gitignore`
- Create: `aai_cli/init/templates/transcribe/env.example`
- Test: `tests/test_init_template_transcribe.py`

This task writes the real, runnable template and a test that loads its FastAPI app via `TestClient` with the AssemblyAI SDK mocked. The template uses a **submit + poll** split (each backend call is short → serverless-friendly), per the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_template_transcribe.py
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/transcribe")


def _load_app(monkeypatch):
    """Import the template's api/index.py as a module and return its FastAPI app.

    The template is a standalone project (not part of aai_cli's import graph), so we
    load it by file path. assemblyai is stubbed so no network/key is needed.
    """
    fake_aai = MagicMock()
    fake_aai.TranscriptStatus.completed = "completed"
    fake_aai.TranscriptStatus.error = "error"
    submitted = MagicMock(id="t-123")
    fake_aai.Transcriber.return_value.submit.return_value = submitted
    done = MagicMock(status="completed", error=None,
                     json_response={"text": "hello world", "utterances": []})
    fake_aai.Transcript.get_by_id.return_value = done
    monkeypatch.setitem(sys.modules, "assemblyai", fake_aai)

    spec = importlib.util.spec_from_file_location(
        "_tmpl_transcribe", TEMPLATE_DIR / "api" / "index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app, fake_aai


def test_required_files_exist():
    for rel in ("api/index.py", "index.html", "vercel.json",
                "requirements.txt", "README.md", "gitignore", "env.example"):
        assert (TEMPLATE_DIR / rel).exists(), rel


def test_template_ships_no_real_key():
    # No committed file may contain a literal .env with a key; only env.example.
    assert not (TEMPLATE_DIR / ".env").exists()
    assert "your_assemblyai_api_key_here" in (TEMPLATE_DIR / "env.example").read_text()


def test_index_route_serves_page(monkeypatch):
    app, _ = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_submit_returns_transcript_id(monkeypatch):
    app, fake = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/transcribe", files={"file": ("a.mp3", b"\x00\x01", "audio/mpeg")})
    assert resp.status_code == 200
    assert resp.json() == {"id": "t-123"}
    fake.Transcriber.return_value.submit.assert_called_once()


def test_status_returns_completed_transcript(monkeypatch):
    app, _ = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.get("/api/status/t-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["transcript"]["text"] == "hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_template_transcribe.py -v`
Expected: FAIL — the template files don't exist yet (`test_required_files_exist` fails; loaders error on missing path).

- [ ] **Step 3: Create the backend `api/index.py`**

```python
# aai_cli/init/templates/transcribe/api/index.py
"""Transcribe & explore — AssemblyAI starter (FastAPI).

Two short endpoints so each request stays fast (serverless-friendly):
  POST /api/transcribe  -> submit the upload, return {"id": ...}
  GET  /api/status/{id} -> poll; returns the full transcript JSON when complete

The browser (index.html) submits a file, then polls status until done.
Your API key stays on the server — the browser never sees it.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import assemblyai as aai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

load_dotenv()
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")

# Audio-intelligence features to showcase. Tweak these — that's the point of a starter.
CONFIG = aai.TranscriptionConfig(
    speaker_labels=True,
    auto_chapters=True,
    sentiment_analysis=True,
    entity_detection=True,
    auto_highlights=True,
)

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.post("/api/transcribe")
async def transcribe(file: UploadFile) -> dict[str, str]:
    suffix = Path(file.filename or "audio").suffix
    tmp = Path(tempfile.gettempdir()) / f"aai-{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await file.read())
    transcript = aai.Transcriber().submit(str(tmp), config=CONFIG)
    return {"id": transcript.id}


@app.get("/api/status/{transcript_id}")
def status(transcript_id: str) -> dict[str, object]:
    t = aai.Transcript.get_by_id(transcript_id)
    if t.status == aai.TranscriptStatus.error:
        raise HTTPException(status_code=502, detail=t.error or "Transcription failed")
    if t.status == aai.TranscriptStatus.completed:
        return {"status": "completed", "transcript": t.json_response}
    return {"status": str(getattr(t.status, "value", t.status))}
```

- [ ] **Step 4: Create the frontend `index.html`**

```html
<!-- aai_cli/init/templates/transcribe/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transcribe &amp; explore · AssemblyAI</title>
  <style>
    body { font: 15px/1.6 system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
    h1 { font-size: 20px; }
    #drop { border: 2px dashed #bbb; border-radius: 10px; padding: 28px; text-align: center; cursor: pointer; }
    #drop.busy { opacity: .6; pointer-events: none; }
    .chips span { display: inline-block; background: #eef; border-radius: 99px; padding: 3px 10px; margin: 2px; font-size: 12px; }
    .turn { margin: 6px 0; }
    .spk { font-weight: 600; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>🎧 Transcribe &amp; explore</h1>
  <p>Pick an audio or video file. The server transcribes it with AssemblyAI and returns speakers, chapters, sentiment, entities, and highlights.</p>

  <label id="drop">
    <input id="file" type="file" accept="audio/*,video/*" hidden />
    <span id="droplabel">Click to choose a file</span>
  </label>

  <p id="status"></p>
  <div id="result"></div>

  <script>
    const fileInput = document.getElementById("file");
    const drop = document.getElementById("drop");
    const statusEl = document.getElementById("status");
    const result = document.getElementById("result");

    drop.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => fileInput.files[0] && upload(fileInput.files[0]));

    async function upload(file) {
      drop.classList.add("busy");
      result.innerHTML = "";
      statusEl.textContent = "Uploading…";
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/transcribe", { method: "POST", body });
      if (!res.ok) { statusEl.textContent = "Error: " + (await res.text()); drop.classList.remove("busy"); return; }
      const { id } = await res.json();
      poll(id);
    }

    async function poll(id) {
      statusEl.textContent = "Transcribing…";
      const res = await fetch("/api/status/" + id);
      if (!res.ok) { statusEl.textContent = "Error: " + (await res.text()); drop.classList.remove("busy"); return; }
      const data = await res.json();
      if (data.status !== "completed") { setTimeout(() => poll(id), 2000); return; }
      statusEl.textContent = "Done.";
      drop.classList.remove("busy");
      render(data.transcript);
    }

    function render(t) {
      const chips = [];
      if (t.chapters) chips.push("Chapters");
      if (t.sentiment_analysis_results) chips.push("Sentiment");
      if (t.entities) chips.push("Entities");
      if (t.auto_highlights_result) chips.push("Highlights");
      const utterances = (t.utterances || []).map(
        (u) => `<div class="turn"><span class="spk">Speaker ${u.speaker}:</span> ${escapeHtml(u.text)}</div>`
      ).join("");
      result.innerHTML =
        `<div class="chips">${chips.map((c) => `<span>${c}</span>`).join("")}</div>` +
        (utterances || `<pre>${escapeHtml(t.text || "")}</pre>`);
    }

    function escapeHtml(s) {
      return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    }
  </script>
</body>
</html>
```

- [ ] **Step 5: Create `vercel.json`, `requirements.txt`, `README.md`, `gitignore`, `env.example`**

```json
// aai_cli/init/templates/transcribe/vercel.json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

```text
# aai_cli/init/templates/transcribe/requirements.txt
fastapi
uvicorn
assemblyai
python-dotenv
python-multipart
```

```markdown
<!-- aai_cli/init/templates/transcribe/README.md -->
# Transcribe & explore — AssemblyAI starter

Upload an audio/video file and see the transcript with speaker labels, chapters,
sentiment, entities, and highlights. Built with FastAPI + a single HTML page.

## Run locally

```sh
uvicorn api.index:app --reload --port 3000
# open http://localhost:3000
```

`ASSEMBLYAI_API_KEY` is read from `.env` (already created for you if you ran `aai init`).

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY`
as a Vercel environment variable (the local `.env` is git-ignored and not deployed).
No extra config — `vercel.json` routes the page and the `/api` function.

## Ideas to extend

- Show chapter summaries and highlight timestamps.
- Add a waveform / audio player synced to the transcript.
- Swap the analysis features in `CONFIG` (api/index.py).
```

```text
# aai_cli/init/templates/transcribe/gitignore
.env
.venv
__pycache__/
```

```text
# aai_cli/init/templates/transcribe/env.example
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

- [ ] **Step 6: Add `python-multipart` to the dev env so the test client can post files**

`fastapi`/`uvicorn`/`python-multipart` are runtime deps of the *generated* app, not the CLI. The test loads the template app in-process, so they must be available in the dev environment. Add them to the `dev` optional-dependencies in `pyproject.toml`:

```toml
dev = [
    "pytest>=9.0.3",
    "pytest-cov>=7.1.0",
    "hypothesis>=6.155.1",
    "ruff>=0.15.15",
    "mypy>=2.1.0",
    "pre-commit>=4.6.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "python-dotenv>=1.0.0",
    "python-multipart>=0.0.9",
]
```

Then: `uv sync --extra dev`

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_init_template_transcribe.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add aai_cli/init/templates/transcribe pyproject.toml uv.lock tests/test_init_template_transcribe.py
git commit -m "feat(init): transcribe template (FastAPI submit+poll, Vercel-ready)"
```

---

## Task 5: Scaffold (copy template + write `.env`)

**Files:**
- Create: `aai_cli/init/scaffold.py`
- Test: `tests/test_init_scaffold.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_scaffold.py
from pathlib import Path

import pytest

from aai_cli.errors import CLIError
from aai_cli.init import scaffold


def test_scaffold_copies_files_and_renames_dotfiles(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("transcribe", target, api_key="sk-real-key")
    assert (target / "api" / "index.py").exists()
    assert (target / "index.html").exists()
    assert (target / "vercel.json").exists()
    # dotfile templates are renamed to their dotted names
    assert (target / ".gitignore").exists()
    assert (target / ".env.example").exists()
    # the plain-named source files are NOT copied verbatim
    assert not (target / "gitignore").exists()
    assert not (target / "env.example").exists()


def test_scaffold_writes_env_with_key(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("transcribe", target, api_key="sk-real-key")
    env = (target / ".env").read_text()
    assert "ASSEMBLYAI_API_KEY=sk-real-key" in env


def test_scaffold_writes_placeholder_when_no_key(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("transcribe", target, api_key=None)
    env = (target / ".env").read_text()
    assert scaffold.PLACEHOLDER_KEY in env


def test_scaffold_unknown_template_raises(tmp_path):
    with pytest.raises(CLIError):
        scaffold.scaffold("nope", tmp_path / "app", api_key=None)


def test_target_conflict_detects_nonempty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scaffold.target_conflict(empty) is False
    assert scaffold.target_conflict(tmp_path / "missing") is False
    nonempty = tmp_path / "full"
    nonempty.mkdir()
    (nonempty / "x.txt").write_text("hi")
    assert scaffold.target_conflict(nonempty) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_scaffold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.init.scaffold'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aai_cli/init/scaffold.py
from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from aai_cli.errors import CLIError
from aai_cli.init import templates

PLACEHOLDER_KEY = "your_assemblyai_api_key_here"

# Template files stored under plain names -> their real dotted names on copy.
_DOTFILE_RENAMES = {"gitignore": ".gitignore", "env.example": ".env.example"}


def _template_root(template: str) -> Traversable:
    if not templates.is_template(template):
        raise CLIError(
            f"Unknown template {template!r}. Choose one of: "
            f"{', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="unknown_template",
            exit_code=1,
        )
    return resources.files("aai_cli.init.templates") / template


def target_conflict(target: Path) -> bool:
    """True when the target exists and is a non-empty directory."""
    return target.is_dir() and any(target.iterdir())


def _copy_tree(node: Traversable, dest: Path) -> None:
    for child in node.iterdir():
        name = _DOTFILE_RENAMES.get(child.name, child.name)
        out = dest / name
        if child.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            _copy_tree(child, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(child.read_bytes())


def scaffold(template: str, target: Path, *, api_key: str | None) -> Path:
    """Copy the template into `target` and write `.env`. Returns `target`."""
    root = _template_root(template)
    target.mkdir(parents=True, exist_ok=True)
    _copy_tree(root, target)
    key = api_key or PLACEHOLDER_KEY
    (target / ".env").write_text(f"ASSEMBLYAI_API_KEY={key}\n")
    return target
```

> Note on `from importlib.resources.abc import Traversable`: on Python 3.10 use `from importlib.abc import Traversable` instead. If supporting both, import defensively:
> ```python
> try:
>     from importlib.resources.abc import Traversable
> except ImportError:  # Python 3.10
>     from importlib.abc import Traversable
> ```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_init_scaffold.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/scaffold.py tests/test_init_scaffold.py
git commit -m "feat(init): scaffold (copy template tree + write .env)"
```

---

## Task 6: Runner — command builders + port/launch helpers

**Files:**
- Create: `aai_cli/init/runner.py`
- Test: `tests/test_init_runner.py`

The pure command builders and port finder are unit-tested directly; the subprocess launch is exercised in Task 7 with mocks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_runner.py
import socket
import sys
from pathlib import Path

from aai_cli.init import runner


def test_has_uv_reflects_path(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/uv")
    assert runner.has_uv() is True
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    assert runner.has_uv() is False


def test_venv_python_path_per_platform(monkeypatch):
    target = Path("/proj")
    monkeypatch.setattr(runner.os, "name", "posix")
    assert runner.venv_python(target) == target / ".venv" / "bin" / "python"
    monkeypatch.setattr(runner.os, "name", "nt")
    assert runner.venv_python(target) == target / ".venv" / "Scripts" / "python.exe"


def test_env_setup_commands_uv():
    cmds = runner.env_setup_commands(Path("/proj"), use_uv=True)
    assert cmds == [["uv", "venv"], ["uv", "pip", "install", "-r", "requirements.txt"]]


def test_env_setup_commands_venv():
    target = Path("/proj")
    cmds = runner.env_setup_commands(target, use_uv=False)
    py = str(runner.venv_python(target))
    assert cmds == [
        [sys.executable, "-m", "venv", ".venv"],
        [py, "-m", "pip", "install", "-r", "requirements.txt"],
    ]


def test_serve_command_uv_and_venv():
    target = Path("/proj")
    assert runner.serve_command(target, port=3000, use_uv=True) == [
        "uv", "run", "uvicorn", "api.index:app", "--port", "3000",
    ]
    py = str(runner.venv_python(target))
    assert runner.serve_command(target, port=3000, use_uv=False) == [
        py, "-m", "uvicorn", "api.index:app", "--port", "3000",
    ]


def test_find_free_port_returns_preferred_when_open():
    # Bind nothing on a high port we expect to be free.
    port = runner.find_free_port(0)  # 0 -> OS assigns a free port
    assert isinstance(port, int) and port > 0


def test_find_free_port_skips_taken_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    taken = s.getsockname()[1]
    s.listen(1)
    try:
        chosen = runner.find_free_port(taken)
        assert chosen != taken
    finally:
        s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.init.runner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aai_cli/init/runner.py
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import webbrowser
from pathlib import Path


def has_uv() -> bool:
    return shutil.which("uv") is not None


def venv_python(target: Path) -> Path:
    if os.name == "nt":
        return target / ".venv" / "Scripts" / "python.exe"
    return target / ".venv" / "bin" / "python"


def env_setup_commands(target: Path, *, use_uv: bool) -> list[list[str]]:
    """Commands (run with cwd=target) to create a venv and install requirements."""
    if use_uv:
        return [["uv", "venv"], ["uv", "pip", "install", "-r", "requirements.txt"]]
    import sys

    py = str(venv_python(target))
    return [
        [sys.executable, "-m", "venv", ".venv"],
        [py, "-m", "pip", "install", "-r", "requirements.txt"],
    ]


def serve_command(target: Path, *, port: int, use_uv: bool) -> list[str]:
    if use_uv:
        return ["uv", "run", "uvicorn", "api.index:app", "--port", str(port)]
    return [str(venv_python(target)), "-m", "uvicorn", "api.index:app", "--port", str(port)]


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(preferred: int, *, tries: int = 20) -> int:
    """The preferred port if free, else the next free port; OS-assigned when preferred is 0."""
    if preferred == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
    for candidate in range(preferred, preferred + tries):
        if not _port_open(candidate):
            return candidate
    return preferred


def wait_for_port(port: int, *, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.2)
    return False


def run_setup(target: Path, *, use_uv: bool) -> subprocess.CompletedProcess:
    """Run env-setup commands in order; return the first failure or the last success."""
    last: subprocess.CompletedProcess | None = None
    for cmd in env_setup_commands(target, use_uv=use_uv):
        last = subprocess.run(cmd, cwd=target, capture_output=True, text=True)
        if last.returncode != 0:
            return last
    assert last is not None
    return last


def launch_and_open(target: Path, *, port: int, use_uv: bool, open_browser: bool) -> int:
    """Start the dev server, wait for it, open the browser, and block until Ctrl-C.

    Returns the process exit code (0 on a clean Ctrl-C shutdown).
    """
    proc = subprocess.Popen(serve_command(target, port=port, use_uv=use_uv), cwd=target)
    try:
        if wait_for_port(port) and open_browser:
            webbrowser.open(f"http://localhost:{port}")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        return 0
    return proc.returncode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_init_runner.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/runner.py tests/test_init_runner.py
git commit -m "feat(init): runner command builders + port/launch helpers"
```

---

## Task 7: Step rendering helper

**Files:**
- Create: `aai_cli/init/steps.py`
- Test: `tests/test_init_steps.py`

Mirrors the `Step`/`_render_steps` shape in `aai_cli/commands/claude.py`, kept local to `init`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_steps.py
from aai_cli.init import steps


def test_render_steps_includes_name_status_detail():
    data = [
        {"name": "scaffold", "status": "created", "detail": "./my-app"},
        {"name": "install", "status": "skipped", "detail": "--no-install"},
    ]
    out = steps.render_steps(data)
    assert "scaffold" in out
    assert "created" in out
    assert "./my-app" in out
    assert "install" in out
    assert "skipped" in out


def test_render_steps_has_heading():
    out = steps.render_steps([{"name": "scaffold", "status": "created", "detail": "x"}])
    assert "init" in out.lower() or "AssemblyAI" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_steps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.init.steps'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aai_cli/init/steps.py
from __future__ import annotations

from typing import TypedDict

from rich.markup import escape

from aai_cli import theme


class Step(TypedDict):
    name: str
    status: str
    detail: str


def render_steps(items: list[Step]) -> str:
    lines = []
    for s in items:
        style = theme.status_style(s["status"])
        lines.append(
            f"  {escape(s['name'])}: "
            f"[{style}]{escape(s['status'])}[/{style}] — {escape(s['detail'])}"
        )
    return "[aai.heading]aai init:[/aai.heading]\n" + "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_init_steps.py -v`
Expected: PASS (2 tests).

> If `theme.status_style` doesn't recognize a status string it should already
> return a safe default (it does for `aai claude`). Confirm by reading
> `aai_cli/theme.py`; reuse the same status vocabulary
> (`created`/`installed`/`skipped`/`failed`/`already`).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/steps.py tests/test_init_steps.py
git commit -m "feat(init): step-status rendering helper"
```

---

## Task 8: The `init` command (wire everything) + registration

**Files:**
- Create: `aai_cli/commands/init.py`
- Modify: `aai_cli/main.py` (import, register, `_COMMAND_ORDER`)
- Modify: `aai_cli/init/__init__.py` (re-export public surface)
- Test: `tests/test_init_command.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_command.py
from pathlib import Path

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_init_scaffold_only_creates_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "myapp" / "api" / "index.py").exists()
    assert (tmp_path / "myapp" / ".env").exists()


def test_init_writes_key_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-from-env")
    runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    assert "ASSEMBLYAI_API_KEY=sk-from-env" in (tmp_path / "myapp" / ".env").read_text()


def test_init_placeholder_key_when_logged_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "your_assemblyai_api_key_here" in env


def test_init_unknown_template_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "nope", "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_refuses_nonempty_dir_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install", "--force"])
    assert result.exit_code == 0


def test_init_no_template_non_interactive_errors(tmp_path, monkeypatch):
    # CliRunner has no TTY, so the picker can't run; bare `aai init` must error helpfully.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0
    assert "transcribe" in result.output  # lists the available templates


def test_init_default_dir_is_template_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "transcribe", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcribe-app" / "api" / "index.py").exists()


def test_init_appears_in_help():
    result = runner.invoke(app, ["--help"])
    assert "init" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_init_command.py -v`
Expected: FAIL — `init` is not a registered command (non-zero exit / usage error).

- [ ] **Step 3: Implement the command**

```python
# aai_cli/commands/init.py
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import output
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.init import keys, runner, scaffold, steps, templates

app = typer.Typer(
    help="Scaffold a runnable AssemblyAI web app and open it in your browser.",
    no_args_is_help=False,  # bare `aai init` runs the interactive picker (or errors if no TTY)
)


def _pick_template() -> str:
    """Interactive picker; raises UsageError when there's no TTY to prompt on."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise UsageError(
            "No template given and not running interactively. "
            f"Pass one of: {', '.join(templates.TEMPLATE_ORDER)}."
        )
    import questionary

    choice = questionary.select(
        "Pick a template",
        choices=[
            questionary.Choice(title=templates.title_for(t), value=t)
            for t in templates.TEMPLATE_ORDER
        ],
    ).ask()
    if choice is None:  # user pressed Ctrl-C
        raise typer.Exit(code=130)
    return str(choice)


def _resolve_dir(directory: str | None, template: str, *, here: bool) -> Path:
    if here:
        return Path.cwd()
    if directory:
        return Path(directory)
    return Path.cwd() / f"{template}-app"


@app.callback(invoke_without_command=True)
def init(
    ctx: typer.Context,
    template: str = typer.Argument(None, help="Template id: transcribe, stream, agent, llm."),
    directory: str = typer.Argument(None, help="Target directory (default: <template>-app)."),
    no_install: bool = typer.Option(False, "--no-install", help="Scaffold only; don't install or launch."),
    no_open: bool = typer.Option(False, "--no-open", help="Install + launch, but don't open the browser."),
    force: bool = typer.Option(False, "--force", help="Overwrite a non-empty target directory."),
    here: bool = typer.Option(False, "--here", help="Scaffold into the current directory."),
    port: int = typer.Option(3000, "--port", help="Local server port."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Pick a template, scaffold it, install deps, launch the server, open the browser."""

    def body(state: AppState, json_mode: bool) -> None:
        chosen = template
        if chosen is None:
            chosen = _pick_template()
        if not templates.is_template(chosen):
            raise UsageError(
                f"Unknown template {chosen!r}. Choose one of: "
                f"{', '.join(templates.TEMPLATE_ORDER)}."
            )

        target = _resolve_dir(directory, chosen, here=here)
        if scaffold.target_conflict(target) and not force:
            raise UsageError(
                f"{target} already exists and is not empty. "
                f"Use --force to overwrite or pick another directory."
            )

        api_key = keys.resolve_optional_api_key(profile=state.profile)
        scaffold.scaffold(chosen, target, api_key=api_key)

        report: list[steps.Step] = [
            {"name": "scaffold", "status": "created", "detail": str(target)}
        ]
        if api_key is None:
            report.append(
                {"name": "key", "status": "skipped",
                 "detail": "no API key found; wrote a placeholder to .env (run `aai login`)"}
            )

        will_launch = not no_install and api_key is not None
        if no_install:
            report.append({"name": "install", "status": "skipped", "detail": "--no-install"})
        else:
            use_uv = runner.has_uv()
            setup = runner.run_setup(target, use_uv=use_uv)
            if setup.returncode != 0:
                report.append(
                    {"name": "install", "status": "failed",
                     "detail": (setup.stderr or setup.stdout).strip()[:300]}
                )
                will_launch = False
            else:
                report.append(
                    {"name": "install", "status": "installed",
                     "detail": "uv" if use_uv else "venv + pip"}
                )

        output.emit(report, lambda d: steps.render_steps(d), json_mode=json_mode)
        if any(s["status"] == "failed" for s in report):
            raise typer.Exit(code=1)

        if will_launch:
            chosen_port = runner.find_free_port(port)
            url = f"http://localhost:{chosen_port}"
            if not json_mode:
                output.console.print(f"[aai.heading]Starting[/aai.heading] {escape(url)}  (Ctrl-C to stop)")
            runner.launch_and_open(
                target, port=chosen_port, use_uv=runner.has_uv(), open_browser=not no_open
            )

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Register the command in `main.py`**

In `aai_cli/main.py`, add `init` to the imports:

```python
from aai_cli.commands import (
    agent,
    claude,
    doctor,
    init,
    llm,
    login,
    samples,
    stream,
    transcribe,
    transcripts,
)
```

Add `"init"` to `_COMMAND_ORDER` (place it after `samples`, with tooling commands):

```python
_COMMAND_ORDER = (
    "transcribe",
    "stream",
    "transcripts",
    "agent",
    "llm",
    "login",
    "logout",
    "whoami",
    "doctor",
    "samples",
    "init",
    "claude",
    "version",
)
```

Register the typer app (near the other `add_typer` calls):

```python
app.add_typer(init.app, name="init")
```

- [ ] **Step 5: Re-export the public surface**

```python
# aai_cli/init/__init__.py
from __future__ import annotations

from aai_cli.init.keys import resolve_optional_api_key
from aai_cli.init.scaffold import scaffold, target_conflict
from aai_cli.init.templates import TEMPLATE_ORDER, TEMPLATES, is_template, title_for

__all__ = [
    "TEMPLATES",
    "TEMPLATE_ORDER",
    "is_template",
    "title_for",
    "scaffold",
    "target_conflict",
    "resolve_optional_api_key",
]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_init_command.py -v`
Expected: PASS (9 tests).

> If `app.add_typer(init.app, name="init")` plus `@app.callback(invoke_without_command=True)`
> doesn't bind the positional `template`/`directory` arguments as expected, fall back to
> registering `init` as a single `@app.command()` on a dedicated sub-typer (mirror how
> `samples`/`claude` expose commands) — the test contract (args + flags + exit codes) is
> what matters, not the registration mechanism.

- [ ] **Step 7: Commit**

```bash
git add aai_cli/commands/init.py aai_cli/main.py aai_cli/init/__init__.py tests/test_init_command.py
git commit -m "feat(init): wire init command (picker, scaffold, install, launch) + register"
```

---

## Task 9: Full suite, lint, types, and README

**Files:**
- Modify: `README.md` (document `aai init`)
- Test: full suite

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -q`
Expected: PASS (all existing tests + the new `test_init_*` modules). Investigate and fix any regression before continuing.

- [ ] **Step 2: Lint and type-check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: clean. Fix issues inline.

> Likely ruff items: `S603`/`S607` (subprocess) are already ignored project-wide.
> The `init` package may need `# noqa` only if a new lint surfaces — prefer fixing
> over suppressing. The `importlib.resources` Traversable import may need the 3.10
> fallback from Task 5.

- [ ] **Step 3: Document the command in `README.md`**

Add a short section after the Quick start (match the README's existing tone):

````markdown
## Scaffold a starter app

```sh
aai init                  # pick a template, scaffold it, install deps, open the browser
aai init transcribe myapp # non-interactive: template + directory
```

`aai init` copies a small, self-contained FastAPI + HTML project you can run
locally and deploy to Vercel as-is. Your key is written to a git-ignored `.env`
(and is never sent to the browser). Use `--no-install` to scaffold only.
````

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document aai init"
```

---

## Self-review (completed during planning)

**Spec coverage (this plan's scope = machinery + transcribe):**
- Command surface, options, default behavior → Task 8. ✓
- Arrow-key picker + non-interactive fallback → Task 8 (`_pick_template`). ✓
- Approach A (copy committed real templates) → Tasks 4, 5. ✓
- Flat Vercel-ready layout (`api/index.py`, `index.html`, `vercel.json`) → Task 4. ✓
- Key handling: env→keyring, gitignored `.env`, placeholder + skip-launch when absent, key never in browser → Tasks 3, 5, 8 + template design. ✓
- Install/launch: uv→venv fallback, free-port, launch+open, Ctrl-C clean exit → Tasks 6, 8. ✓
- Step-status rendering + `--json` → Tasks 7, 8. ✓
- Packaging template files (incl. dotfiles) → Tasks 1, 4, 5. ✓
- Testing approach (TestClient, mocked subprocess/SDK, no real network) → Tasks 4, 6, 8. ✓
- `stream`/`agent`/`llm` templates → **out of scope; sibling plans** (stated up front). ✓ (intentional)

**Placeholder scan:** none — every code/step block is concrete.

**Type/name consistency:** `scaffold.scaffold(template, target, *, api_key)`, `scaffold.target_conflict`, `scaffold.PLACEHOLDER_KEY`, `runner.has_uv/venv_python/env_setup_commands/serve_command/find_free_port/run_setup/launch_and_open`, `steps.Step`/`steps.render_steps`, `templates.TEMPLATES/TEMPLATE_ORDER/is_template/title_for`, `keys.resolve_optional_api_key` — used consistently across tasks.
