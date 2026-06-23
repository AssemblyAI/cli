# Subagents (`task` tool) for `assembly live` (M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or :executing-plans.

**Goal:** Under `--files`, give the live agent deepagents' `task` tool — one gateway-bound, sandbox-backed, **gated** general-purpose subagent it can delegate a focused multi-step subtask to.

**Spec:** `docs/superpowers/specs/2026-06-22-live-sandboxed-execute-design.md` (Milestone **M2**). Builds on M1 (sandboxed `execute` + memory), already committed.

**Verification spike — RESOLVED (PASS).** The spec's one genuine unknown was whether a subagent's HITL write/exec interrupt surfaces at the PARENT graph's `get_state(config).interrupts` (what `brain._pending_writes` reads). A spike built a real deepagents graph with a `subagents=[…]` spec carrying `interrupt_on`, drove `main → task → subagent → write_file`, and confirmed: the parent `state.interrupts` carries the subagent's `action_requests` (write **not** run), and `Command(resume={"decisions":[{"type":"approve"}]})` then lands the write and clears the interrupt — identical to M1's main-agent gating. **Therefore: ship the FULL-TOOLS subagent** (no read-only fallback). A regression test formalizes the spike.

## Global Constraints
- `from __future__ import annotations`; modern typing. No new dependency.
- Live-only, behind `--files`. `--files`-off path byte-identical.
- **`brain.py` is at 495/500 lines** (concurrent work). Keep brain edits minimal (≈3 lines); the subagent spec lives in a NEW module `aai_cli/agent_cascade/subagents.py`.
- **AssemblyAI-only invariant:** the subagent spec MUST omit `model` (inherits the gateway-bound model). A test asserts the spec has no `model` key.
- **Gated-mutation invariant:** the subagent's `interrupt_on` mirrors `_WRITE_TOOLS` (`write_file`/`edit_file`/`execute`) so its mutations prompt through the same approver. Never an ungated mutating subagent.
- 100% patch coverage + diff-scoped mutation gate; no new escape hatches; tests-pyright via `pyright -p pyrightconfig.tests.json`.
- Concurrent session is live on this branch: commit only M2 files (selective `git add`); `AAI_ALLOW_COMMIT=1` per task; final `./scripts/check.sh` (sandbox-disabled for swift `mktemp`).

---

### Task 1: The general-purpose subagent spec (`subagents.py`)

**Files:** Create `aai_cli/agent_cascade/subagents.py`; Test `tests/test_agent_cascade_subagents.py`.

**Interface:** `general_purpose_subagent(interrupt_on: dict[str, bool]) -> dict[str, object]` — a deepagents `SubAgent` dict with `name`/`description`/`system_prompt`/`interrupt_on`, and **no** `model` or `tools` keys (both inherit: gateway-bound model, full sandboxed toolset). Takes `interrupt_on` as a param to avoid importing `_WRITE_TOOLS` from `brain` (would be circular).

- [ ] **Step 1: Tests (write first, run, see fail)**
```python
from aai_cli.agent_cascade.subagents import general_purpose_subagent

def test_spec_has_required_keys_and_no_model():
    spec = general_purpose_subagent({"write_file": True, "edit_file": True, "execute": True})
    assert spec["name"] == "general-purpose"
    assert isinstance(spec["description"], str) and spec["description"]
    assert isinstance(spec["system_prompt"], str) and spec["system_prompt"]
    # AssemblyAI-only: never a provider:model string — must inherit the gateway-bound model.
    assert "model" not in spec
    # Full-tools path: tools omitted so it inherits the sandboxed main toolset.
    assert "tools" not in spec

def test_spec_interrupt_on_gates_every_mutating_tool():
    io = {"write_file": True, "edit_file": True, "execute": True}
    spec = general_purpose_subagent(io)
    assert spec["interrupt_on"] == io  # its write_file/edit_file/execute also prompt

def test_spec_interrupt_on_is_the_passed_mapping_not_hardcoded():
    spec = general_purpose_subagent({"write_file": True})
    assert spec["interrupt_on"] == {"write_file": True}
```
- [ ] **Step 2: Implement**
```python
"""The general-purpose subagent for `assembly live --files` (deepagents' `task` tool).

One subagent the live agent delegates a focused multi-step subtask to. It OMITS `model` (so it
inherits the AssemblyAI gateway-bound model — never a provider:model string) and `tools` (so it
inherits the main sandboxed toolset, keeping its `execute` OS-confined). Its `interrupt_on`
mirrors the main agent's write tools, so its mutations prompt through the same approval loop.
"""
from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a focused coworker handling one delegated subtask in the user's project. Work in "
    "the current directory, use the available tools to research or make a contained change, and "
    "return a concise, spoken-length summary of what you did or found — not a transcript."
)

def general_purpose_subagent(interrupt_on: dict[str, bool]) -> dict[str, object]:
    """The `task` subagent spec: gateway-bound (no `model`), full sandboxed tools (no `tools`),
    with `interrupt_on` mirroring the caller's write tools so its mutations stay gated."""
    return {
        "name": "general-purpose",
        "description": (
            "Delegate a focused multi-step subtask — research, gather context, or implement a "
            "contained change — and get back a short summary. Keeps the main voice turn lean."
        ),
        "system_prompt": _SYSTEM_PROMPT,
        "interrupt_on": interrupt_on,
    }
```
- [ ] **Step 3: run tests green; commit** (`feat(live): general-purpose subagent spec for the task tool`)

---

### Task 2: Wire `subagents` + the `task` label into `brain.py`

**Files:** Modify `aai_cli/agent_cascade/brain.py`; Test `tests/test_agent_cascade_brain.py`.

**Edits (locate by content):**
- import: `from aai_cli.agent_cascade.subagents import general_purpose_subagent`
- in `_graph_kwargs` return dict (when `config.files`): add `"subagents": [general_purpose_subagent(dict.fromkeys(_WRITE_TOOLS, True))]`
- `_TOOL_LABELS`: add `"task": "Working on a subtask"`

- [ ] **Step 1: Tests** (extend the existing `_graph_kwargs` test or add):
```python
def test_graph_kwargs_wires_one_gated_gateway_bound_subagent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kwargs = brain._graph_kwargs(CascadeConfig(files=True))
    subs = kwargs["subagents"]
    assert isinstance(subs, list) and len(subs) == 1
    spec = subs[0]
    assert spec["name"] == "general-purpose"
    assert "model" not in spec  # inherits the gateway-bound model
    assert spec["interrupt_on"] == {"write_file": True, "edit_file": True, "execute": True}

def test_graph_kwargs_off_has_no_subagents():
    assert "subagents" not in brain._graph_kwargs(CascadeConfig(files=False))

def test_tool_label_task_is_working_on_a_subtask():
    assert brain._tool_label("task") == "Working on a subtask"
```
- [ ] **Step 2: Implement the 3 edits.** Re-check `wc -l brain.py < 500` after.
- [ ] **Step 3: green; commit** (`feat(live): wire the gated general-purpose subagent + task label`)

---

### Task 3: Subagent HITL-surfacing regression test (formalize the spike)

**Files:** Test `tests/test_agent_cascade_files.py` (or `_subagents` test) — a real deepagents graph with the spec, driving `main → task → subagent → write_file`, asserting the parent approval loop sees it.

**Interface consumed:** `brain._pending_writes(graph, config)`, `brain._stream_gated` / `build_streamer`.

- [ ] **Step 1: Test** — build a `_gated_graph`-style real graph WITH `subagents=[general_purpose_subagent({"write_file":True,...})]`, a `FakeChatModel` scripted `[task_call, write_file_call, AIMessage("done"), AIMessage("ok")]`. Stream one turn through `build_streamer` with a recording approver; assert the approver was consulted for the subagent's `write_file` (i.e. the interrupt surfaced through `_pending_writes`/`_decide`), and on approve the file is written under cwd; on reject it is not. (This is the go/no-go, now PASS, locked as a regression.)
- [ ] **Step 2: green; commit** (`test(live): lock subagent write surfacing through the parent gate`)

---

### Task 4: Capability phrase + docs + full gate

**Files:** `aai_cli/agent_cascade/prompt.py` (+ its test), `aai_cli/AGENTS.md`, `REFERENCE.md`.

- [ ] Advertise delegation when `--files` is on (task is bound iff `--files`): extend the `--files` capability phrase to mention delegating a bigger job to a helper. Test asserts the phrase appears under `files=True`, absent under `files=False`. (System prompt isn't snapshot-pinned.)
- [ ] `aai_cli/AGENTS.md` `--files` paragraph + `REFERENCE.md`: note the `task` delegation tool. (Coordinate with the concurrent session if `prompt.py` is dirty — commit only M2 hunks.)
- [ ] Run `./scripts/check.sh` (sandbox-disabled) to green; final commit.

## Self-Review
- Spec M2 coverage: subagent passed to `create_deep_agent` ✅ (T2); spec omits `model` ✅ (T1+T2); full-tools `interrupt_on` includes execute/write/edit ✅; `_tool_label("task")` ✅; task capability phrase ✅ (T4); HITL-surfacing spike → full-tools, regression-locked ✅ (T3). Read-only fallback NOT needed (spike PASS).
- Deferred to M3: spoken approval.
