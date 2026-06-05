# `aai claude` — Wire up Claude Code for AssemblyAI

**Date:** 2026-06-02
**Status:** Approved design

## Problem

Developers building with AssemblyAI through a coding agent get better, more
current code when their agent is connected to AssemblyAI's live context. Two
artifacts already exist for this:

1. **Docs MCP server** — remote, Streamable HTTP at
   `https://mcp.assemblyai.com/docs`. Exposes `search_docs`, `get_pages`,
   `list_sections`, `get_api_reference`. "Installing" it means registering the
   URL in the client's MCP config.
2. **Claude Code skill** — the `AssemblyAI/assemblyai-skill` GitHub repo, a
   `skills/assemblyai/` directory (a `SKILL.md` plus reference docs) installed
   via the universal *skills* CLI: `npx skills add AssemblyAI/assemblyai-skill`,
   landing in `~/.claude/skills/assemblyai/`.

Today a developer must find the docs page and run the install steps by hand. We
want a single CLI command that wires both into Claude Code.

### Discrepancy to flag (out of scope for this CLI change)

The published docs (`coding-agent-prompts.mdx`, `agent-instructions.mdx`)
instruct users to run `claude install-skill <url>` and `claude skill list`.
**These subcommands do not exist** in Claude Code (verified against 2.1.161;
the real surface is `claude mcp …` and `claude plugin …`). The canonical skill
installer is `npx skills add AssemblyAI/assemblyai-skill`. The docs need a fix
independent of this work; noted here so it isn't lost.

## Scope

- **Target client:** Claude Code only.
- **Install method:** shell out to existing tools (`claude` for MCP, `npx
  skills` for the skill) rather than writing config formats natively.
- **Out of scope:** Cursor/Windsurf/other clients; running our own MCP proxy
  (AssemblyAI's MCP is remote-hosted, so there is nothing to run); fixing the
  upstream docs.

## Command surface

A new `claude` command group in `assemblyai_cli/commands/claude.py`, registered in
`main.py`:

```python
app.add_typer(claude.app, name="claude")
```

| Command | Behavior |
|---|---|
| `aai claude install` | Installs **both** the docs MCP server and the skill into Claude Code. |
| `aai claude status` | Reports whether each artifact is currently wired up. |
| `aai claude remove` | Unwinds both. |

Flags:

- `install`: `--scope {user,project,local}` (default `user`), `--force`, `--json`.
- `status`: `--json`.
- `remove`: `--scope {user,project,local}` (default `user`), `--json`.

Rationale for naming: peer CLI Deepgram established the verb vocabulary
(`install` / `status` / `remove`) on its `skills` group. Its `mcp` vs `skills`
noun split does not fit here because Deepgram's `mcp` *runs* a stdio proxy,
whereas AssemblyAI's MCP is remote-hosted and only needs registering. "agent"
names the thing being configured (the user's coding agent) and unifies both
artifacts under the single install command requested.

## Constants

```python
MCP_NAME = "assemblyai-docs"
MCP_URL = "https://mcp.assemblyai.com/docs"
SKILL_REPO = "AssemblyAI/assemblyai-skill"
SKILL_DIR = Path.home() / ".claude" / "skills" / "assemblyai"  # contains SKILL.md
```

## Behavior

### `aai claude install`

Two independent steps, each preflighted, run via `subprocess.run` (output
captured), and reported individually.

1. **MCP step** — requires `claude` on PATH.
   - Idempotency: check `claude mcp get assemblyai-docs`. If present, report
     `already` and skip — unless `--force`, in which case `claude mcp remove
     assemblyai-docs --scope <scope>` then re-add.
   - Install:
     `claude mcp add --transport http --scope <scope> assemblyai-docs https://mcp.assemblyai.com/docs`
2. **Skill step** — requires `npx` on PATH.
   - `npx skills add AssemblyAI/assemblyai-skill` (re-runnable; it updates /
     de-dupes on its own, so `--force` simply re-runs it).

### Dependency detection & graceful partial behavior

- Detect each tool with `shutil.which` **before** running its step (`claude`
  for the MCP step, `npx` for the skill step).
- A missing required tool makes that step `skipped`, with a one-line fix and the
  exact command printed so the user can run it manually (e.g. "Install Node.js
  to get `npx`", "Install Claude Code: https://claude.com/claude-code").
- Steps are independent: a missing `npx` does not block the MCP install.
- Exit code is non-zero only when a step that *could* have run actually
  **failed** — a step skipped for a missing tool does not fail the command. The
  final summary states what succeeded, was skipped, and failed.

### `aai claude status`

- MCP present? — `claude mcp get assemblyai-docs` exit code (or parse `claude
  mcp list`). If `claude` is missing, report MCP status as `unknown` with
  guidance.
- Skill present? — `SKILL_DIR / "SKILL.md"` exists.
- Emit a two-row report (MCP, skill).

### `aai claude remove`

- MCP — `claude mcp remove assemblyai-docs --scope <scope>`.
- Skill — delete the `SKILL_DIR` directory directly. Removing a directory of
  markdown is safe and avoids guessing the `skills` CLI's removal syntax.
- Report per artifact. Absent artifacts report `not installed` (not an error).

## Error handling & output

- Command bodies run through `context.run_command`; failures raise `CLIError`
  with descriptive `error_type`s (e.g. `claude_not_found`, `npx_not_found`,
  `mcp_install_failed`, `skill_install_failed`).
- Human and JSON output via `output.emit`; JSON mode auto-engages under agents
  via the existing `output.resolve_json`.
- JSON shape:

```json
{
  "steps": [
    { "name": "mcp",   "status": "installed", "detail": "assemblyai-docs @ user scope" },
    { "name": "skill", "status": "already",   "detail": "~/.claude/skills/assemblyai" }
  ]
}
```

`status` values: `installed`, `already`, `skipped`, `failed`, `removed`,
`not_installed`, `unknown`.

## File layout

- `assemblyai_cli/commands/claude.py` — new command group (`install`, `status`,
  `remove`) plus a small `_run(cmd: list[str]) -> subprocess.CompletedProcess`
  helper and the per-step functions.
- `assemblyai_cli/commands/__init__.py` import + `main.py` registration.
- No new runtime dependencies (`subprocess`, `shutil`, `pathlib` are stdlib).

## Testing

`tests/test_claude.py`, mirroring `tests/test_samples.py`. All `subprocess.run`
and `shutil.which` calls are monkeypatched — no real `claude`/`npx` invocation.

Cases:

- `install` happy path: asserts exact argv for both steps, including
  `--transport http --scope user`.
- `--scope project` / `--scope local` passthrough.
- Idempotency: MCP already present → `already`, no re-add; `--force` → remove
  then add.
- Partial: `claude` missing → MCP `skipped`, skill still installs, exit 0;
  `npx` missing → skill `skipped`, MCP installs, exit 0.
- Failure: `claude mcp add` non-zero exit → step `failed`, command exit non-zero.
- `--json` shape for `install`/`status`.
- `status`: each combination of MCP present/absent and skill dir present/absent.
- `remove`: removes both; absent artifacts report `not_installed`, not an error.
- Smoke: `aai claude --help` registers the subcommands.

## Docs

- Add an "AI coding agents" section to `README.md` documenting `aai claude
  install`.
- Separately flag the upstream docs discrepancy (the non-existent `claude
  install-skill` / `claude skill list`) to the docs owners. Not part of this
  CLI change.
