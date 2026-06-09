# Setup & Tools

Commands for scaffolding projects, validating the environment, and setting up
your coding agent (docs MCP + skills) for AssemblyAI.

## `aai init [TEMPLATE] [DIRECTORY]` — scaffold a starter app

**This is how you build a new app.** When the goal is to create an
application or project — including a voice-agent app — start here, not with the
`aai agent` / `aai transcribe` / `aai stream` run commands (those just *run* a
one-off action in the terminal and produce no code).

Picks a template, scaffolds it into a directory, optionally installs
dependencies, starts the local server, and opens the browser. Available
templates: `audio-transcription`, `live-captions`, `voice-agent`. The API key
is written to a git-ignored `.env` file in the scaffolded directory.

To build a **voice agent app**, use `aai init voice-agent` (a full
FastAPI + browser starter) — `aai agent` only runs a live mic conversation and
writes no code.

Key options:

- `--no-install` — scaffold only; skip install and launch.
- `--no-open` — install and launch but don't open the browser.
- `--force` — overwrite a non-empty target directory.
- `--here` — scaffold into the current directory instead of a new subdirectory.
- `--port INTEGER` — local server port (default 3000).
- `--json` — machine-readable output.

Examples:

```bash
aai init
aai init audio-transcription my-app
aai init audio-transcription --here
```

## `aai doctor` — environment health check

Verifies that your environment is ready to use AssemblyAI (checks credentials,
network reachability, and runtime dependencies).

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai doctor
```

## `aai setup` — set up your coding agent for AssemblyAI

Sub-app that wires three things into your coding agent: the `assemblyai-docs`
MCP server (via `claude mcp add`), the `assemblyai` skill (downloaded with
`npx skills add`), and the `aai-cli` skill (this skill — bundled in the pip
package and copied in directly, no network needed). Missing `claude` or `npx`
is reported and skipped, not treated as an error; the bundled `aai-cli` skill
installs regardless.

### `aai setup install`

Install the docs MCP server and both skills into your coding agent.

Key options:

- `--scope user|project|local` — config scope to register the MCP under
  (default `user`); presence is detected across all scopes.
- `--force` — reinstall even if already present.
- `--json` — machine-readable output.

Examples:

```bash
aai setup install
aai setup install --scope project
```

### `aai setup status`

Show whether the MCP server and both skills are currently set up.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai setup status
```

### `aai setup remove`

Remove the MCP server and both skills from your coding agent.

Key options:

- `--scope user|project|local` — remove the MCP only from this scope (default:
  remove from whichever scope it is found in).
- `--json` — machine-readable output.

Examples:

```bash
aai setup remove
```

## `aai --version` — show CLI version

Prints the installed `aai` version string and exits.

Examples:

```bash
aai --version
```
