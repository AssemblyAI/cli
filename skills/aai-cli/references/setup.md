# Setup & Tools

Commands for scaffolding projects, validating the environment, and wiring
AssemblyAI into Claude Code.

## `aai init [TEMPLATE] [DIRECTORY]` — scaffold a starter app

Picks a template, scaffolds it into a directory, optionally installs
dependencies, starts the local server, and opens the browser. Available
templates: `audio-transcription`, `live-captions`, `voice-agent`. The API key
is written to a git-ignored `.env` file in the scaffolded directory.

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

## `aai samples` — scaffold runnable starter scripts

Sub-app for listing and scaffolding single-file Python starter scripts that read
`ASSEMBLYAI_API_KEY` from the environment.

### `aai samples list`

List the available sample script names.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai samples list
```

### `aai samples create NAME`

Scaffold a named starter script into the current directory.

Key options:

- `--force` — overwrite an existing file.
- `--json` — machine-readable output.

Examples:

```bash
aai samples create transcribe
aai samples create transcribe --force
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

## `aai claude` — wire AssemblyAI into Claude Code

Sub-app that installs the `assemblyai-docs` MCP server and the `assemblyai`
skill (plus the `aai-cli` skill) into Claude Code via `claude mcp add` /
`npx skills add`. Missing `claude` or `npx` is reported and skipped, not
treated as an error.

### `aai claude install`

Install the AssemblyAI docs MCP server and skill into Claude Code.

Key options:

- `--scope user|project|local` — config scope to register the MCP under
  (default `user`); presence is detected across all scopes.
- `--force` — reinstall even if already present.
- `--json` — machine-readable output.

Examples:

```bash
aai claude install
aai claude install --scope project
```

### `aai claude status`

Show whether the AssemblyAI MCP server and skill are currently wired into
Claude Code.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai claude status
```

### `aai claude remove`

Remove the AssemblyAI MCP server and skill from Claude Code.

Key options:

- `--scope user|project|local` — remove only from this scope (default: remove
  from whichever scope it is found in).
- `--json` — machine-readable output.

Examples:

```bash
aai claude remove
```

## `aai version` — show CLI version

Prints the installed `aai` version string.

Examples:

```bash
aai version
```
