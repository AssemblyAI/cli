# `aai api` — raw AssemblyAI API passthrough command

**Date:** 2026-06-09
**Status:** Design approved, ready for implementation plan

## Summary

Add `aai api`, a curl-style authenticated passthrough to the AssemblyAI REST and
LLM-gateway APIs, driven by bundled OpenAPI specs. Modeled on Vercel CLI's `vercel
api`, adapted to this CLI's conventions: bundled (not live-fetched) specs,
multi-spec/multi-host resolution through `environments.active()`, keyring-only auth,
and the errors→stderr / data→stdout split.

This lets users hit any documented endpoint — including ones with no dedicated
sub-command — without leaving the CLI's auth, environment, and output machinery.

## Motivation

The CLI wraps the `assemblyai` SDK for the common flows (`transcribe`, `stream`,
`agent`, `llm`, `transcripts`, …). Anything outside those — a new endpoint, a query
parameter we don't expose, ad-hoc scripting against `/v2/transcript` — currently
means dropping to raw `curl` and hand-managing the API key, base URL, and auth
header. `aai api` closes that gap while preserving the CLI's guarantees:

- key resolved from env→keyring (never on the command line, never in `ps`),
- base host follows `--env`/`--sandbox` automatically,
- clean stderr errors + machine-readable `--json`,
- a keyless `--show-code` escape hatch consistent with `transcribe`/`stream`/`agent`.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Spec source | **Bundled only** — vendored JSON regenerated at release time; no runtime fetch. |
| Spec scope | **REST + LLM-gateway** (request/response APIs). Websocket streaming excluded — not a passthrough fit. |
| Command surface | **Full parity** — passthrough + `list` + interactive picker + `--show-code`. |
| Codegen verb | `--show-code` → curl (this CLI's existing vocabulary), not Vercel's `--generate=FORMAT`. |
| `--api-key` flag | **Omitted** — run-style command; key resolves from env→keyring only. |

## Command surface

```
aai api <endpoint> [flags]                 # authenticated passthrough
aai api list [--api rest|llm-gateway]      # enumerate endpoints from the bundled spec
aai api                                     # interactive picker (TTY only)
```

### Flags

| Flag | Purpose |
|---|---|
| `-X/--method METHOD` | HTTP method; defaults `GET`, or `POST` when a body is present. |
| `-F/--field KEY=VALUE` | Typed field (numbers/bools/JSON parsed; `@file` reads file contents). |
| `-f/--raw-field KEY=VALUE` | String field, no type parsing. |
| `-H/--header KEY:VALUE` | Extra HTTP header. |
| `--input FILE` | Request body from file (`-` = stdin). |
| `--api {rest,llm-gateway}` | Which bundled spec/host (default `rest`; auto-inferred in `list`/picker). |
| `-i/--include` | Include response headers in output. |
| `--paginate` | Follow `page_details.next_url` (AssemblyAI's cursor shape) until exhausted. |
| `--show-code` | Print a runnable `curl` and exit — no key needed, reads `$ASSEMBLYAI_API_KEY`. |
| `--raw` | Output raw/compact JSON (no pretty-printing). |
| `--silent` | Suppress response body. |
| `--verbose` | Debug: full request + response. |

`--json` is inherited from the global behavior (auto-enabled when piped/agent-run).

There is **no `--api-key` flag** by design — consistent with `transcribe`/`stream`/`agent`.

### Behavior

- Endpoint must start with `/` (a Vercel-style guard); otherwise a `UsageError`
  pointing at interactive mode.
- The resolved URL must stay on the active environment's host (no external URLs).
- Interactive picker only runs on a TTY; non-TTY with no endpoint is a `UsageError`.

## Multi-spec & multi-host resolution (the crux)

Two bundled specs, each owning its paths and mapped to a host from the active
environment. **The spec's own `servers[].url` is ignored** — the host is derived
from `environments.active()` so `--sandbox`/`--env` work without per-endpoint
configuration:

| `--api` | Host source |
|---|---|
| `rest` | `environments.active().api_base` |
| `llm-gateway` | `environments.active().llm_gateway_base` |

### Spec-driven auth

The loader reads each spec's `components.securitySchemes`, and the command applies
the scheme rather than hardcoding per-API branches:

- REST (apiKey scheme) → `Authorization: <key>`
- LLM-gateway (http/bearer scheme) → `Authorization: Bearer <key>`

If a future spec changes its scheme, the header follows the spec, not a code edit.

## Module layout (fits the import-linter contracts)

### `aai_cli/openapi/` — new library layer

A core/library package: **never imports `aai_cli.commands`**, **never imports Rich**
(added to `.importlinter` contract 1 and contract 3).

- `loader.py` — `load_spec(api)` reads the vendored JSON, parses to typed
  `Endpoint` / `BodyField` dataclasses, resolves `$ref` and merges `allOf`.
  Mirrors Vercel's `OpenApiCache` minus all fetch/cache machinery (bundled-only).
  Exposes the security scheme per spec.
- `request.py` — pure request assembly: `-F` typed-field parsing, `@file`
  expansion, `--input`/stdin body, method defaulting. Independently testable, no I/O
  beyond reading the referenced files.
- `specs/rest.json`, `specs/llm-gateway.json` — vendored spec snapshots,
  force-included in the wheel via the existing
  `[tool.hatch.build.targets.wheel] artifacts` list.

### `aai_cli/commands/api.py` — Typer sub-app

Added to `.importlinter` contract 2 (command independence) and `_COMMAND_ORDER`.

- Builds the request via `openapi.request`, resolves host × env, applies
  spec-derived auth, executes over **`httpx2`** (already a first-class dependency —
  no new dep, deptry-clean).
- Renders Rich tables for `list` and the interactive picker.
- Runs through `context.run_command(ctx, fn, json=...)` like every other command.

### Reuse

`config.resolve_api_key`, `environments.active()`, `errors`
(`auth_failure`/`APIError`/`UsageError`), `output` (stderr/stdout + auto-`--json`),
`context.run_command`.

## Error handling

Matches `client.py`'s normalization shape:

- `401` / `403` → single clean `auth_failure()` `CLIError`.
- Other non-2xx → `APIError` carrying the response body for context.
- Network/timeout failures → `APIError`.
- No tracebacks for expected failures; errors go to stderr, data to stdout.

## Bundled-spec freshness

- **`scripts/update-openapi-specs.py`** — downloads `openapi.json` and
  `llm-gateway.yml` from the `AssemblyAI/assemblyai-api-spec` GitHub repo,
  normalizes both to JSON, writes them into `aai_cli/openapi/specs/`. Run at release
  time; documented in CLAUDE.md / AGENTS.md.
- **Parametrized contract test** (modeled on the `init` template contract tests):
  for every bundled spec — it parses, expected endpoints exist (e.g.
  `POST /v2/transcript`), a security scheme is present, and the spec maps to a known
  environment host. Catches a stale or malformed vendored spec under the gate.

## Placement & help

- New help panel **"API"** (or folded under "Setup & Tools"), slotted into
  `_COMMAND_ORDER` after `llm`.
- `--show-code` output and `list` rendering pinned by syrupy snapshots, like the
  other generators and tables.

## Testing

- Unit: `openapi.request` field parsing (typed `-F`, `@file`, stdin), `openapi.loader`
  parsing/`$ref`/`allOf`/security-scheme extraction.
- Command: host×env resolution per `--api`, auth header per scheme, method
  defaulting, error mapping (401/403 vs other), `--paginate` cursor following,
  `--show-code` curl output (snapshot), `list` table (snapshot), picker on a faked
  TTY.
- Contract: parametrized bundled-spec validation (above).
- Coverage must clear the 90% branch / 100% patch gates; no new escape hatches.

## Out of scope (YAGNI)

- Live spec fetching / caching (explicitly rejected in favor of bundled).
- Websocket streaming and voice-agent realtime endpoints.
- A generic `--generate=FORMAT` matrix beyond `--show-code` curl.
