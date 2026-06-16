# CLI reference

The contracts scripts and agents can rely on: exit codes, environment
variables, configuration precedence, and machine-readable output shapes.

## Exit codes

Stable, and deliberately split the way `gh` splits them (the source of truth
is the docstring in `aai_cli/errors.py`):

| Code | Meaning |
| ---- | ------- |
| `0` | Success. |
| `1` | Generic runtime failure: an API/network error, a missing dependency, or an unexpected internal error. |
| `2` | Usage/validation error: bad flags, a bad path, a malformed id, or an unusable config file. |
| `4` | Not authenticated: no usable credential, a rejected key, or a self-service command that needs a browser login. |
| `130` | Cancelled with Ctrl-C. |

A subprocess the CLI shells out to (`assembly deploy`, `assembly dev`,
`assembly update`) propagates that process's own exit code unchanged. Under
`--json`, every failure also emits one `{"error": {"type": …, "message": …}}`
object on stderr; the `error.type` pairs 1:1 with the exit code.

## Environment variables

Product-scoped variables are `ASSEMBLYAI_*`; CLI-behavior variables are
`AAI_*`. Keep new variables in that split.

| Variable | Effect |
| -------- | ------ |
| `ASSEMBLYAI_API_KEY` | API key for all API calls; beats the keyring, loses to nothing but a `--api-key` validation flag. |
| `AAI_ENV` | Backend environment (`production`, `sandbox000`); beats the profile's stored env, loses to `--env`/`--sandbox`. |
| `AAI_AUTH_PORT` | Loopback callback port for `assembly login` (dev/test only; default 8585). |
| `AAI_NO_UPDATE_CHECK` | Disables the "update available" notice and its background refresh. |
| `AAI_TELEMETRY_DISABLED` / `DO_NOT_TRACK` | Disables anonymous usage telemetry (always beats the persisted choice). |
| `NO_COLOR` / `FORCE_COLOR` | Standard color overrides; `--color always` / `--color never` sets them for child consoles too. |
| `CI` | Suppresses interactive affordances (spinners, the update notice); never changes output shape. |

## Configuration and precedence

Non-secret settings persist in `config.toml` (`assembly config path` prints
where; `assembly config list/get/set` reads and writes it). The API key lives
only in the OS keyring — never in a file.

Precedence, highest first:

1. Command flags (`--profile`, `--env`/`--sandbox`).
2. Environment variables (`ASSEMBLYAI_API_KEY`, `AAI_ENV`).
3. Stored settings (`config.toml` + keyring): the active profile, its env
   binding, and its key.
4. Built-in defaults (`production`, profile `default`).

## Non-interactive authentication

Pipe the key on stdin so it never reaches shell history or `ps`:

```sh
printenv ASSEMBLYAI_API_KEY | assembly login --with-api-key
```

Or skip storage entirely and set `ASSEMBLYAI_API_KEY` per invocation. On a
remote/SSH machine the browser flow also works by forwarding the callback
port (`ssh -L 8585:127.0.0.1:8585 <host>`) and opening the printed URL in
your local browser.

## JSON output

`--json` (or `-o json`) is always an explicit opt-in — piping never switches
the output shape. One-shot commands emit a single JSON object on stdout;
errors and warnings are single JSON objects on stderr.

Streaming commands emit newline-delimited JSON (NDJSON), one event per line,
each carrying a `"type"` field to dispatch on:

| Command | Event types |
| ------- | ----------- |
| `assembly stream --json` | `begin`, `turn`, `termination` (with `--from-stdin`, a `source` event precedes each file's events) |
| `assembly agent --json` | `session.ready`, `transcript.user.delta`, `transcript.user`, `reply.started`, `transcript.agent`, `reply.done` |
| `assembly agent-cascade --json` | `session.ready`, `transcript.user.delta`, `transcript.user`, `reply.started`, `transcript.agent`, `reply.done` |
| `assembly dictate --json` | `utterance` |
| `assembly llm --follow --json` | `answer` |
| `assembly transcribe <batch> --json` | `result` (one per source), then `reduce` if `--llm-reduce` is set |

New event types may be added; existing fields are stable. Consumers should
ignore types they don't recognize.

With `--llm-reduce`, batch mode emits one final
`{"type":"reduce","model","prompts","output"}` record after the per-source
`result` records — the aggregate prompt(s) run once over every result, with the
output printed to stdout (the progress table is routed to stderr so stdout stays
clean for piping). `--llm-reduce` is repeatable, each prompt running on the
previous one's output; for a single source it extends the `--llm` chain over
that transcript.

`assembly eval` takes the same `--llm`/`--llm-reduce` flags but emits a single
JSON object (not NDJSON): `--llm` runs a chain over each transcript and attaches
`{"model","steps"}` under the row's `llm` key (the WER score still uses the raw
transcript), and `--llm-reduce` runs one prompt over every item's result and
adds a top-level `reduce` (`{"model","prompts","output"}`) to the object.
