# Account

Commands for authentication and read-only account queries. Auth rules and
environment precedence live in `SKILL.md`; the notes below are per-command.
All commands accept `--json`.

## `aai login` — authenticate

Opens a browser OAuth flow and stores the resulting CLI API key in the OS
keyring, bound to the active `--env`; pass `--api-key` to authenticate
non-interactively (CI).

Key options:

- `--api-key TEXT` — provide a key directly without opening a browser.
- `--json` — machine-readable confirmation output.

Examples:

```bash
aai login
aai login --api-key sk_...
```

## `aai logout` — clear stored credentials

Removes stored credentials for the active profile from the OS keyring.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai logout
```

## `aai whoami` — show active profile

Reports the active profile name and whether its stored key is currently usable
(i.e. validates against the active environment).

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai whoami
```

## `aai balance` — account balance

Read-only query that shows your remaining account balance.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai balance
```

## `aai usage` — usage over a date range

Read-only query that shows API usage; defaults to the last 30 days.

Key options:

- `--start YYYY-MM-DD` — start date (default: 30 days ago).
- `--end YYYY-MM-DD` — end date (default: today).
- `--window day|month` — bucket size for the report.
- `--all` — include zero-usage windows.
- `--json` — machine-readable output.

Examples:

```bash
aai usage
aai usage --start 2026-05-01 --end 2026-06-01
```

## `aai limits` — rate limits

Read-only query that shows your account's rate limits per service.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai limits
```

## `aai keys` — manage API keys

Sub-app for listing, creating, and renaming AssemblyAI API keys; keys are shown
masked in list output.

### `aai keys list`

List all API keys across your projects (values masked).

Key options:

- `--json` — machine-readable output, useful for scripting.

Examples:

```bash
aai keys list
aai keys list --json
```

### `aai keys create`

Create a new API key; the full key value is printed once — copy it immediately.

Key options:

- `--name TEXT` — label for the new key (required).
- `--project INTEGER` — project id to create the key in (defaults to your
  first project).
- `--json` — machine-readable output.

Examples:

```bash
aai keys create --name ci-pipeline
aai keys create --name prod --project 7
```

### `aai keys rename TOKEN_ID NEW_NAME`

Rename (relabel) an existing API key; use `aai keys list` to find the key id.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
aai keys rename 123 "prod"
```

## `aai audit` — audit log

Read-only query that lists recent audit-log entries for the account; login
events are omitted by default.

Key options:

- `--limit N` — how many entries to show (default 20).
- `--action TEXT` — filter by raw action name (e.g. `token.create`).
- `--resource TEXT` — filter by raw resource type.
- `--include-logins` — include successful login events.
- `--json` — machine-readable output.

Examples:

```bash
aai audit --limit 20
aai audit --include-logins
aai audit --action token.create
```
