# Account

Commands for authentication and read-only account queries. Auth rules and
environment precedence live in `SKILL.md`; the notes below are per-command.
All commands accept `--json`.

## `assembly login` — authenticate

Opens a browser OAuth flow and stores the resulting CLI API key in the OS
keyring. Login defaults to **production** — unlike other commands it does not
inherit the profile's previously-stored env, so a bare re-login never silently
re-targets a sandbox; pass `--sandbox`/`--env` (or set `AAI_ENV`) to sign in
elsewhere, and that becomes the env the profile is bound to. Pass `--api-key`
to authenticate non-interactively (CI).

Key options:

- `--api-key TEXT` — provide a key directly without opening a browser.
- `--json` — machine-readable confirmation output.

Examples:

```bash
assembly login
assembly login --api-key sk_...
```

## `assembly logout` — clear stored credentials

Removes stored credentials for the active profile from the OS keyring.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
assembly logout
```

## `assembly whoami` — show active profile

Reports the active profile name and whether its stored key is currently usable
(i.e. validates against the active environment).

Key options:

- `--json` — machine-readable output.

Examples:

```bash
assembly whoami
```

## `assembly balance` — account balance

Read-only query that shows your remaining account balance.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
assembly balance
```

## `assembly usage` — usage over a date range

Read-only query that shows API usage; defaults to the last 30 days.

Key options:

- `--start YYYY-MM-DD` — start date (default: 30 days ago).
- `--end YYYY-MM-DD` — end date (default: today).
- `--window day|month` — bucket size for the report.
- `--all` — include zero-usage windows.
- `--json` — machine-readable output.

Examples:

```bash
assembly usage
assembly usage --start 2026-05-01 --end 2026-06-01
```

## `assembly limits` — rate limits

Read-only query that shows your account's rate limits per service.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
assembly limits
```

## `assembly keys` — manage API keys

Sub-app for listing, creating, and renaming AssemblyAI API keys; keys are shown
masked in list output.

### `assembly keys list`

List all API keys across your projects (values masked).

Key options:

- `--json` — machine-readable output, useful for scripting.

Examples:

```bash
assembly keys list
assembly keys list --json
```

### `assembly keys create`

Create a new API key; the full key value is printed once — copy it immediately.

Key options:

- `--name TEXT` — label for the new key (required).
- `--project INTEGER` — project id to create the key in (defaults to your
  first project).
- `--json` — machine-readable output.

Examples:

```bash
assembly keys create --name ci-pipeline
assembly keys create --name prod --project 7
```

### `assembly keys rename TOKEN_ID NEW_NAME`

Rename (relabel) an existing API key; use `assembly keys list` to find the key id.

Key options:

- `--json` — machine-readable output.

Examples:

```bash
assembly keys rename 123 "prod"
```

## `assembly audit` — audit log

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
assembly audit --limit 20
assembly audit --include-logins
assembly audit --action token.create
```
