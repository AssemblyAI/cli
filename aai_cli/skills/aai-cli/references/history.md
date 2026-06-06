# History

Two sub-apps for browsing past work. All output supports `--json` (auto-enabled
when piped) so an agent can parse listings and retrieve records by id. To run a
prompt over a past transcript's text, use `aai llm --transcript-id ID` (see
`transcription.md`).

## `aai transcripts` — browse and fetch past transcripts

Sub-app with two subcommands: `list` and `get`.

### `aai transcripts list`

List the most recent batch transcription jobs for the active account.

Key options:

- `--limit N` — how many transcripts to return (default 10).
- `--json` — machine-readable output; pipe to `jq` to extract ids.

Examples:

```bash
aai transcripts list
aai transcripts list --limit 50
aai transcripts list --json | jq '.[].id'
```

### `aai transcripts get TRANSCRIPT_ID`

Fetch a past transcript by id and print its text.

Key options:

- `-o/--output text|id|status|utterances|srt|json` — print one field; omit for
  the default human view.
- `--json` — full raw JSON.

Examples:

```bash
aai transcripts get 5551234-abcd
aai transcripts get 5551234-abcd --json
aai transcripts get 5551234-abcd -o text
```

## `aai sessions` — browse past real-time streaming sessions

Sub-app for the v3 real-time API session history, with two subcommands: `list`
and `get`.

### `aai sessions list`

List the most recent streaming sessions for the active account.

Key options:

- `--limit N` — how many sessions to return (default 10).
- `--status created|completed|error` — filter by session status.
- `--json` — machine-readable output.

Examples:

```bash
aai sessions list
aai sessions list --status completed
aai sessions list --limit 25 --json
```

### `aai sessions get SESSION_ID`

Show details for a single streaming session by id.

Key options:

- `--json` — raw JSON output.

Examples:

```bash
aai sessions get <session-id>
aai sessions get <session-id> --json
```
