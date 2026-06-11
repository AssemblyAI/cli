# History

Two sub-apps for browsing past work. All output supports `--json` (auto-enabled
when piped) so an agent can parse listings and retrieve records by id. To run a
prompt over a past transcript's text, use `assembly llm --transcript-id ID` (see
`transcription.md`).

## `assembly transcripts` — browse and fetch past transcripts

Sub-app with two subcommands: `list` and `get`.

### `assembly transcripts list`

List the most recent batch transcription jobs for the active account.

Key options:

- `--limit N` — how many transcripts to return (default 10).
- `--json` — machine-readable output; pipe to `jq` to extract ids.

Examples:

```bash
assembly transcripts list
assembly transcripts list --limit 50
assembly transcripts list --json | jq '.[].id'
```

### `assembly transcripts get TRANSCRIPT_ID`

Fetch a past transcript by id and print its text.

Key options:

- `-o/--output text|id|status|utterances|srt|json` — print one field; omit for
  the default human view.
- `--json` — full raw JSON.

Examples:

```bash
assembly transcripts get 5551234-abcd
assembly transcripts get 5551234-abcd --json
assembly transcripts get 5551234-abcd -o text
```

## `assembly sessions` — browse past real-time streaming sessions

Sub-app for the v3 real-time API session history, with two subcommands: `list`
and `get`.

### `assembly sessions list`

List the most recent streaming sessions for the active account.

Key options:

- `--limit N` — how many sessions to return (default 10).
- `--status created|completed|error` — filter by session status.
- `--json` — machine-readable output.

Examples:

```bash
assembly sessions list
assembly sessions list --status completed
assembly sessions list --limit 25 --json
```

### `assembly sessions get SESSION_ID`

Show details for a single streaming session by id.

Key options:

- `--json` — raw JSON output.

Examples:

```bash
assembly sessions get <session-id>
assembly sessions get <session-id> --json
```
