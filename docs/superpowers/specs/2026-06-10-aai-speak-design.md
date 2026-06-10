# `aai speak` — streaming text-to-speech (sandbox-only)

**Date:** 2026-06-10
**Status:** Approved design

## Summary

Add an `aai speak` command that synthesizes speech from text via the sandbox
streaming-TTS WebSocket and plays it through the speakers by default, or writes
a WAV file when `--out` is given. The feature only exists in the sandbox
(`streaming-tts.sandbox000.assemblyai-labs.com`); running it against the default
production environment fails with a clean, actionable error.

```sh
aai speak "Hello there, friend."                       # play through speakers
aai speak "Hello there." --voice jane --language English
echo "Hello" | aai speak                               # text from stdin
aai speak "Hi" --out /tmp/out.wav                      # write a WAV instead of playing
aai speak "Hi" --json                                  # machine-readable metadata
```

## Behavior

- **Sandbox-only.** TTS only exists at `streaming-tts.sandbox000.assemblyai-labs.com`.
  Running against production (the shipped default) raises a `CLIError`
  instructing the user to pass `--sandbox` (exit 2).
- **Requires an API key**, resolved the normal way (`ASSEMBLYAI_API_KEY` env →
  OS keyring). The key is sent as `Authorization: Bearer <key>`, mirroring the
  `agent` command. A missing key takes the standard not-authenticated path
  (exit 4). No `--api-key` flag (keys must not leak into shell history / `ps`).
- **Text source:** the positional `TEXT` argument, or stdin when the argument is
  omitted (`echo … | aai speak`). Empty text from both → usage error (exit 2).
- **Connection params** (`--voice`, `--language`, `--sample-rate`) are only added
  to the WebSocket query string when explicitly passed; otherwise the server
  applies its own defaults (currently `Vivian` / `English` / `24000`). The server
  stays the source of truth — no hardcoded client-side voice list.
- **Output:** with no `--out`, play the audio through the speakers (uses
  `sounddevice`, the dependency `agent` already relies on). With `--out PATH`,
  write a WAV instead. Headless machines have no speakers — documented; use
  `--out` there.

## Protocol

Reference: `assemblyai/engineering/projects/realtime/api_tts/scripts/sample_session.py`
in the DeepLearning monorepo. A synchronous `websockets` client (same library and
pattern as `aai_cli/agent/session.py`):

1. Connect to `wss://{tts_host}/v1/ws/?voice=…&language=…&sample_rate=…` — only
   the params the user set are included in the query string.
2. Receive a `Begin` frame (`{"type":"Begin","id":…,"expires_at":…}`); anything
   else is an error.
3. Send one `Generate` frame: `{"type":"Generate","text": <text>}`.
4. Send a `Flush` frame: `{"type":"Flush"}` to trigger synthesis.
5. Receive `Audio` frames: `{"type":"Audio","audio": <base64 pcm>,
   "sample_rate": int, "encoding": str, "is_final_for_flush": bool}`. Base64-decode
   `audio`, accumulate PCM, track `sample_rate`, and stop when
   `is_final_for_flush` is `true`.
   - `Warning` frames → printed to stderr, non-fatal.
   - `Error` frames (`{"type":"Error","error_code":int,"error":str}`) → mapped to
     a clean `CLIError`.
6. Send `Terminate` (`{"type":"Terminate"}`); read the optional `Termination`
   frame (`session_duration_seconds`, `total_input_char_length`,
   `generated_audio_duration_seconds`) for the final stats.

Audio is 16-bit mono PCM. The WAV is written with `nchannels=1`, `sampwidth=2`,
`framerate=sample_rate` (read from the `Audio` frames, not assumed).

## Components

Mirrors the existing `streaming/` and `agent/` feature subsystems.

- **`aai_cli/environments.py`** — add a `streaming_tts_host: str` field to the
  frozen `Environment` dataclass. `sandbox000` →
  `"streaming-tts.sandbox000.assemblyai-labs.com"`; `production` → `""` (the
  empty string is the sandbox-only signal the command checks).

- **`aai_cli/tts/session.py`** — the protocol client. Connects, drives
  Begin→Generate→Flush→Audio→Terminate, and returns
  `(pcm_bytes, sample_rate, termination_stats)`. The `connect` factory is
  injectable (defaults to the `websockets` sync client) so tests run hermetically
  with a fake socket. Connect/auth failures map through the existing
  `errors.py` helpers (`auth_failure` / `is_auth_failure`); a rejected key →
  not-authenticated, other failures → `APIError`. `websockets` internal logging
  is silenced for the session (same helper shape as `agent`).

- **`aai_cli/tts/audio.py`** — `write_wav(path, pcm, sample_rate)` (stdlib
  `wave`) and `play_pcm(pcm, sample_rate)` (`sounddevice`).

- **`aai_cli/commands/speak.py`** — Typer sub-app run through
  `context.run_command`. Parses options, enforces sandbox-only, resolves the key,
  reads text (arg or stdin), drives the session, then plays or writes. Human mode
  shows a status spinner while synthesizing plus a final note; `--json` emits
  `{voice, language, sample_rate, audio_duration_seconds, bytes, out}`.

- **`aai_cli/main.py`** — register the `speak` sub-app and slot it into
  `_COMMAND_ORDER` near `agent` / `stream`.

## Error handling

| Condition | Result |
| --- | --- |
| Production environment (no sandbox) | `CLIError`, exit 2, suggestion: re-run with `--sandbox` |
| No API key | not-authenticated path, exit 4 |
| Empty text (arg + stdin both empty) | usage `CLIError`, exit 2 |
| Server `Error` frame / rejected key / unexpected first frame | mapped to `APIError` or auth failure — never a traceback |

## Testing

Must clear the project's mutation and 100%-patch-coverage tail gates, so tests
assert *behavior*, not just line execution.

- `tts/session.py`: a fake injected websocket drives the full
  Begin→Generate→Flush→Audio→Terminate flow. Assert the exact frames sent, the
  base64 decode, stop-on-`is_final_for_flush`, `Error`/`Warning` handling, and
  auth-failure mapping (rejected key → exit 4).
- `tts/audio.py`: `write_wav` produces a WAV with the correct header
  (mono / 16-bit / given rate) and body bytes; `play_pcm` invokes a monkeypatched
  `sounddevice`.
- `commands/speak.py` (Typer `CliRunner`): production → clean error; missing key
  → exit 4; stdin path; query-param building per flag (only set params appear);
  `--out` writes a file; default plays (monkeypatched playback); `--json` shape
  asserted on the behavioral split (a parsed JSON object, not human text).
- `environments`: new field present; production host empty.
- Regenerate the syrupy snapshots for `aai speak --help` and the updated
  `aai --help` command ordering (`--snapshot-update`); never hand-edit `.ambr`.

No replay fixture is added: the replay harness records real REST responses, but
TTS is a WebSocket. The injected-`connect` unit tests provide the offline
end-to-end coverage instead.

## Out of scope (v1)

- `--show-code` (the protocol is a custom WebSocket client, not an SDK call).
- Multiple `Generate` segments, `--file` input, and `--keepalive`.
- A hardcoded client-side voice list.
- Any production wiring (TTS does not yet exist in production).
