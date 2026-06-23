"""Voice-controlled computer use: `assembly control`.

A local agent loop that turns spoken instructions into real macOS UI actions —
the "voice-in, hands-on-the-machine" tool that a browser/web service can't be,
because it drives the actual desktop (keystrokes, clicks, app focus) through a
native Swift helper.

The slice is split so every external leg is an injectable seam and the loop
itself is pure:

- `actions` — the action vocabulary the helper understands (pure data).
- `tools` — those actions as OpenAI function-calling tool definitions.
- `prompt` — the system prompt that briefs the model on the loop.
- `engine` — the observe/act loop over a transcript stream (no I/O of its own).
- `bridge` — adapts the LLM Gateway into the engine's `Responder` seam.
- `helper` — spawns and talks JSON to the native `macos_ui_control.swift` helper.
- `listen` — adapts mic Streaming STT into a stream of finalized utterances.
"""

from __future__ import annotations
