from __future__ import annotations

from assemblyai_cli.render import BaseRenderer


class StreamRenderer(BaseRenderer):
    """Renders streaming events: a live-updating line for humans, NDJSON for agents."""

    def begin(self, event: object) -> None:
        if self.json_mode:
            self._emit({"type": "begin", "id": getattr(event, "id", None)})
        else:
            self._write("Listening… (Ctrl-C to stop)\n")

    def turn(self, event: object) -> None:
        text = getattr(event, "transcript", "") or ""
        end = bool(getattr(event, "end_of_turn", False))
        if self.json_mode:
            self._emit({"type": "turn", "transcript": text, "end_of_turn": end})
        elif end:
            self._finalize_line(text)
        else:
            self._update_line(text)

    def termination(self, event: object) -> None:
        if self.json_mode:
            self._emit(
                {
                    "type": "termination",
                    "audio_duration_seconds": getattr(event, "audio_duration_seconds", None),
                }
            )
