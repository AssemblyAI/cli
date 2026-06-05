from __future__ import annotations

import threading
from typing import TextIO

from rich.console import Console
from rich.text import Text

from aai_cli.render import BaseRenderer


class StreamRenderer(BaseRenderer):
    """Renders streaming events in one of three modes.

    - JSON: newline-delimited JSON to stdout (pipe-safe, machine-readable).
    - text: only finalized turn transcripts, one plain line each, to stdout; status
      notices ("Listening…") go to stderr. Lets `aai stream -o text | aai llm "…"`
      pipe clean transcript text downstream.
    - human (default): a live-updating line through Rich.

    Construction and the json/text/human plumbing live in BaseRenderer.
    """

    def __init__(
        self,
        *,
        json_mode: bool,
        out: TextIO | None = None,
        console: Console | None = None,
        text_mode: bool = False,
        err: TextIO | None = None,
    ) -> None:
        super().__init__(
            json_mode=json_mode,
            out=out,
            console=console,
            text_mode=text_mode,
            err=err,
        )
        self._lock = threading.RLock()

    @staticmethod
    def _with_source(payload: dict[str, object], source: str | None) -> dict[str, object]:
        if source is not None:
            payload["source"] = source
        return payload

    @staticmethod
    def _source_label(source: str) -> tuple[str, str]:
        labels = {
            "system": ("System", "aai.agent"),
            "you": ("You", "aai.you"),
        }
        return labels.get(source, (source, "aai.label"))

    @classmethod
    def _label(cls, text: str, source: str | None) -> str:
        if source is None:
            return text
        label, _style = cls._source_label(source)
        return f"{label}: {text}"

    @classmethod
    def _styled_label(cls, text: str, source: str | None) -> str | Text:
        if source is None:
            return text
        label, style = cls._source_label(source)
        rendered = Text()
        rendered.append(f"{label}: ", style=style)
        rendered.append(text)
        return rendered

    def begin(self, event: object, *, source: str | None = None) -> None:
        # The "Listening…" notice waits for the mic (see listening()); opening the
        # session only emits the protocol event for JSON consumers.
        with self._lock:
            if self.json_mode:
                self._emit(
                    self._with_source({"type": "begin", "id": getattr(event, "id", None)}, source)
                )

    def listening(self) -> None:
        """Announce capture has started — called once the mic is open and recording."""
        with self._lock:
            if self.text_mode:
                self._status("Listening… (Ctrl-C to stop)")
            elif not self.json_mode:
                self._line(Text("Listening… (Ctrl-C to stop)", style="aai.muted"))

    def turn(self, event: object, *, source: str | None = None) -> None:
        text = getattr(event, "transcript", "") or ""
        end = bool(getattr(event, "end_of_turn", False))
        with self._lock:
            if self.json_mode:
                self._emit(
                    self._with_source(
                        {"type": "turn", "transcript": text, "end_of_turn": end},
                        source,
                    )
                )
            elif self.text_mode:
                if end and text:
                    self._write(self._label(text, source) + "\n")  # plain finalized line
            elif end:
                self._finalize_line(self._styled_label(text, source))
            else:
                self._update_line(self._styled_label(text, source))

    def termination(self, event: object, *, source: str | None = None) -> None:
        with self._lock:
            if self.json_mode:
                self._emit(
                    self._with_source(
                        {
                            "type": "termination",
                            "audio_duration_seconds": getattr(
                                event, "audio_duration_seconds", None
                            ),
                        },
                        source,
                    )
                )

    def stopped(self) -> None:
        with self._lock:
            if self.text_mode:
                self._status("Stopped.")
            else:
                super().stopped()

    def close(self) -> None:
        with self._lock:
            super().close()
