from __future__ import annotations

import threading
from typing import TextIO

from rich.console import Console
from rich.text import Text

from aai_cli import jsonshape, theme
from aai_cli.render import BaseRenderer

# Source label -> (display text, Rich style). System audio borrows the agent color;
# the microphone ("you") its own. Unknown sources fall back to the raw label.
_SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "system": ("System", "aai.agent"),
    "you": ("You", "aai.you"),
}


def speaker_prefix(source: str | None, speaker: str | None) -> tuple[str, str] | None:
    """The lead-in label and Rich style for a turn, or None when it has neither a
    source nor a diarized speaker.

    - source + speaker -> "System (A)" (system audio diarized via --speaker-labels)
    - source only      -> "System"     (parallel system/you streams)
    - speaker only      -> "Speaker A"  (single-stream diarization, no source label)

    When a speaker is present the whole label is tinted by `theme.speaker_style` so each
    speaker reads in its own color (matching batch transcribe's diarized output); a
    sourced turn with no speaker keeps the source's own color.
    """
    label, style = (None, "aai.label")
    if source is not None:
        label, style = _SOURCE_LABELS.get(source, (source, "aai.label"))
    if speaker is not None:
        style = theme.speaker_style(speaker)
        return (f"{label} ({speaker})" if label is not None else f"Speaker {speaker}"), style
    if label is not None:
        return label, style
    return None


class StreamRenderer(BaseRenderer):
    """Renders streaming events in one of three modes.

    - JSON: newline-delimited JSON to stdout (pipe-safe, machine-readable).
    - text: only finalized turn transcripts, one plain line each, to stdout; status
      notices ("Listening…") go to stderr. Lets `assembly stream -o text | assembly llm "…"`
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
    def _label(text: str, source: str | None, speaker: str | None = None) -> str:
        prefix = speaker_prefix(source, speaker)
        return text if prefix is None else f"{prefix[0]}: {text}"

    @staticmethod
    def _styled_label(text: str, source: str | None, speaker: str | None = None) -> str | Text:
        prefix = speaker_prefix(source, speaker)
        if prefix is None:
            return text
        label, style = prefix
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
        speaker = getattr(event, "speaker_label", None)  # set when --speaker-labels diarizes
        with self._lock:
            if self.json_mode:
                # speaker is omitted entirely when undiarized (not null).
                payload = jsonshape.compact(
                    {
                        "type": "turn",
                        "transcript": text,
                        "end_of_turn": end,
                        "speaker": speaker,
                    }
                )
                self._emit(self._with_source(payload, source))
            elif self.text_mode:
                if end and text:
                    self._write(self._label(text, source, speaker) + "\n")  # plain finalized line
            elif end:
                self._finalize_line(self._styled_label(text, source, speaker))
            else:
                self._update_line(self._styled_label(text, source, speaker))

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
