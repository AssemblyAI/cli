"""Turn the microphone into a stream of finalized spoken instructions.

The engine consumes an ``Iterable[str]`` of utterances; this adapts mic
Streaming STT into exactly that. The blocking stream runs on a worker thread and
pushes each finalized turn onto a queue that the generator drains, so a turn the
agent is still acting on doesn't drop the next thing you say. The stream call and
the microphone are injected, so the queue/threading logic is unit-tested with a
fake stream that just invokes the turn callback — no real audio, socket, or
``websockets`` thread.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator

from assemblyai.streaming.v3 import StreamingParameters

from aai_cli.core import client
from aai_cli.core.errors import CLIError
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.streaming.sources import TARGET_RATE

# Run one streaming session: same shape as ``client.stream_audio``'s keyword call.
type StreamRunner = Callable[..., None]
# Build the mic byte source for a device/rate.
type MicFactory = Callable[..., Iterable[bytes]]


def _finalized_text(event: object) -> str | None:
    """The spoken text of a finalized, non-empty turn, or None for a partial/empty one."""
    if not getattr(event, "end_of_turn", False):
        return None
    text = getattr(event, "transcript", "") or ""
    return text or None


def _build_mic(device: int | None, sample_rate: int | None, mic_factory: MicFactory) -> object:
    """Construct the mic source (kept tiny so the default factory stays substitutable)."""
    return mic_factory(device=device, capture_rate=sample_rate)


def listen(
    api_key: str,
    *,
    device: int | None = None,
    sample_rate: int | None = None,
    stream: StreamRunner = client.stream_audio,
    mic_factory: MicFactory = MicrophoneSource,
) -> Iterator[str]:
    """Yield finalized spoken utterances from the mic until the stream ends.

    Surfaces a streaming error (raised on the worker) to the caller after the
    queue drains, so a connection failure isn't silently swallowed.
    """
    utterances: queue.Queue[str | None] = queue.Queue()
    failure: list[CLIError] = []

    def on_turn(event: object) -> None:
        text = _finalized_text(event)
        if text is not None:
            utterances.put(text)

    def worker() -> None:
        try:
            mic = _build_mic(device, sample_rate, mic_factory)
            rate = getattr(mic, "sample_rate", None)
            sample = rate if isinstance(rate, int) else (sample_rate or TARGET_RATE)
            params = StreamingParameters(sample_rate=sample, format_turns=True)
            stream(api_key, mic, params=params, on_turn=on_turn)
        except CLIError as exc:
            # The streaming legs raise CLIError/APIError; capture it and re-raise on
            # the main thread so a connection failure isn't lost on the worker.
            failure.append(exc)
        finally:
            utterances.put(None)

    # daemon=True is an interpreter-exit safety net (a wedged mic worker can't block
    # shutdown); not observable from a test, which always drains to the sentinel.
    thread = threading.Thread(
        target=worker,
        name="aai-control-listen",
        daemon=True,  # pragma: no mutate
    )
    thread.start()
    while True:
        item = utterances.get()
        if item is None:
            break
        yield item
    thread.join()
    if failure:
        raise failure[0]
