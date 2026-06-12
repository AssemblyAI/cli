"""Opt-in diagnostic logging behind the root ``-v/--verbose`` flag.

The CLI normally configures no logging at all, and the realtime paths actively
*silence* library loggers so stderr stays clean next to the CLI's normalized
errors (``aai_cli.ws``, ``aai_cli.streaming.diagnostics``). Verbose mode is the
inverse switch: ``enable`` installs one stderr handler so library logs become
visible — ``-v`` surfaces request-level lines (httpx and friends at INFO),
``-vv`` wire-level detail (websockets frames, httpcore events at DEBUG) — and
the silencers stand down while it is ``active``.

Secrets never print in clear: ``register_secret`` records sensitive values as
they are resolved (API key, session JWT) and the handler's formatter masks them
in every rendered record. Masking must live in the formatter, not at call
sites, because the leak comes from *library* logs — websockets logs the raw
Authorization header at DEBUG during the handshake.

Binary frames are summarized, not hex-dumped: a realtime session logs dozens of
``> BINARY fb ff …`` lines per second whose payload is PCM audio — meaningless
as hex — so the formatter collapses them to ``> BINARY [9600 bytes]``, keeping
the direction and size that make ``-vv`` useful for debugging the wire.

Stdlib-only on purpose: ``config`` (a Rich-free library layer) registers
secrets here, so this module must not pull in Rich via ``output``/``theme``.
"""

from __future__ import annotations

import logging
import re
import sys

_MASK = "[redacted]"

_verbosity = 0
_secrets: set[str] = set()

# A websockets frame log's hex payload: opcode, hex byte pairs (possibly elided
# with "..." by the library's own 75-char cap), then the "[N bytes]" metadata.
# CONT is included because a fragmented binary message continues as CONT frames.
_BINARY_FRAME_HEX = re.compile(r"\b(BINARY|CONT) [0-9a-f][0-9a-f. ]* \[")


class _RedactingFormatter(logging.Formatter):
    """Formats records normally, then masks every registered secret and
    collapses websockets binary-frame hex dumps to their byte count."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in _secrets:
            text = text.replace(secret, _MASK)
        if record.name.partition(".")[0] == "websockets":
            text = _BINARY_FRAME_HEX.sub(r"\1 [", text)
        return text


def register_secret(value: str | None) -> None:
    """Record a sensitive value so verbose output masks it. Empty values are
    ignored (replacing "" would shred every record)."""
    if value:
        _secrets.add(value)


def active() -> bool:
    """Whether verbose logging is on — the realtime silencers stand down then."""
    return _verbosity > 0


def enable(verbosity: int) -> None:
    """Install the stderr diagnostics handler: ``-v`` (1) at INFO, ``-vv``+ at DEBUG.

    Zero is the everyday no-op — no handler, the CLI stays log-silent. The
    handler goes on the root logger so third-party loggers (httpx, websockets,
    the assemblyai SDK) are covered without naming each one; stderr keeps the
    errors-to-stderr / data-to-stdout split intact for pipelines.
    """
    global _verbosity
    if verbosity <= 0:
        return
    _verbosity = verbosity
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_RedactingFormatter("[%(name)s] %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO if verbosity == 1 else logging.DEBUG)
