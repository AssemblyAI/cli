"""Shared fakes for the `assembly live` cascade tests (engine/command/TUI).

The cascade's three network legs and its thread spawner are injected through
``CascadeDeps``, so the suites drive the orchestration against these fakes — no
sockets, mic, or speaker. Kept in one module so the engine, command, and TUI tests
share one set of doubles (and so no single test file grows past the 500-line gate).
"""

from __future__ import annotations

import types

from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.engine import CascadeDeps, CascadeSession


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def connected(self):
        self.calls.append(("connected",))

    def user_partial(self, text):
        self.calls.append(("user_partial", text))

    def user_final(self, text):
        self.calls.append(("user_final", text))

    def tool_call(self, label):
        self.calls.append(("tool_call", label))

    def reply_started(self):
        self.calls.append(("reply_started",))

    def agent_transcript(self, text, *, interrupted):
        self.calls.append(("agent_transcript", text, interrupted))

    def reply_done(self, *, interrupted):
        self.calls.append(("reply_done", interrupted))


class FakePlayer:
    def __init__(self):
        self.enqueued = []
        self.flushed = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def enqueue(self, pcm):
        self.enqueued.append(pcm)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


class FakeWorker:
    def __init__(self, *, alive):
        self._alive = alive
        self.joined = 0

    def is_alive(self):
        return self._alive

    def join(self):
        self.joined += 1
        self._alive = False


def sync_spawn(target):
    """Run the reply body inline and hand back a finished worker, so the cascade is
    driven deterministically without real threads."""
    target()
    return FakeWorker(alive=False)


def turn(text, *, end_of_turn=True, turn_is_formatted=True):
    return types.SimpleNamespace(
        transcript=text, end_of_turn=end_of_turn, turn_is_formatted=turn_is_formatted
    )


def make_session(
    *,
    complete_reply=lambda messages, on_tool=None: "Hello there.",
    synthesize=lambda text: b"pcm:" + text.encode(),
    spawn=sync_spawn,
    run_stt=lambda on_turn: None,
    config=None,
):
    deps = CascadeDeps(
        run_stt=run_stt, complete_reply=complete_reply, synthesize=synthesize, spawn=spawn
    )
    renderer = FakeRenderer()
    player = FakePlayer()
    session = CascadeSession(
        deps=deps, renderer=renderer, player=player, config=config or CascadeConfig()
    )
    return session, renderer, player
