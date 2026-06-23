"""Shared fakes for the ``assembly control`` test modules (``test_control*.py``).

Every external leg (mic Streaming STT, the LLM Gateway, the native Swift helper)
is faked here so each test module drives the control loop with no microphone,
network, subprocess, or macOS.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import IO

from openai.types.chat import ChatCompletion

from aai_cli.commands.control import _exec as control_exec
from aai_cli.control import engine, helper
from aai_cli.control.actions import Action

OPTS = control_exec.ControlOptions(
    device=None, sample_rate=None, model="m", max_tokens=8, max_steps=4, dry_run=False
)


class RecordingRenderer:
    """A Renderer that records every event for assertions."""

    def __init__(self) -> None:
        self.users: list[str] = []
        self.actions: list[Action] = []
        self.results: list[tuple[Action, dict[str, object]]] = []
        self.refused: list[tuple[Action, str]] = []
        self.invalid: list[str] = []
        self.replies: list[str] = []

    def on_user(self, text: str) -> None:
        self.users.append(text)

    def on_action(self, action: Action) -> None:
        self.actions.append(action)

    def on_result(self, action: Action, result: dict[str, object]) -> None:
        self.results.append((action, result))

    def on_refused(self, action: Action, reason: str) -> None:
        self.refused.append((action, reason))

    def on_invalid(self, reason: str) -> None:
        self.invalid.append(reason)

    def on_reply(self, text: str) -> None:
        self.replies.append(text)


def scripted(replies: list[engine.Reply]) -> engine.Responder:
    """A responder that returns the next scripted reply on each call."""
    calls = iter(replies)

    def respond(messages: list[engine.Message]) -> engine.Reply:
        return next(calls)

    return respond


def fake_completion(content, tool_calls) -> ChatCompletion:
    # Build a real ChatCompletion the lenient way the SDK parses a wire response
    # (model_construct), stuffing SimpleNamespace internals so we needn't hand-build
    # every nested SDK model — the replay-fixtures idiom (see tests/AGENTS.md).
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return ChatCompletion.model_construct(choices=[SimpleNamespace(message=message)])


class FakeProc:
    """A stand-in helper process with in-memory JSON-lines pipes."""

    def __init__(self, response_lines: str, *, stdin: IO[str] | None = None) -> None:
        self.stdin: IO[str] | None = io.StringIO() if stdin is None else stdin
        self.stdout: IO[str] | None = io.StringIO(response_lines)
        self.terminated = False
        self._exit: int | None = None

    def poll(self) -> int | None:
        return self._exit

    def terminate(self) -> None:
        self.terminated = True
        self._exit = 0

    def wait(self, timeout: float | None = None) -> int | None:
        return self._exit


class BrokenStdin(io.StringIO):
    def write(self, _data: str, /) -> int:
        raise OSError("broken pipe")


class FakeMic:
    """An iterable-of-bytes mic stand-in that also reports a sample rate."""

    sample_rate = 16000

    def __iter__(self) -> Iterator[bytes]:
        return iter(())


class RecordingHelper(helper.UiHelper):
    """A real UiHelper (so it satisfies the dep type) that records close()."""

    def __init__(self) -> None:
        super().__init__(helper=Path("/fake/bin"), popen=lambda command: FakeProc(""))
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


def last_json(out: str) -> dict[str, object]:
    parsed = json.loads(out.strip().splitlines()[-1])
    assert isinstance(parsed, dict)
    return parsed


def deps_for(
    hands: helper.UiHelper, *, transcripts: list[str], respond: engine.Responder
) -> control_exec.ControlDeps:
    return control_exec.ControlDeps(
        transcripts=lambda api_key, opts: transcripts,
        responder=lambda api_key, opts: respond,
        helper=lambda: hands,
    )
