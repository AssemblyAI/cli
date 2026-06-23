"""Low-level reply-runtime primitives for the cascade engine.

The cascade streams each LLM reply on a throwaway producer thread that feeds a
queue the consumer drains under a wall-clock deadline (see ``engine.py``). This
module holds the pieces that machinery is built from — the queue sentinels, the
timeout error, the worker protocol, and the ``concurrent.futures`` executor
detach that keeps an abandoned graph leg from wedging interpreter exit — kept
separate from the orchestration in :class:`~aai_cli.agent_cascade.engine.CascadeSession`
so each file stays focused.

The module name is underscore-prefixed (package-private); ``engine`` imports
these names and aliases them back to its own ``_``-prefixed internals.
"""

from __future__ import annotations

import concurrent.futures.thread as cf_thread
import contextlib
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aai_cli.agent_cascade import brain
from aai_cli.core.errors import CLIError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Wall-clock backstop for one reply turn. The reply is streamed on a throwaway producer
# thread feeding a queue; a stalled gateway can block inside a token read the worker can't
# observe, so the consumer's queue.get is bounded by a monotonic deadline. After this long
# we stop waiting and surface a timeout so the session stays usable. Generous on purpose.
REPLY_TIMEOUT_SECONDS = 60.0  # pragma: no mutate


@dataclass(frozen=True)
class Done:
    """Producer sentinel: the reply stream finished normally."""


@dataclass(frozen=True)
class Failure:
    """Producer sentinel: the reply leg raised a (clean) CLIError."""

    error: CLIError


@dataclass(frozen=True)
class Timeout:
    """Consumer sentinel: the wall-clock deadline elapsed before the next event arrived."""


# What the producer thread puts on the consumer's queue: a speech/tool event from the
# streaming leg, an approval-pause marker (--files write gating), or a terminal sentinel.
type ReplyEvent = brain.SpeechDelta | brain.ToolNotice | brain.ApprovalPause | Done | Failure


def timeout_error() -> CLIError:
    """The backstop error raised when a reply overruns the wall-clock deadline."""
    return CLIError(
        f"the agent took longer than {REPLY_TIMEOUT_SECONDS:.0f}s to respond and was cut off",
        error_type="agent_timeout",
    )


class Worker(Protocol):
    """The slice of a thread the session drives: started already, queryable, joinable."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Whether the reply worker is still running."""

    def join(self) -> None:
        """Block until the reply worker finishes."""


def new_history() -> list[ChatCompletionMessageParam]:
    """Typed empty-history factory (ChatCompletionMessageParam is import-time-only)."""
    return []


def executor_threads() -> set[threading.Thread]:
    """A snapshot of every live ThreadPoolExecutor worker concurrent.futures tracks for its
    interpreter-exit join. Empty if a future Python drops the internal registry."""
    return set(getattr(cf_thread, "_threads_queues", ()))


def detach_executor_threads_since(before: set[threading.Thread]) -> None:
    """Drop executor workers spawned since ``before`` from concurrent.futures' exit-join list,
    so an abandoned (timed-out) graph leg can't wedge process exit.

    ``complete_reply`` runs the deepagents graph, which drives each node through a langchain
    ``ThreadPoolExecutor``. Abandoning a timed-out call leaves that executor's worker blocked on
    the network leg, and concurrent.futures registers an interpreter-exit hook (``_python_exit``)
    that joins *every* executor worker unconditionally — even daemons — by putting a shutdown
    sentinel on its queue and waiting. A worker mid-call never reads that sentinel, so the join
    (and the whole process exit) hangs until the user Ctrl-Cs — the threading-shutdown traceback
    this prevents. The worker was created on our own daemon thread so it inherits ``daemon=True``;
    once it's off this registry neither ``_python_exit`` nor ``threading._shutdown`` waits on it,
    and the orphaned network call dies with the process as a daemon should. Best-effort: a future
    Python that renames the internals simply skips the detach (regressing to the old hang, not
    crashing). The diff is scoped to threads that appeared during the call, so a co-running
    executor elsewhere keeps its normal exit-time join.
    """
    registry = getattr(cf_thread, "_threads_queues", None)
    if registry is None:
        return
    # Mutate under the same lock concurrent.futures holds for the registry, so a concurrent
    # submit (or _python_exit itself) never sees a torn dict.
    with getattr(cf_thread, "_global_shutdown_lock", contextlib.nullcontext()):
        for thread in executor_threads() - before:
            registry.pop(thread, None)


def spawn_thread(target: Callable[[], None]) -> Worker:
    """Start ``target`` on a daemon thread so a reply is generated without blocking
    the STT reader (which must stay free to detect a barge-in)."""
    thread = threading.Thread(target=target, daemon=True)  # pragma: no mutate
    thread.start()
    return thread


def final_tail(buffer: str, held: list[str], *, used_tool: bool) -> str:
    """End-of-stream remainder to flush: joined post-tool narration, else the live pre-tool buffer."""
    return "".join(held) if used_tool else buffer


def approval_deadline(pause: brain.ApprovalPause) -> float | None:
    """The reply deadline across a write-approval pause: ``None`` (clock suspended) while the user
    is deciding on a gated write — a slow y/n keypress must not trip the reply timeout — and a fresh
    finite deadline once answered."""
    return None if pause.active else time.monotonic() + REPLY_TIMEOUT_SECONDS


def is_final_turn(event: object, *, format_turns: bool) -> bool:
    """True for an end-of-turn that's the cue to generate a reply.

    With formatting on, wait for the *formatted* turn (better text for the LLM); with it off the
    server never sets ``turn_is_formatted``, so a bare end-of-turn is the cue — otherwise
    ``--no-format-turns`` would make the agent never reply.
    """
    if not bool(getattr(event, "end_of_turn", False)):
        return False
    return bool(getattr(event, "turn_is_formatted", False)) or not format_turns
