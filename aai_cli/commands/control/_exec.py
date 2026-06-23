"""Run logic for `assembly control`: a gh-style options/run split.

The command module parses argv into a :class:`ControlOptions` and hands it to
:func:`run_control`. The three external legs — mic Streaming STT, the LLM
Gateway, and the native UI helper — are bundled in :class:`ControlDeps` with
real-implementation defaults, so a test drives the whole session by passing
fakes to :func:`_run_control` with no microphone, network, subprocess, or macOS.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from aai_cli.app.context import AppState
from aai_cli.control import bridge, engine, prompt
from aai_cli.control import listen as listen_mod
from aai_cli.control.helper import UiHelper
from aai_cli.control.render import ControlRenderer
from aai_cli.core import signals


@dataclass(frozen=True)
class ControlOptions:
    """Every `assembly control` flag as plain data."""

    device: int | None
    sample_rate: int | None
    model: str
    max_tokens: int
    max_steps: int
    dry_run: bool


def _default_transcripts(api_key: str, opts: ControlOptions) -> Iterable[str]:
    """Real mic→utterance leg."""
    return listen_mod.listen(api_key, device=opts.device, sample_rate=opts.sample_rate)


def _default_responder(api_key: str, opts: ControlOptions) -> engine.Responder:
    """Real LLM-Gateway leg."""
    return bridge.build_responder(api_key, model=opts.model, max_tokens=opts.max_tokens)


def _default_helper() -> UiHelper:
    """Real native-helper leg (compiles + spawns the Swift binary on first action)."""
    return UiHelper()


@dataclass(frozen=True)
class ControlDeps:
    """The three external legs, injectable so the session is exercised with fakes."""

    transcripts: Callable[[str, ControlOptions], Iterable[str]] = _default_transcripts
    responder: Callable[[str, ControlOptions], engine.Responder] = _default_responder
    helper: Callable[[], UiHelper] = _default_helper


_DEFAULT_DEPS = ControlDeps()


def _run_control(
    opts: ControlOptions,
    state: AppState,
    *,
    json_mode: bool,
    deps: ControlDeps,
) -> None:
    """Drive one hands-free control session with the given dependencies."""
    # Build the native helper first: on a non-macOS host this fails fast with the
    # "macOS only" message, before the user is ever asked to authenticate. Once it
    # exists, everything else runs under try/finally so the child is always closed.
    hands = deps.helper()
    try:
        api_key = state.resolve_api_key()
        respond = deps.responder(api_key, opts)
        transcripts = deps.transcripts(api_key, opts)
        renderer = ControlRenderer(json_mode=json_mode)
        with signals.terminate_as_interrupt():
            engine.run_session(
                transcripts,
                system=prompt.system_prompt(),
                respond=respond,
                execute=hands.execute,
                renderer=renderer,
                max_steps=opts.max_steps,
                allow_mutate=not opts.dry_run,
            )
    finally:
        hands.close()


def run_control(opts: ControlOptions, state: AppState, /, *, json_mode: bool) -> None:
    """Execute one `assembly control` invocation from already-parsed flags."""
    _run_control(opts, state, json_mode=json_mode, deps=_DEFAULT_DEPS)
