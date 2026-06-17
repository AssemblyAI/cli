"""Concurrency tests for the two code paths that are genuinely contended at runtime.

Two real concurrency surfaces exist in the CLI, and neither was directly exercised:

1. ``core.config`` persists ``config.toml`` with a temp-file + atomic ``os.replace``
   (`config._dump`) so a reader never observes a truncated file, and serializes its
   read-modify-write under a cross-process ``filelock`` (`config._write_lock`) so two
   concurrent ``assembly`` processes can't lose each other's updates. These tests pin
   both: the at-rest atomicity under thread contention, and that distinct concurrent
   updates all survive (no last-writer-wins clobber).
2. ``streaming.StreamSession.on_turn`` runs on the SDK reader thread, and the
   ``--system-audio`` path drives two of those threads at once (`session._drive`). The
   turn write is serialized by ``_callback_lock`` so two sources can't interleave a
   partial line into the saved transcript. These tests pin that mutual exclusion.
"""

from __future__ import annotations

import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from aai_cli.core import config, config_lock

# --- config.toml: the Windows os.replace sharing-window retry -----------------------


def test_retry_on_sharing_violation_returns_without_retrying_on_success(monkeypatch):
    # The common case: the op succeeds first try, so no backoff sleep happens.
    sleeps: list[float] = []
    monkeypatch.setattr(config, "time", types.SimpleNamespace(sleep=sleeps.append))
    calls = []

    def op():
        calls.append(1)
        return "ok"

    assert config._retry_on_sharing_violation(op) == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retry_on_sharing_violation_rides_out_transient_permission_errors(monkeypatch):
    # Two transient PermissionErrors (Windows' replace window) then success: the helper
    # backs off between attempts and ultimately returns the value, never raising.
    sleeps: list[float] = []
    monkeypatch.setattr(config, "time", types.SimpleNamespace(sleep=sleeps.append))
    calls = []

    def op():
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("file is being replaced")
        return "ok"

    assert config._retry_on_sharing_violation(op) == "ok"
    assert len(calls) == 3  # two failures, then the success
    assert sleeps == [config._SHARING_BACKOFF, config._SHARING_BACKOFF]  # one per retry


def test_retry_on_sharing_violation_reraises_a_persistent_permission_error(monkeypatch):
    # A genuine, persistent permission problem is not a transient sharing race: after the
    # full budget the last attempt's error propagates rather than looping forever.
    sleeps: list[float] = []
    monkeypatch.setattr(config, "time", types.SimpleNamespace(sleep=sleeps.append))
    calls = []

    def op():
        calls.append(1)
        raise PermissionError("denied")

    with pytest.raises(PermissionError, match="denied"):
        config._retry_on_sharing_violation(op)
    # Exactly the full budget of attempts (loop retries + one final attempt), no more.
    assert len(calls) == config._SHARING_RETRIES


# --- config.toml: atomic writes vs. lost updates -----------------------------------


def test_config_concurrent_writers_always_leave_a_valid_file(tmp_config):
    # Many threads rewriting config.toml at once, plus a reader hammering it throughout:
    # the temp-file + atomic os.replace in _dump means no writer and no reader ever sees
    # a truncated/half-written file (which would surface as an invalid_config CLIError),
    # the surviving value is exactly one writer's, and no .config-*.toml.tmp is left behind.
    workers = 24
    barrier = threading.Barrier(workers + 1)  # writers + the reader, released together
    stop = threading.Event()

    def writer(i: int) -> None:
        barrier.wait()
        config.set_profile_env("default", f"sandbox{i:03d}")

    def reader() -> None:
        barrier.wait()
        while not stop.is_set():
            config.get_profile_env("default")  # must never raise on a partial file

    # future.result() re-raises any worker error in the main thread, so a truncated-file
    # read (an invalid_config CLIError) fails the test cleanly instead of being swallowed.
    with ThreadPoolExecutor(max_workers=workers + 1) as pool:
        read_future = pool.submit(reader)
        write_futures = [pool.submit(writer, i) for i in range(workers)]
        for f in write_futures:
            f.result()
        stop.set()
        read_future.result()

    assert config.get_profile_env("default") in {f"sandbox{i:03d}" for i in range(workers)}
    assert sorted(p.name for p in tmp_config.iterdir()) == ["config.toml"]  # no temp leftover


def test_concurrent_writers_do_not_lose_distinct_updates(tmp_config):
    # The cross-process write lock makes the read-modify-write atomic, so many threads
    # each adding a DISTINCT profile through the public API all survive. Without the lock
    # the interleaved RMW would drop some (two writers _load the same config; the second
    # _dump clobbers the first's new profile) — this is the lost-update race the lock closes.
    workers = 16
    barrier = threading.Barrier(workers)  # release all writers at once for max contention

    def add(i: int) -> None:
        barrier.wait()
        config.set_profile_env(f"p{i:02d}", f"sandbox{i:03d}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(add, i) for i in range(workers)]:
            f.result()

    # Every concurrent update is present with its own value — none clobbered.
    assert config.list_profiles() == {f"p{i:02d}": f"sandbox{i:03d}" for i in range(workers)}


def test_write_lock_targets_the_config_dir_lock_file(tmp_config):
    # The lock guards config.toml via a sibling lock file in the same dir.
    assert config_lock.write_lock().lock_file == str(tmp_config / "config.toml.lock")


def test_write_lock_rebuilds_when_config_dir_changes(monkeypatch, tmp_path):
    # The instance is cached per path, but a changed config dir (the suite repoints it per
    # test) must yield a fresh lock pointed at the new dir — not the stale one.
    first = config_lock.write_lock()
    moved = tmp_path / "moved"
    moved.mkdir()
    monkeypatch.setattr(config, "config_dir", lambda: moved)
    second = config_lock.write_lock()
    assert second is not first
    assert second.lock_file == str(moved / "config.toml.lock")


def test_update_holds_the_write_lock_during_the_dump(tmp_config, monkeypatch):
    # The mutate -> dump runs inside the lock: while _dump executes, the lock is held.
    # (Drop the `with locked()` and is_locked would be False here.)
    seen: dict[str, bool] = {}
    real_dump = config._dump

    def spy_dump(cfg):
        seen["locked"] = config_lock.write_lock().is_locked
        return real_dump(cfg)

    monkeypatch.setattr(config, "_dump", spy_dump)
    config.set_profile_env("default", "sandbox000")
    assert seen["locked"] is True


# --- streaming: on_turn serialization under _callback_lock -------------------------


def _turn(text: str):
    # No source/speaker label, so _finalized_turn_line writes the bare text as its line.
    return types.SimpleNamespace(transcript=text, end_of_turn=True, speaker_label=None)


def _stream_session(out_path):
    import io

    from aai_cli.streaming.render import StreamRenderer
    from aai_cli.streaming.session import StreamSession

    return StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
        save_transcript=out_path,
    )


def test_on_turn_holds_callback_lock_across_the_save(tmp_path):
    # on_turn must run its render + save + meta critical section under _callback_lock, so a
    # second SDK reader thread (--system-audio runs two) can't interleave into it. Proven
    # deterministically: pin a worker mid-save (inside write_turn) and assert no other
    # thread can take the lock while it's there. Drop the `with self._callback_lock` and
    # the lock is free during the save -> acquire(blocking=False) would succeed and fail this.
    from aai_cli.streaming.transcript import TranscriptWriter

    out = tmp_path / "transcript.txt"
    session = _stream_session(out)
    entered = threading.Event()
    release = threading.Event()

    class _BlockingWriter(TranscriptWriter):
        # A real writer whose write_turn blocks, so the test can hold on_turn's critical
        # section open and probe the lock. It opens the real handle (closed below).
        def __init__(self, path) -> None:
            super().__init__(path)
            self.lines: list[str] = []

        def write_turn(self, line: str) -> None:
            self.lines.append(line)
            entered.set()  # we're now inside _save_line, holding _callback_lock
            assert release.wait(timeout=5)  # hold the lock open until the test releases us

    writer = _BlockingWriter(out)
    session._transcript_writer = writer
    worker = threading.Thread(target=lambda: session.on_turn(_turn("first")))
    worker.start()

    assert entered.wait(timeout=5)  # worker reached the critical section
    # While the worker is inside it, no other thread can acquire _callback_lock.
    assert session._callback_lock.acquire(blocking=False) is False

    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert writer.lines == ["first"]
    writer.close()


def test_concurrent_turns_record_every_line_without_interleaving(tmp_path):
    # Two source threads firing finalized turns at once (the --system-audio shape) must
    # each land in the saved transcript exactly once and intact: _callback_lock serializes
    # the writes so no line is lost or interleaved with another's characters.
    from aai_cli.streaming.transcript import TranscriptWriter

    out = tmp_path / "transcript.txt"
    session = _stream_session(out)
    session._transcript_writer = TranscriptWriter(out)

    threads_count, per_thread = 2, 50
    barrier = threading.Barrier(threads_count)  # start the bursts simultaneously

    def worker(tid: int) -> None:
        barrier.wait()
        for n in range(per_thread):
            session.on_turn(_turn(f"t{tid}-{n}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    session._transcript_writer.close()

    lines = out.read_text(encoding="utf-8").splitlines()
    expected = {f"t{i}-{n}" for i in range(threads_count) for n in range(per_thread)}
    assert len(lines) == len(expected)  # every turn written exactly once (none lost)
    assert set(lines) == expected  # none interleaved/corrupted into an unexpected line
