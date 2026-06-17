"""Concurrency tests for the two code paths that are genuinely contended at runtime.

Two real concurrency surfaces exist in the CLI, and neither was directly exercised:

1. ``core.config`` persists ``config.toml`` with a temp-file + atomic ``os.replace``
   (`config._dump`), the guard that lets concurrent CLI invocations and readers never
   observe a truncated file. These tests pin that atomicity guarantee under real thread
   contention, and document the flip side it deliberately does *not* solve (lost updates,
   since there is no cross-process lock).
2. ``streaming.StreamSession.on_turn`` runs on the SDK reader thread, and the
   ``--system-audio`` path drives two of those threads at once (`session._drive`). The
   turn write is serialized by ``_callback_lock`` so two sources can't interleave a
   partial line into the saved transcript. These tests pin that mutual exclusion.
"""

from __future__ import annotations

import threading
import types
from concurrent.futures import ThreadPoolExecutor

from aai_cli.core import config

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


def test_concurrent_read_modify_write_drops_an_interleaved_update(tmp_config):
    # Documents a known limitation, not a bug fixed here: config.py has no cross-process
    # write lock (a filelock would add a dependency; fcntl is POSIX-only), so two
    # `assembly` processes that each load -> mutate -> dump concurrently lose one update —
    # last writer wins. Atomic os.replace prevents *corruption*, never *lost updates*.
    # If config.py ever grows a write lock, this assertion is the one that should flip.
    config.set_profile_env("seed", "sandbox000")
    proc_a = config._load()  # both "processes" observe the same starting config
    proc_b = config._load()

    proc_a.profiles["alpha"] = config.Profile(env="sandbox111")
    config._dump(proc_a)  # process A commits its new profile

    proc_b.profiles["beta"] = config.Profile(env="sandbox222")
    config._dump(proc_b)  # process B, unaware of A's write, clobbers it

    names = set(config.list_profiles())
    assert "beta" in names  # the last writer's update survives
    assert "alpha" not in names  # A's interleaved update was silently lost
    assert "seed" in names  # present in both snapshots, so it persists either way


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
