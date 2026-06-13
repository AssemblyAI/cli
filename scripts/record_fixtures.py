#!/usr/bin/env python
"""Record real AssemblyAI API responses as scrubbed JSON fixtures for replay tests.

This is a *manual* tool, deliberately outside the test suite and the gate: it reaches
the real network. It drives the same `client.* / llm.* / ams.*` functions the CLI uses,
then serializes each result to ``tests/fixtures/api/`` with every credential scrubbed,
so the committed fixtures carry no secrets (the gate's gitleaks scan would catch any
that slipped through).

Usage::

    ASSEMBLYAI_API_KEY=<key> uv run python scripts/record_fixtures.py

The API key is read from the environment; the AMS session (JWT) is read from the OS
keyring of whoever ran ``assembly login`` (profile ``default``). Neither is ever written to
a fixture. Re-run it to refresh the fixtures after an API shape change.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import assemblyai as aai

from aai_cli.auth import ams
from aai_cli.core import client, config, environments, llm
from aai_cli.core.errors import CLIError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "api"
PROFILE = "default"

# Stable placeholders substituted for real secrets/identifiers on the way out. Replay
# tests assert against these, so they are part of the fixture contract.
FAKE_ACCOUNT_ID = 12345
FAKE_EMAIL = "user@example.com"
REDACTED = "REDACTED"
UPLOAD_PREFIX = "https://cdn.assemblyai.com/upload/"
# API responses are shallow; a deeper structure means malformed/hostile input, so cap
# the recursion rather than risk a stack overflow on a pathologically nested payload.
_MAX_SCRUB_DEPTH = 100


def _scrub_str(value: str, secret_set: set[str]) -> str:
    """Redact a string value: a known secret becomes ``REDACTED``; an account upload
    URL keeps its prefix but drops the high-entropy hash so no private audio leaks."""
    if value in secret_set:
        return REDACTED
    if value.startswith(UPLOAD_PREFIX):
        return UPLOAD_PREFIX + REDACTED
    return value


def _build_scrubber(secrets: list[str]) -> Callable[[Any], Any]:
    """A recursive scrubber that redacts known secret strings and identifying keys.

    ``secrets`` are exact string values (the API key, session JWT/token) replaced with
    ``REDACTED`` wherever they appear as a value. Keys named ``email``/``account_id``
    are replaced with stable fakes regardless of value, so the committed fixtures are
    inert but still shaped exactly like the real responses.
    """
    secret_set = {s for s in secrets if s}

    def scrub(obj: Any, depth: int = 0) -> Any:
        if depth > _MAX_SCRUB_DEPTH:
            raise CLIError(
                f"Fixture nesting exceeded {_MAX_SCRUB_DEPTH} levels; refusing to scrub."
            )
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for key, value in obj.items():
                if key == "email":
                    out[key] = FAKE_EMAIL
                elif key == "account_id":
                    out[key] = FAKE_ACCOUNT_ID
                else:
                    out[key] = scrub(value, depth + 1)
            return out
        if isinstance(obj, list):
            return [scrub(item, depth + 1) for item in obj]
        if isinstance(obj, str):
            return _scrub_str(obj, secret_set)
        return obj

    return scrub


def _out(message: str) -> None:
    sys.stdout.write(message + "\n")


def _err(message: str) -> None:
    sys.stderr.write(message + "\n")


def _write(name: str, payload: object, scrub: Callable[[Any], Any]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps(scrub(payload), indent=2) + "\n")
    _out(f"  wrote {path.relative_to(Path.cwd())}")


def _transcript_payload(transcript: aai.Transcript) -> dict[str, object]:
    """The raw API ``json_response`` for a transcript — what get_by_id would parse."""
    payload = client.transcript_json_payload(transcript)
    return dict(payload)


def main() -> int:
    environments.set_active(environments.get(environments.DEFAULT_ENV))

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        _err("ASSEMBLYAI_API_KEY is not set.")
        return 1

    session = config.get_session(PROFILE)
    account_id = config.get_account_id(PROFILE)
    if session is None or account_id is None:
        _err(f"No AMS session for profile {PROFILE!r}; run 'assembly login' first.")
        return 1
    jwt = session["jwt"]
    scrub = _build_scrubber([api_key, jwt, session.get("token", "")])

    # (name, thunk) — each runs independently so one failure (e.g. an LLM entitlement
    # block) doesn't lose the others. The sample transcript's id feeds the get fixture.
    _out(f"Recording fixtures into {FIXTURE_DIR}")

    sample = client.transcribe(api_key, client.SAMPLE_AUDIO_URL, config=aai.TranscriptionConfig())
    _write("transcribe_sample", _transcript_payload(sample), scrub)
    sample_id = sample.id
    if sample_id is None:  # a completed transcript always has an id; guard for the type checker
        raise CLIError("Transcribe returned no transcript id.")

    _write("transcripts_list", client.list_transcripts(api_key, limit=10), scrub)

    got = client.get_transcript(api_key, sample_id)
    _write("transcript_get", _transcript_payload(got), scrub)

    today = datetime.now(UTC).date()
    start = datetime(today.year, today.month, 1, tzinfo=UTC).isoformat()
    end = datetime(today.year, today.month, today.day, tzinfo=UTC).isoformat()

    jobs: list[tuple[str, Callable[[], object]]] = [
        ("account_balance", lambda: ams.get_balance(jwt)),
        ("account_usage", lambda: ams.get_usage(jwt, start, end, "day")),
        ("account_limits", lambda: ams.get_rate_limits(account_id, jwt)),
        (
            "llm_complete",
            lambda: llm.complete(
                api_key,
                model=llm.DEFAULT_MODEL,
                messages=llm.build_messages("Reply with exactly one word: PONG"),
                max_tokens=16,
            ).model_dump(mode="json"),
        ),
    ]
    for name, thunk in jobs:
        try:
            _write(name, thunk(), scrub)
        except CLIError as exc:
            # The client/ams/llm wrappers funnel every expected network failure into a
            # CLIError, so a blocked LLM entitlement skips just that fixture.
            _err(f"  SKIP {name}: {exc}")

    _out("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
