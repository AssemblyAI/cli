"""Golden snapshots pinning the human and JSON renders of representative errors.

These lock the error surface we otherwise only spot-check with substring
asserts, including the ``Suggestion:`` line. A diff means the visible CLI
changed; if intentional, refresh with::

    uv run pytest tests/test_snapshots_errors.py --snapshot-update
"""

from __future__ import annotations

import pytest

from aai_cli import output
from aai_cli.errors import APIError, CLIError, NotAuthenticated, UsageError, auth_failure
from tests._snapshot_surface import normalize

pytestmark = pytest.mark.usefixtures("fixed_render_size")

# Representative errors: each auth/usage/api shape, plus one with no suggestion so
# the golden proves no stray "Suggestion:" line is emitted.
ERROR_CASES = {
    "api_no_suggestion": APIError("Transcription request failed: boom."),
    "not_authenticated": NotAuthenticated(),
    "plain_no_suggestion": CLIError("Something went wrong."),
    "rejected_key": auth_failure(),
    "usage_with_suggestion": UsageError(
        "Unknown voice 'nope'.",
        suggestion="Run 'assembly agent --list-voices' to see the options.",
    ),
}


@pytest.mark.parametrize("case", sorted(ERROR_CASES), ids=lambda c: c)
def test_error_human_render_matches_snapshot(case, capsys, snapshot):
    output.emit_error(ERROR_CASES[case], json_mode=False)
    captured = capsys.readouterr()
    assert captured.out == ""  # errors never touch stdout (keeps pipelines clean)
    assert normalize(captured.err) == snapshot


@pytest.mark.parametrize("case", sorted(ERROR_CASES), ids=lambda c: c)
def test_error_json_render_matches_snapshot(case, capsys, snapshot):
    output.emit_error(ERROR_CASES[case], json_mode=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == snapshot  # exact JSON line, including the suggestion key
