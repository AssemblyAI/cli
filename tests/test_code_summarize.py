"""Tests for the shared tool-activity summarizers (`aai_cli.code_agent.summarize`).

These keep the coding-agent transcript scannable: a tool call shows its identifying arg
(not the whole file being written), and tool output is previewed with a hidden-line tail.
"""

from __future__ import annotations

from aai_cli.code_agent.summarize import (
    describe_args,
    full_args,
    summarize_call,
    summarize_result,
)


def test_describe_args_prefers_identity_arg_and_elides_bulk() -> None:
    # write_file's content is the bulk we must NOT inline — only the path identifies the call.
    body = "\n".join(f"line {i}" for i in range(50))
    assert describe_args({"file_path": "app.py", "content": body}) == "app.py"
    # A shell command is the identity arg for execute.
    assert describe_args({"command": "pip install flask"}) == "pip install flask"


def test_describe_args_clips_long_identity_value() -> None:
    out = describe_args({"command": "echo " + "x" * 200})
    assert out.endswith("…")
    assert len(out) == 60  # exact: clipped to the per-arg budget, ellipsis included


def test_describe_args_without_identity_shows_capped_key_values() -> None:
    out = describe_args({"a": 1, "b": 2, "c": 3, "d": 4})
    # Only the first few args render, then an ellipsis marks the elided remainder.
    assert out.startswith("a=1, b=2, c=3")
    assert out.endswith(", …")
    assert "d=4" not in out


def test_describe_args_collapses_newlines_in_values() -> None:
    # A newline-bearing value must not break the one-line transcript entry.
    assert "\n" not in describe_args({"x": "a\nb\nc"})


def test_summarize_call_wraps_args_in_tool_name() -> None:
    assert (
        summarize_call("write_file", {"file_path": "app.py", "content": "x"})
        == "write_file(app.py)"
    )


def test_summarize_result_previews_and_counts_hidden_lines() -> None:
    out = summarize_result("\n".join(f"line {i}" for i in range(20)))
    assert "line 0" in out and "line 3" in out
    assert "line 4" not in out  # only the first few lines are kept
    assert "+16 more lines" in out  # the rest are counted, not dropped silently


def test_summarize_result_shows_short_output_in_full() -> None:
    assert summarize_result("done\n") == "done"  # no tail when nothing is hidden
    assert summarize_result("   ") == ""  # whitespace-only collapses to empty


def test_full_args_shows_every_arg_whole_with_newlines() -> None:
    # The expanded view keeps content (and its newlines) that describe_args elides.
    out = full_args({"file_path": "app.py", "content": "a\nb\nc"})
    assert "file_path=app.py" in out
    assert "content=a\nb\nc" in out  # full value, newlines preserved


def test_full_args_caps_a_huge_value_with_char_count() -> None:
    out = full_args({"content": "z" * 1500})  # over the 1000-char expanded budget
    assert "+500 more chars" in out  # exact: 1500 minus the 1000 budget
    assert out.startswith("content=" + "z" * 1000)


def test_full_args_shows_a_value_at_the_budget_whole() -> None:
    # Boundary: exactly the budget is shown whole (guards the cap's `>` against a `>=` slip).
    out = full_args({"content": "z" * 1000})
    assert "more chars" not in out
    assert out == "content=" + "z" * 1000


def test_summarize_result_counts_a_single_hidden_line() -> None:
    # Boundary: exactly one line over the preview budget still gets a tail (guards the
    # `hidden_lines > 0` threshold against a `> 1` slip that would silently drop it).
    out = summarize_result("\n".join(f"line {i}" for i in range(5)))  # 4 shown, 1 hidden
    assert out.endswith("(+1 more lines)")


def test_summarize_result_clips_one_huge_line_with_char_count() -> None:
    out = summarize_result("z" * 500)  # a single line longer than the char budget
    assert "+200 more chars" in out  # exact: 500 minus the 300-char budget = 200 hidden
    assert out.startswith("z" * 300)
