"""Tests for the shared choice/listing helpers in `aai_cli.core.choices`."""

from __future__ import annotations

from aai_cli.core import choices


def test_complete_prefix_filters_by_prefix():
    assert choices.complete_prefix(["alpha", "alto", "beta"], "al") == ["alpha", "alto"]


def test_render_grouped_headers_indentation_and_separator():
    rendered = choices.render_grouped([("English", ["jane", "michael"]), ("Italian", ["giovanni"])])
    # A blank line separates groups; each name is indented two spaces under its header.
    assert rendered == "English:\n  jane\n  michael\n\nItalian:\n  giovanni"


def test_render_grouped_skips_empty_groups():
    # A label with no names contributes nothing — no dangling "French:" header and
    # no extra blank-line separator around it.
    rendered = choices.render_grouped(
        [("English", ["jane"]), ("French", []), ("Italian", ["luca"])]
    )
    assert rendered == "English:\n  jane\n\nItalian:\n  luca"
    assert "French" not in rendered


def test_render_grouped_empty_input_is_empty_string():
    assert choices.render_grouped([]) == ""
