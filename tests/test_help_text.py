from aai_cli.help_text import examples_epilog


def test_examples_epilog_has_header_and_entries():
    epi = examples_epilog([("Do a thing", "aai do --thing")])
    assert "[bold]Examples[/bold]" in epi
    assert "[dim]Do a thing[/dim]" in epi
    assert "$ aai do --thing" in epi


def test_examples_epilog_blank_line_separates_entries():
    # rich_markup_mode="rich" reflows single newlines; blank lines keep each
    # entry on its own row.
    epi = examples_epilog([("First", "aai a"), ("Second", "aai b")])
    assert "\n\n" in epi
    assert epi.count("\n\n") >= 3  # header + 2 descs + 2 cmds, joined by blanks


def test_examples_epilog_escapes_markup_in_commands():
    # Brackets in example commands (jq filters, arrays) must not be parsed as
    # rich markup tags.  rich.markup.escape escapes [word] patterns that Rich
    # would otherwise parse as markup tags (e.g. [key], [bold], [/bold]).
    epi = examples_epilog([("Filter JSON", "aai transcribe x -o json | jq '.[key]'")])
    assert "jq '.\\[key]'" in epi
