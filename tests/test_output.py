import json
from typing import cast

from aai_cli import output
from aai_cli.errors import CLIError


def test_resolve_json_true_only_when_explicit():
    # JSON is opt-in: the flag is the single source of truth.
    assert output.resolve_json(explicit=True) is True


def test_resolve_json_false_when_not_explicit_even_off_tty(monkeypatch):
    # Human text is the default everywhere — piped, in CI, or under an agent — so a
    # plain-text pipeline (`assembly transcribe x | grep word`) keeps getting text, not JSON.
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: False)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLAUDECODE", "1")
    assert output.resolve_json(explicit=False) is False


def test_resolve_json_false_for_human(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    assert output.resolve_json(explicit=False) is False


def test_is_agentic_true_for_agent_env_var_even_with_tty(monkeypatch):
    # Interactivity detection (used to suppress the spinner) still reports "no human"
    # when a CI/agent env var is set — independent of resolve_json, which stays text.
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert output.is_agentic() is True


def test_is_agentic_false_for_plain_interactive_tty(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    assert output.is_agentic() is False


def test_is_agentic_true_when_stdout_not_a_tty(monkeypatch):
    # Piped/redirected stdout means no interactive human, so the spinner is
    # suppressed even with no agent env var set -- guards the `not a tty -> True`
    # early return (without it, this path would fall through to the env-var check).
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: False)
    for var in output._AGENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert output.is_agentic() is True


def test_redact_secret_preserves_only_short_edges():
    assert output.redact_secret("sk_1234567890") == "sk_…7890"
    assert output.redact_secret("12345678") == "123…5678"
    assert output.redact_secret("short") == "***"


def test_emit_json_serializes(capsys):
    output.emit({"a": 1}, lambda d: "human", json_mode=True)
    out = capsys.readouterr().out
    assert json.loads(out) == {"a": 1}


def test_emit_human_uses_renderer(capsys):
    output.emit({"a": 1}, lambda d: f"value={d['a']}", json_mode=False)
    assert "value=1" in capsys.readouterr().out


def test_emit_error_escapes_markup(capsys):
    import types

    err = types.SimpleNamespace(
        message="bad [tag] here", suggestion=None, to_dict=lambda: {"error": {}}
    )
    output.emit_error(cast(CLIError, err), json_mode=False)
    captured = capsys.readouterr()
    assert "[tag]" in captured.err  # error goes to stderr, not stripped as markup
    assert captured.out == ""  # stdout stays clean for pipelines


def test_emit_error_json_goes_to_stderr(capsys):
    import types

    err = types.SimpleNamespace(message="boom", to_dict=lambda: {"error": {"message": "boom"}})
    output.emit_error(cast(CLIError, err), json_mode=True)
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": {"message": "boom"}}
    assert captured.out == ""


def test_emit_error_renders_suggestion_line(capsys):
    import types

    err = types.SimpleNamespace(
        message="bad thing",
        suggestion="try this instead",
        to_dict=lambda: {"error": {}},
    )
    output.emit_error(cast(CLIError, err), json_mode=False)
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "bad thing" in captured.err
    assert "Suggestion:" in captured.err
    assert "try this instead" in captured.err
    assert captured.out == ""


def test_emit_error_no_suggestion_line_when_absent(capsys):
    import types

    err = types.SimpleNamespace(message="bad thing", suggestion=None, to_dict=lambda: {"error": {}})
    output.emit_error(cast(CLIError, err), json_mode=False)
    captured = capsys.readouterr()
    assert "Suggestion:" not in captured.err


def test_affordance_helpers_carry_their_symbol():
    from aai_cli import theme

    assert theme.SYMBOL_SUCCESS in output.success("done")
    assert theme.SYMBOL_WARN in output.warn("careful")
    assert theme.SYMBOL_HINT in output.hint("do this next")
    # heading has no glyph, just the brand style wrapper
    assert "aai.heading" in output.heading("Section")


def test_affordance_helpers_use_resolvable_styles(capsys):
    from aai_cli import theme

    # Rendering through the themed console proves the markup parses and the
    # aai.* style names resolve (a bad name would raise MissingStyle).
    console = theme.make_console(force_terminal=True, color_system="truecolor")
    for line in (
        output.success("ok"),
        output.warn("hmm"),
        output.hint("next"),
        output.heading("H"),
    ):
        console.print(line)
    out = capsys.readouterr().out
    assert theme.SYMBOL_SUCCESS in out
    assert theme.SYMBOL_HINT in out
    assert "\x1b[" in out  # themed -> ANSI present


def test_print_code_plain_when_piped(monkeypatch, capsys):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: False)
    output.print_code("import os\nprint(os.getcwd())\n")
    out = capsys.readouterr().out
    assert "import os" in out
    assert "\x1b[" not in out  # no ANSI for pipes/redirects -> runnable when saved


def test_print_code_highlights_for_interactive_human(monkeypatch, capsys):
    from aai_cli import theme

    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr(
        output, "console", theme.make_console(force_terminal=True, color_system="truecolor")
    )
    output.print_code("import os\n")
    out = capsys.readouterr().out
    assert "import" in out
    assert "\x1b[" in out  # syntax-highlighted -> ANSI present


def test_data_table_is_minimal_and_themed():
    from rich import box

    table = output.data_table("id", "status")
    # One shared, quiet look: a header-rule box (no heavy outer border) and the
    # brand heading style — so every listing command renders identically.
    assert table.box is box.SIMPLE_HEAD
    assert table.header_style == "aai.heading"
    assert table.pad_edge is False  # no leading/trailing pad column -> flush-left listing
    assert [str(col.header) for col in table.columns] == ["id", "status"]


def test_detail_table_is_borderless_label_value_grid():
    table = output.detail_table()
    # A grid (no box) with a muted label column, shared by whoami / sessions get.
    assert table.box is None
    assert len(table.columns) == 2
    assert table.columns[0].style == "aai.muted"
    # padding=(0, 3): no vertical pad, 3 cols of horizontal gap between label/value.
    assert table.padding == (0, 3, 0, 3)


def test_muted_returns_dim_text_line():
    from rich.text import Text

    line = output.muted("Hidden: 1 window.")
    assert isinstance(line, Text)
    assert line.plain == "Hidden: 1 window."
    assert line.style == "aai.muted"


def test_stack_returns_lone_item_bare():
    # A single surviving renderable isn't wrapped in a redundant Group.
    table = output.data_table("id")
    assert output.stack(table, None) is table


def test_stack_drops_none_and_groups_the_rest():
    from rich.console import Group

    first = output.muted("a")
    second = output.muted("b")
    result = output.stack(first, None, second)
    assert isinstance(result, Group)
    assert list(result.renderables) == [first, second]


def test_emit_ndjson_writes_one_flushed_line(monkeypatch):
    import sys

    class _RecordingStdout:
        def __init__(self):
            self.text = ""
            self.flushed = 0

        def write(self, s):
            self.text += s
            return len(s)

        def flush(self):
            self.flushed += 1

    rec = _RecordingStdout()
    monkeypatch.setattr(sys, "stdout", rec)
    output.emit_ndjson({"a": 1})
    # One newline-terminated JSON record, explicitly flushed so live pipelines see it.
    assert rec.text == '{"a": 1}\n'
    assert rec.flushed >= 1


def test_status_is_noop_in_json_mode(monkeypatch):
    # JSON mode must never enter the spinner (it would render to stderr unnecessarily).
    monkeypatch.setattr(output, "is_agentic", lambda: False)
    entered = {"status": False}
    monkeypatch.setattr(
        output.error_console, "status", lambda *a, **k: entered.__setitem__("status", True)
    )
    with output.status("Working…", json_mode=True):
        pass
    assert entered["status"] is False


def test_status_is_noop_when_agentic(monkeypatch):
    monkeypatch.setattr(output, "is_agentic", lambda: True)
    entered = {"status": False}
    monkeypatch.setattr(
        output.error_console, "status", lambda *a, **k: entered.__setitem__("status", True)
    )
    with output.status("Working…", json_mode=False):
        pass
    assert entered["status"] is False


def test_status_enters_spinner_when_not_json_and_not_quiet(monkeypatch):
    # The default quiet=False must let the spinner run (pins the parameter default).
    import contextlib

    monkeypatch.setattr(output, "is_agentic", lambda: False)
    entered = {"status": False}

    @contextlib.contextmanager
    def fake_status(message):
        entered["status"] = True
        yield

    monkeypatch.setattr(output.error_console, "status", fake_status)
    with output.status("Working…", json_mode=False):
        pass
    assert entered["status"] is True


def test_status_is_noop_when_quiet(monkeypatch):
    # --quiet suppresses the spinner even for an interactive human.
    monkeypatch.setattr(output, "is_agentic", lambda: False)
    entered = {"status": False}
    monkeypatch.setattr(
        output.error_console, "status", lambda *a, **k: entered.__setitem__("status", True)
    )
    with output.status("Working…", json_mode=False, quiet=True):
        pass
    assert entered["status"] is False


def test_emit_warning_human_writes_yellow_line_to_stderr(capsys):
    from aai_cli import theme

    output.emit_warning("env mismatch", json_mode=False)
    captured = capsys.readouterr()
    assert "env mismatch" in captured.err
    assert theme.SYMBOL_WARN in captured.err  # the ! affordance glyph
    assert captured.out == ""  # stdout stays clean for pipelines


def test_emit_warning_json_goes_to_stderr_as_warning_object(capsys):
    output.emit_warning("env mismatch", json_mode=True)
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"warning": "env mismatch"}
    assert captured.out == ""


def test_status_shows_spinner_for_interactive_human(monkeypatch):
    monkeypatch.setattr(output, "is_agentic", lambda: False)
    calls = []
    with output.error_console.capture():
        with output.status("Transcribing…", json_mode=False):
            calls.append("inside")
    # The body ran inside the spinner context and the spinner targeted stderr.
    assert calls == ["inside"]
