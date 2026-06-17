"""`assembly llm` file- and directory-context tests.

Split out of ``test_llm_command.py`` (which would otherwise exceed the 500-line
gate) — the prompt's file arguments, directory recursion, and the input-source
priority warnings all live here.
"""

import types

from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_api_key("default", "sk_live")


def _payload(content="four"):
    # Mimics the OpenAI SDK response object the command reads via content_of/usage_of.
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    usage = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 3})
    return types.SimpleNamespace(choices=[choice], usage=usage)


def test_llm_reads_file_argument_as_context(monkeypatch, tmp_path):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        seen["transcript_id"] = transcript_id
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    note = tmp_path / "alpha.md"
    note.write_text("bob owns the deploy")
    result = runner.invoke(app, ["llm", "who owns the deploy?", str(note), "--json"])
    assert result.exit_code == 0
    # The file content is injected, under a header naming the file's stem.
    assert "who owns the deploy?" in seen["content"]
    assert "bob owns the deploy" in seen["content"]
    assert "===== alpha =====" in seen["content"]
    assert seen["transcript_id"] is None


def test_llm_concatenates_multiple_files_with_headers_in_order(monkeypatch, tmp_path):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    first = tmp_path / "first.md"
    first.write_text("ship friday")
    second = tmp_path / "second.md"
    second.write_text("freeze monday")
    result = runner.invoke(app, ["llm", "summarize", str(first), str(second), "--json"])
    assert result.exit_code == 0
    content = seen["content"]
    assert "===== first =====" in content
    assert "===== second =====" in content
    assert "ship friday" in content
    assert "freeze monday" in content
    # Both note bodies appear under their own header, in the order passed.
    assert content.index("===== first =====") < content.index("===== second =====")
    assert content.index("ship friday") < content.index("freeze monday")


def test_llm_files_take_priority_over_stdin(monkeypatch, tmp_path):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    note = tmp_path / "note.md"
    note.write_text("from the file")
    result = runner.invoke(
        app, ["llm", "summarize", str(note)], input="from stdin, should be ignored"
    )
    assert result.exit_code == 0
    assert "from the file" in seen["content"]
    assert "from stdin, should be ignored" not in seen["content"]
    assert "Ignoring piped stdin; file arguments take priority." in result.output


def test_llm_missing_file_exits_2_without_network(monkeypatch, tmp_path):
    # A bad path (e.g. an unmatched shell glob passed through literally) is a usage
    # error raised before auth or the gateway, not a crash.
    _auth()
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the gateway")),
    )
    missing = tmp_path / "nope.md"
    result = runner.invoke(app, ["llm", "summarize", str(missing)])
    assert result.exit_code == 2
    assert "Couldn't read" in result.output
    # The clean OS reason (errno's strerror) is shown, not the raw exception repr —
    # so no "[Errno N] …: '/path'" bracket leaks into the message.
    assert "[Errno" not in result.output


def test_llm_directory_argument_recurses_for_md_and_txt(monkeypatch, tmp_path):
    # A directory recurses for .md/.txt files (including nested ones), reads each
    # under its own header in a deterministic (path-sorted) order, and skips
    # non-matching extensions. This is the glob-and-guard the justfile hand-rolled.
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    (tmp_path / "aaa.md").write_text("ship friday")
    (tmp_path / "bbb.txt").write_text("freeze monday")
    (tmp_path / "ignore.json").write_text("not included")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "ccc.md").write_text("nested note")

    result = runner.invoke(app, ["llm", "summarize", str(tmp_path), "--json"])
    assert result.exit_code == 0
    content = seen["content"]
    # Both top-level matches plus the nested one are present...
    assert "ship friday" in content
    assert "freeze monday" in content
    assert "nested note" in content
    assert "===== aaa =====" in content
    assert "===== bbb =====" in content
    assert "===== ccc =====" in content
    # ...the .json file is not...
    assert "not included" not in content
    assert "ignore" not in content
    # ...and the order is path-sorted, not filesystem-arbitrary.
    assert content.index("ship friday") < content.index("freeze monday")
    assert content.index("freeze monday") < content.index("nested note")


def test_llm_directory_match_is_case_insensitive(monkeypatch, tmp_path):
    # An uppercase .MD/.TXT suffix still recurses (the membership test lowercases).
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    (tmp_path / "shout.MD").write_text("loud note")
    result = runner.invoke(app, ["llm", "summarize", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert "loud note" in seen["content"]


def test_llm_directory_skips_subdir_named_like_a_match(monkeypatch, tmp_path):
    # rglob also yields directories; a folder literally named notes.md must not be
    # read as a file (the is_file guard), and a real match alongside it still loads.
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    (tmp_path / "trap.md").mkdir()  # a directory whose name ends in .md
    (tmp_path / "real.md").write_text("real content")
    result = runner.invoke(app, ["llm", "summarize", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert "real content" in seen["content"]


def test_llm_empty_directory_exits_2_without_network(monkeypatch, tmp_path):
    # A directory with no .md/.txt files is a usage error before auth/network —
    # the empty-guard the justfile used to carry for the CLI.
    _auth()
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the gateway")),
    )
    empty = tmp_path / "notes"
    empty.mkdir()
    (empty / "data.json").write_text("not a note")  # present but non-matching
    result = runner.invoke(app, ["llm", "summarize", str(empty)])
    assert result.exit_code == 2
    assert "No .md or .txt files found" in result.output


def test_llm_files_with_terminal_stdin_emits_no_warning(monkeypatch, tmp_path):
    # With files given and stdin a terminal (not piped), there's nothing being
    # ignored, so the "Ignoring piped stdin" warning must not fire.
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm._exec.stdio.stdin_is_piped", lambda: False)
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    note = tmp_path / "note.md"
    note.write_text("only the file")
    result = runner.invoke(app, ["llm", "summarize", str(note)])
    assert result.exit_code == 0
    assert "only the file" in seen["content"]
    assert "Ignoring piped stdin" not in result.output


def test_llm_transcript_id_takes_priority_over_files(monkeypatch, tmp_path):
    _auth()
    seen = {}
    # Pin stdin to a terminal so only the file argument is the ignored source.
    monkeypatch.setattr("aai_cli.commands.llm._exec.stdio.stdin_is_piped", lambda: False)

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["content"] = messages[0]["content"]
        seen["transcript_id"] = transcript_id
        return _payload("s")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    note = tmp_path / "note.md"
    note.write_text("file content here")
    result = runner.invoke(app, ["llm", "summarize", str(note), "--transcript-id", "t_9"])
    assert result.exit_code == 0
    assert seen["transcript_id"] == "t_9"
    assert "file content here" not in seen["content"]
    assert "Ignoring file arguments; --transcript-id takes priority." in result.output
