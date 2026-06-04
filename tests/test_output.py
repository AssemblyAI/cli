import json

from assemblyai_cli import output


def test_resolve_json_true_when_explicit(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    assert output.resolve_json(explicit=True) is True


def test_resolve_json_true_when_not_tty(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: False)
    assert output.resolve_json(explicit=False) is True


def test_resolve_json_true_in_ci(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    monkeypatch.setenv("CI", "true")
    assert output.resolve_json(explicit=False) is True


def test_resolve_json_true_for_agent(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert output.resolve_json(explicit=False) is True


def test_resolve_json_false_for_human(monkeypatch):
    monkeypatch.setattr(output, "_stdout_is_tty", lambda: True)
    assert output.resolve_json(explicit=False) is False


def test_emit_json_serializes(capsys):
    output.emit({"a": 1}, lambda d: "human", json_mode=True)
    out = capsys.readouterr().out
    assert json.loads(out) == {"a": 1}


def test_emit_human_uses_renderer(capsys):
    output.emit({"a": 1}, lambda d: f"value={d['a']}", json_mode=False)
    assert "value=1" in capsys.readouterr().out


def test_emit_error_escapes_markup(capsys):
    import types

    err = types.SimpleNamespace(message="bad [tag] here", to_dict=lambda: {"error": {}})
    output.emit_error(err, json_mode=False)
    captured = capsys.readouterr()
    assert "[tag]" in captured.err  # error goes to stderr, not stripped as markup
    assert captured.out == ""  # stdout stays clean for pipelines


def test_emit_error_json_goes_to_stderr(capsys):
    import types

    err = types.SimpleNamespace(message="boom", to_dict=lambda: {"error": {"message": "boom"}})
    output.emit_error(err, json_mode=True)
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": {"message": "boom"}}
    assert captured.out == ""
