import pytest

from aai_cli.core import project
from aai_cli.core.errors import UsageError
from aai_cli.ui import output


def test_parse_fields_splits_and_trims():
    # Comma-split with surrounding whitespace trimmed and empty segments dropped.
    assert project.parse_fields(" id , status ") == ["id", "status"]
    assert project.parse_fields("id") == ["id"]
    assert project.parse_fields("a,,b") == ["a", "b"]


def test_parse_fields_rejects_empty_spec():
    with pytest.raises(UsageError) as exc:
        project.parse_fields(" , ")
    assert "-o" in exc.value.message


def test_lookup_descends_dotted_paths_and_yields_none_when_missing():
    record = {"a": {"b": "nested"}, "top": 1}
    assert project._lookup(record, "top") == 1
    assert project._lookup(record, "a.b") == "nested"
    # A missing key, and a path that runs off a non-object, both yield None.
    assert project._lookup(record, "a.z") is None
    assert project._lookup(record, "missing") is None
    assert project._lookup(record, "top.deeper") is None


def test_render_value_scalars_and_containers():
    assert project.render_value(None) == ""
    # JSON booleans render lowercased so they read like the --json payload.
    assert project.render_value(True) == "true"
    assert project.render_value(False) == "false"
    assert project.render_value("text") == "text"
    assert project.render_value(7) == "7"
    assert project.render_value(1.5) == "1.5"
    # A nested object/list re-serializes as JSON, not Python repr.
    assert project.render_value({"a": 1}) == '{"a": 1}'
    assert project.render_value([1, 2]) == "[1, 2]"


def test_project_record_tab_separates_columns():
    record = {"id": 1, "status": "done", "flag": True, "none": None}
    assert project.project_record(record, ["id", "status"]) == "1\tdone"
    # Missing field and None both become empty columns; tab is the separator.
    assert project.project_record(record, ["flag", "none", "missing"]) == "true\t\t"


def test_project_rows_one_line_per_record():
    rows = [{"id": 1, "status": "done"}, {"id": 2}]
    assert project.project_rows(rows, ["id", "status"]) == ["1\tdone", "2\t"]


def test_project_any_dispatches_on_shape():
    # A single object -> one line; a list -> one line per row.
    assert project.project_any({"id": 9, "name": "k"}, ["id", "name"]) == ["9\tk"]
    assert project.project_any([{"id": 1}, {"id": 2}], ["id"]) == ["1", "2"]
    # A bare scalar has nothing to project -> no lines.
    assert project.project_any("scalar", ["id"]) == []


def test_emit_fields_lists_and_records(capsys):
    output.emit_fields([{"id": 1}, {"id": 2}], ["id"])
    assert capsys.readouterr().out == "1\n2\n"
    output.emit_fields({"id": 9, "name": "k"}, ["id", "name"])
    assert capsys.readouterr().out == "9\tk\n"
    # A non-list, non-object value has nothing to project, so emits nothing.
    output.emit_fields("not-json", ["id"])
    assert capsys.readouterr().out == ""


def test_emit_fields_precedence_over_json(capsys):
    # When fields is set it wins over json_mode: columns, not a JSON dump.
    output.emit([{"id": 1}], lambda _d: "human", json_mode=True, fields="id")
    assert capsys.readouterr().out == "1\n"


def test_emit_without_fields_keeps_json_and_human(capsys):
    output.emit({"id": 1}, lambda _d: "human-line", json_mode=True, fields=None)
    assert capsys.readouterr().out.strip() == '{"id": 1}'
    output.emit({"id": 1}, lambda _d: "human-line", json_mode=False, fields=None)
    assert "human-line" in capsys.readouterr().out
