import datetime

from aai_cli.core import jsonshape


def test_as_mapping_accepts_json_objects_only():
    assert jsonshape.as_mapping({"ok": 1}) == {"ok": 1}
    assert jsonshape.as_mapping(["bad"]) is None


def test_object_list_accepts_lists_only():
    assert jsonshape.object_list([{"ok": 1}, "bad"]) == [{"ok": 1}, "bad"]
    assert jsonshape.object_list({"bad": "shape"}) == []


def test_mapping_list_filters_non_objects():
    assert jsonshape.mapping_list([{"ok": 1}, "bad", {"also": "ok"}]) == [
        {"ok": 1},
        {"also": "ok"},
    ]
    assert jsonshape.mapping_list("bad") == []


def test_as_int_coerces_scalars_and_defaults():
    assert jsonshape.as_int(True) == 0  # bool is not a count
    assert jsonshape.as_int(12) == 12
    assert jsonshape.as_int(12.9) == 0  # non-integral float is not a valid int
    assert jsonshape.as_int("13") == 13
    assert jsonshape.as_int("bad") == 0
    assert jsonshape.as_int(object()) == 0
    assert jsonshape.as_int(None, default=-1) == -1


def test_as_float_coerces_scalars_and_defaults():
    assert jsonshape.as_float(True) == 0.0  # bool is not a count
    assert jsonshape.as_float(1) == 1.0
    assert jsonshape.as_float("1.5") == 1.5
    assert jsonshape.as_float("bad") == 0.0
    assert jsonshape.as_float(object()) == 0.0
    assert jsonshape.as_float(None, default=-1.0) == -1.0


def test_dumps_round_trips_plain_json():
    assert jsonshape.dumps({"a": 1, "b": [2, 3]}) == '{"a": 1, "b": [2, 3]}'


def test_dumps_falls_back_to_str_for_unserializable_values():
    # A datetime isn't natively JSON-serializable; default=str must stringify it
    # instead of raising — the safety every CLI emission path depends on.
    moment = datetime.datetime(2026, 6, 13, 14, 0, 0)
    assert jsonshape.dumps({"at": moment}) == '{"at": "2026-06-13 14:00:00"}'


def test_compact_drops_only_none_values():
    assert jsonshape.compact({"keep": 0, "blank": "", "false": False, "drop": None}) == {
        "keep": 0,
        "blank": "",
        "false": False,
    }
