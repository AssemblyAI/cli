from aai_cli import jsonshape


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
