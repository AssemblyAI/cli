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
