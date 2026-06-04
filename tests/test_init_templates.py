from aai_cli.init import templates


def test_all_four_template_ids_present():
    assert set(templates.TEMPLATES) == {"transcribe", "stream", "agent", "llm"}


def test_template_order_is_complete_and_stable():
    # Display order mirrors the CLI's command order; every id appears exactly once.
    assert templates.TEMPLATE_ORDER == ("transcribe", "stream", "agent", "llm")
    assert set(templates.TEMPLATE_ORDER) == set(templates.TEMPLATES)


def test_title_for_known_and_unknown():
    assert "Transcribe" in templates.title_for("transcribe")
    assert templates.title_for("nope") == "nope"  # falls back to the raw id


def test_is_template():
    assert templates.is_template("agent") is True
    assert templates.is_template("nope") is False
