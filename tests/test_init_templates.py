from pathlib import Path

from aai_cli.init import templates

_TEMPLATES_ROOT = Path("aai_cli/init/templates")


def test_transcribe_is_registered():
    assert "transcribe" in templates.TEMPLATES
    assert "transcribe" in templates.TEMPLATE_ORDER


def test_order_matches_registry():
    # Every ordered id is registered and vice versa (no stray/missing entries).
    assert set(templates.TEMPLATE_ORDER) == set(templates.TEMPLATES)
    assert len(templates.TEMPLATE_ORDER) == len(templates.TEMPLATES)


def test_every_registered_template_has_a_directory():
    # The registry must never advertise a template whose files don't ship — that
    # would crash `aai init <id>` with a FileNotFoundError. This guards the picker.
    for tid in templates.TEMPLATES:
        assert (_TEMPLATES_ROOT / tid / "api" / "index.py").exists(), (
            f"template {tid!r} is registered but aai_cli/init/templates/{tid}/ is missing"
        )


def test_title_for_known_and_unknown():
    assert "Transcribe" in templates.title_for("transcribe")
    assert templates.title_for("nope") == "nope"  # falls back to the raw id


def test_is_template():
    assert templates.is_template("transcribe") is True
    assert templates.is_template("nope") is False
