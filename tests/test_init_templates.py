from pathlib import Path

from aai_cli.init import templates

_TEMPLATES_ROOT = Path("aai_cli/init/templates")


def test_audio_transcription_is_registered():
    assert "audio-transcription" in templates.TEMPLATES
    assert "audio-transcription" in templates.TEMPLATE_ORDER


def test_order_matches_registry():
    # Every ordered id is registered and vice versa (no stray/missing entries).
    assert set(templates.TEMPLATE_ORDER) == set(templates.TEMPLATES)
    assert len(templates.TEMPLATE_ORDER) == len(templates.TEMPLATES)


def test_every_registered_template_has_a_directory():
    # The registry must never advertise a template whose files don't ship — that
    # would crash `assembly init <id>` with a FileNotFoundError. This guards the picker.
    for tid in templates.TEMPLATES:
        assert (_TEMPLATES_ROOT / tid / "api" / "index.py").exists(), (
            f"template {tid!r} is registered but aai_cli/init/templates/{tid}/ is missing"
        )


def test_every_shipped_directory_is_registered():
    # The other direction: a template dir that ships but isn't registered is invisible
    # in the picker and unreachable via `assembly init <id>`. Together with the test above
    # this enforces registry == shipped directories.
    for path in _TEMPLATES_ROOT.iterdir():
        if path.is_dir() and not path.name.startswith("__"):
            assert path.name in templates.TEMPLATES, (
                f"aai_cli/init/templates/{path.name}/ ships but isn't registered in TEMPLATES"
            )


def test_title_for_known_and_unknown():
    assert "Audio Transcription" in templates.title_for("audio-transcription")
    assert templates.title_for("nope") == "nope"  # falls back to the raw id


def test_is_template():
    assert templates.is_template("audio-transcription") is True
    assert templates.is_template("nope") is False
