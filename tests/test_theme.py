import io

from assemblyai_cli import theme


def test_make_console_resolves_named_styles():
    console = theme.make_console()
    # get_style raises rich.errors.MissingStyle if a name is not in the theme.
    for name in (
        "aai.brand",
        "aai.heading",
        "aai.label",
        "aai.success",
        "aai.error",
        "aai.warn",
        "aai.muted",
    ):
        console.get_style(name)
    for name in theme.SPEAKER_STYLES:
        console.get_style(name)


def test_make_console_passes_kwargs_through():
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, width=42)
    assert console.file is buf
    assert console.width == 42


def test_status_style_maps_known_statuses():
    assert theme.status_style("completed") == "aai.success"
    assert theme.status_style("ERROR") == "aai.error"
    assert theme.status_style("failed") == "aai.error"
    assert theme.status_style("queued") == "aai.warn"
    assert theme.status_style("processing") == "aai.warn"


def test_status_style_unknown_falls_back_to_muted():
    assert theme.status_style("something-else") == "aai.muted"


def test_speaker_style_deterministic_and_in_palette():
    assert theme.speaker_style("A") in theme.SPEAKER_STYLES
    assert theme.speaker_style("A") == theme.speaker_style("A")
    assert theme.speaker_style("A") != theme.speaker_style("B")
