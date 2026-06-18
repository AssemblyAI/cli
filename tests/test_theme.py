import io

from aai_cli.ui import theme


def test_make_console_resolves_named_styles():
    console = theme.make_console()
    # get_style raises rich.errors.MissingStyle if a name is not in the theme.
    for name in (
        "aai.brand",
        "aai.heading",
        "aai.label",
        "aai.url",
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


def test_you_color_reserved_outside_speaker_palette():
    # "You" keeps the brand accent; no diarized speaker may resolve to that same color,
    # so the mic is always visually distinct from a system speaker.
    console = theme.make_console()
    you_color = console.get_style("aai.you").color
    speaker_colors = {console.get_style(name).color for name in theme.SPEAKER_STYLES}
    assert you_color not in speaker_colors


def test_you_and_agent_stay_distinct_after_downsampling():
    # "you" and "agent" used to be two near-identical cobolt purples that downsampled to
    # the *same* 16-color ANSI slot, so a transcript looked single-colored on a basic
    # terminal. Assert they're different hues and stay different once downgraded.
    from rich.color import ColorSystem

    console = theme.make_console()
    you = console.get_style("aai.you").color
    agent = console.get_style("aai.agent").color
    assert you is not None and agent is not None
    assert you != agent
    assert you.downgrade(ColorSystem.STANDARD) != agent.downgrade(ColorSystem.STANDARD)


def test_output_console_is_themed_and_error_is_styled(monkeypatch):
    from aai_cli.core.errors import CLIError
    from aai_cli.ui import output, theme

    buf = io.StringIO()
    monkeypatch.setattr(
        output,
        "error_console",  # errors render on the stderr console
        theme.make_console(file=buf, force_terminal=True, color_system="truecolor"),
    )
    output.emit_error(CLIError("boom"), json_mode=False)
    out = buf.getvalue()
    assert "Error:" in out
    assert "boom" in out
    assert "\x1b[" in out  # themed error emits ANSI on a forced-color console


def test_pipe_safe_console_reraises_broken_pipe():
    # Rich's default on_broken_pipe converts EPIPE to SystemExit(1); the CLI's
    # consoles must re-raise so main.run() can treat a closed pipe as success.
    import io

    import pytest

    from aai_cli.ui import theme

    class BrokenFile(io.StringIO):
        def write(self, s):
            raise BrokenPipeError

    console = theme.make_console(file=BrokenFile())
    with pytest.raises(BrokenPipeError):
        console.print("hello")
