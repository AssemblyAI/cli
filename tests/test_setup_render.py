import io

from aai_cli import theme
from aai_cli.setup_exec import render
from aai_cli.steps import Step


def test_render_steps_colors_status() -> None:
    data: dict[str, list[Step]] = {
        "steps": [
            {"name": "mcp", "status": "installed", "detail": "/path"},
            {"name": "skill", "status": "failed", "detail": "nope"},
        ]
    }
    rendered = render(data)
    # The markup string carries the semantic style tags per status...
    assert "[aai.success]installed[/aai.success]" in rendered
    assert "[aai.error]failed[/aai.error]" in rendered
    assert "[aai.heading]" in rendered
    # ...and renders to real ANSI through the themed console.
    buf = io.StringIO()
    # _environ={} pins the color depth: Rich reads ambient color env at render time,
    # which leaks across tests and otherwise flips truecolor to a 16-color downsample.
    console = theme.make_console(
        file=buf, force_terminal=True, color_system="truecolor", _environ={}
    )
    console.print(rendered)
    out = buf.getvalue()
    assert "installed" in out
    assert "failed" in out
    assert "\x1b[1;32m" in out  # aai.success (bold green) → "installed"
    assert "\x1b[1;38;2;240;68;56m" in out  # aai.error (bold brand red #F04438) → "failed"
