import io

from assemblyai_cli import theme
from assemblyai_cli.commands.claude import _render_steps


def test_render_steps_colors_status():
    data = {
        "steps": [
            {"name": "mcp", "status": "installed", "detail": "/path"},
            {"name": "skill", "status": "failed", "detail": "nope"},
        ]
    }
    rendered = _render_steps(data)
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, color_system="truecolor")
    console.print(rendered)
    out = buf.getvalue()
    assert "installed" in out
    assert "failed" in out
    assert "\x1b[" in out  # statuses are colored
