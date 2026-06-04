from aai_cli.init import steps


def test_render_steps_includes_name_status_detail():
    data = [
        {"name": "scaffold", "status": "created", "detail": "./my-app"},
        {"name": "install", "status": "skipped", "detail": "--no-install"},
    ]
    out = steps.render_steps(data)
    assert "scaffold" in out
    assert "created" in out
    assert "./my-app" in out
    assert "install" in out
    assert "skipped" in out


def test_render_steps_has_heading():
    out = steps.render_steps([{"name": "scaffold", "status": "created", "detail": "x"}])
    assert "init" in out.lower() or "AssemblyAI" in out
