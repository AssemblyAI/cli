import ast
import re
from pathlib import Path

import pytest

TEMPLATES_ROOT = Path("aai_cli/init/templates")
TEMPLATE_DIRS = sorted(
    p for p in TEMPLATES_ROOT.iterdir() if p.is_dir() and not p.name.startswith("__")
)
# Map an import name to its PyPI distribution where they differ.
_PKG_MAP = {"dotenv": "python-dotenv", "multipart": "python-multipart"}
_STDLIB = {"os", "tempfile", "uuid", "pathlib", "__future__", "json", "typing"}
_LOCAL_IMPORTS = {"api"}


@pytest.fixture(params=TEMPLATE_DIRS, ids=lambda p: p.name)
def template_dir(request):
    return request.param


def test_required_files_present(template_dir):
    for rel in (
        "api/index.py",
        "api/__init__.py",
        "api/settings.py",
        "public/index.html",
        "public/static/app.js",
        "public/static/styles.css",
        "requirements.txt",
        "README.md",
        "AGENTS.md",
        "gitignore",
        "env.example",
    ):
        assert (template_dir / rel).exists(), f"{template_dir.name} missing {rel}"


def test_realtime_templates_have_audio_helpers(template_dir):
    if template_dir.name in {"live-captions", "voice-agent"}:
        assert (template_dir / "public" / "static" / "audio.js").exists()


def test_static_assets_referenced_by_html_exist(template_dir):
    html = (template_dir / "public" / "index.html").read_text()
    refs = set(re.findall(r'(?:href|src)=["\'](/static/[^"\']+)', html))
    assert refs, f"{template_dir.name}: public/index.html should load static assets"
    for ref in refs:
        assert (template_dir / "public" / ref.lstrip("/")).exists(), (
            f"{template_dir.name}: public/index.html references missing asset {ref!r}"
        )


def test_codex_edit_points_are_explicit(template_dir):
    notes = (template_dir / "AGENTS.md").read_text()
    app_js = (template_dir / "public" / "static" / "app.js").read_text()
    assert "ASSEMBLYAI_API_KEY" in notes
    assert "buildless" in notes
    assert "public/static/app.js" in notes
    assert "_CONFIG" in app_js


def test_no_committed_dotenv_or_real_key(template_dir):
    assert not (template_dir / ".env").exists(), f"{template_dir.name} ships a real .env"
    assert "your_assemblyai_api_key_here" in (template_dir / "env.example").read_text()


def test_frontend_routes_exist_in_backend(template_dir):
    """Every /api path the page fetches must be a route the backend registers."""
    frontend = (template_dir / "public" / "index.html").read_text()
    frontend += "\n".join(
        path.read_text() for path in (template_dir / "public" / "static").glob("*.js")
    )
    fetched = set(re.findall(r'fetch\(\s*["\'`](/api/[^"\'`?]+)', frontend))
    # Also catch template-literal paths like fetch(`/api/status/${id}`) and "/api/x/" + id
    fetched |= set(re.findall(r'["\'`](/api/[A-Za-z0-9_\-/]+?)(?:/?\$\{|/?["\'`]\s*\+)', frontend))
    src = "\n".join(path.read_text() for path in (template_dir / "api").glob("*.py"))
    registered = set(re.findall(r'@app\.\w+\(\s*["\']([^"\']+)["\']', src))
    registered_bases = {re.sub(r"/\{[^}]+\}$", "", r).rstrip("/") for r in registered}
    for path in fetched:
        base = path.rstrip("/")
        assert any(base == r or base.startswith(r + "/") for r in registered_bases), (
            f"{template_dir.name}: public/index.html fetches {path!r}, "
            f"not registered in api/index.py (routes: {sorted(registered_bases)})"
        )


def test_requirements_cover_backend_imports(template_dir) -> None:
    """Every third-party import in api/*.py appears in requirements.txt."""
    imports: set[str] = set()
    for path in (template_dir / "api").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.add(node.names[0].name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    third_party = imports - _STDLIB - _LOCAL_IMPORTS
    reqs = (template_dir / "requirements.txt").read_text().lower()
    for pkg in third_party:
        dist = _PKG_MAP.get(pkg, pkg)
        assert dist in reqs, (
            f"{template_dir.name}: import {pkg!r} ({dist}) missing from requirements.txt"
        )


def test_status_endpoint_does_not_block(template_dir):
    """Guard against the blocking SDK call: a poll endpoint must not wait_for_completion."""
    src = (template_dir / "api" / "index.py").read_text()
    tree = ast.parse(src)
    blocking = {"get_by_id", "wait_for_completion"}
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (called & blocking), (
        f"{template_dir.name}: uses blocking SDK call {called & blocking} — "
        f"poll endpoints must do a single non-blocking fetch"
    )
