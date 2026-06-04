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


@pytest.fixture(params=TEMPLATE_DIRS, ids=lambda p: p.name)
def template_dir(request):
    return request.param


def test_required_files_present(template_dir):
    for rel in ("api/index.py", "index.html", "vercel.json",
                "requirements.txt", "README.md", "gitignore", "env.example"):
        assert (template_dir / rel).exists(), f"{template_dir.name} missing {rel}"


def test_no_committed_dotenv_or_real_key(template_dir):
    assert not (template_dir / ".env").exists(), f"{template_dir.name} ships a real .env"
    assert "your_assemblyai_api_key_here" in (template_dir / "env.example").read_text()


def test_frontend_routes_exist_in_backend(template_dir):
    """Every /api path the page fetches must be a route the backend registers."""
    html = (template_dir / "index.html").read_text()
    fetched = set(re.findall(r'fetch\(\s*["\'`](/api/[^"\'`?]+)', html))
    # Also catch template-literal paths like fetch(`/api/status/${id}`) and "/api/x/" + id
    fetched |= set(re.findall(r'["\'`](/api/[A-Za-z0-9_\-/]+?)(?:/?\$\{|/?["\'`]\s*\+)', html))
    src = (template_dir / "api" / "index.py").read_text()
    registered = set(re.findall(r'@app\.\w+\(\s*["\']([^"\']+)["\']', src))
    registered_bases = {re.sub(r"/\{[^}]+\}$", "", r).rstrip("/") for r in registered}
    for path in fetched:
        base = path.rstrip("/")
        assert any(base == r or base.startswith(r + "/") for r in registered_bases), (
            f"{template_dir.name}: index.html fetches {path!r}, "
            f"not registered in api/index.py (routes: {sorted(registered_bases)})"
        )


def test_requirements_cover_backend_imports(template_dir):
    """Every third-party import in api/index.py appears in requirements.txt."""
    tree = ast.parse((template_dir / "api" / "index.py").read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.add(node.names[0].name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".")[0])
    third_party = imports - _STDLIB
    reqs = (template_dir / "requirements.txt").read_text().lower()
    for pkg in third_party:
        dist = _PKG_MAP.get(pkg, pkg)
        assert dist in reqs, f"{template_dir.name}: import {pkg!r} ({dist}) missing from requirements.txt"


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
