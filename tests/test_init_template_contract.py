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
        "static/index.html",
        "static/app.js",
        "static/styles.css",
        "requirements.txt",
        "README.md",
        "AGENTS.md",
        "gitignore",
        "env.example",
        "Procfile",
        "runtime.txt",
        "Dockerfile",
        "dockerignore",
    ):
        assert (template_dir / rel).exists(), f"{template_dir.name} missing {rel}"


def test_dockerfile_runs_uvicorn_on_platform_port(template_dir):
    """Fly/Railway/Render(Docker)/Cloudflare-Containers build this image. It must run
    uvicorn on the app, bind 0.0.0.0, and honor the platform's injected ${PORT}."""
    dockerfile = (template_dir / "Dockerfile").read_text()
    assert "uvicorn api.index:app" in dockerfile, (
        f"{template_dir.name}: Dockerfile must run uvicorn api.index:app"
    )
    assert "--host 0.0.0.0" in dockerfile, (
        f"{template_dir.name}: Dockerfile must bind --host 0.0.0.0"
    )
    assert "${PORT" in dockerfile, (
        f"{template_dir.name}: Dockerfile must honor the platform's ${{PORT}}"
    )
    # Fly auto-detects internal_port from EXPOSE; it must match the CMD's default
    # port or Fly's proxy hits a port the app never binds (connection refused).
    exposed = re.search(r"^EXPOSE\s+(\d+)\s*$", dockerfile, re.MULTILINE)
    cmd_default = re.search(r"--port \$\{PORT:-(\d+)\}", dockerfile)
    assert exposed is not None and exposed.group(1) == "8080", (
        f"{template_dir.name}: Dockerfile must declare EXPOSE 8080"
    )
    assert cmd_default is not None and cmd_default.group(1) == "8080", (
        f"{template_dir.name}: Dockerfile CMD must default to ${{PORT:-8080}}"
    )
    assert exposed.group(1) == cmd_default.group(1), (
        f"{template_dir.name}: EXPOSE {exposed.group(1)} must match "
        f"CMD default ${{PORT:-{cmd_default.group(1)}}}"
    )
    # Container hardening: the image must drop root (Aikido/Checkov CKV_DOCKER_3).
    user = re.search(r"^USER\s+(\S+)\s*$", dockerfile, re.MULTILINE)
    assert user is not None, (
        f"{template_dir.name}: Dockerfile must declare a non-root USER (CKV_DOCKER_3)"
    )
    assert user.group(1) not in {"root", "0"}, (
        f"{template_dir.name}: Dockerfile USER must not be root; "
        f"got {user.group(1)!r} (CKV_DOCKER_3)"
    )


def test_dockerignore_excludes_env(template_dir):
    """`.env` holds the real API key; the Dockerfile does COPY . . so it must be
    excluded from the build context or the key gets baked into the image."""
    lines = {line.strip() for line in (template_dir / "dockerignore").read_text().splitlines()}
    assert ".env" in lines, (
        f"{template_dir.name}: dockerignore must list .env so the API key isn't baked in"
    )


def test_template_ships_no_public_dir(template_dir):
    # Vercel serves a top-level public/** from its CDN and omits it from the Python
    # lambda, so a FastAPI app that reads from public/ crashes at import on deploy.
    assert not (template_dir / "public").exists(), (
        f"{template_dir.name}: ships a public/ dir; Vercel drops it from the function "
        f"bundle and the app crashes (FUNCTION_INVOCATION_FAILED). Use static/."
    )


def test_procfile_starts_the_app(template_dir):
    """The Procfile gives non-Vercel hosts (Render/Railway/Heroku/Cloud Run) a start
    command. The contract gate boots it for real; here we pin its shape."""
    web = [
        line.split("web:", 1)[1].strip()
        for line in (template_dir / "Procfile").read_text().splitlines()
        if line.strip().startswith("web:")
    ]
    assert web, f"{template_dir.name}: Procfile has no web: process"
    assert "uvicorn" in web[0] and "api.index:app" in web[0], (
        f"{template_dir.name}: Procfile must launch uvicorn api.index:app, got {web[0]!r}"
    )
    assert "$PORT" in web[0] or "${PORT" in web[0], (
        f"{template_dir.name}: Procfile must bind the platform's $PORT, got {web[0]!r}"
    )


def test_runtime_pins_supported_python(template_dir):
    pin = (template_dir / "runtime.txt").read_text().strip()
    assert re.fullmatch(r"python-3\.(12|13)(\.\d+)?", pin), (
        f"{template_dir.name}: runtime.txt pins {pin!r}; must be python-3.12 or python-3.13"
    )


def test_realtime_templates_have_audio_helpers(template_dir):
    if template_dir.name in {"live-captions", "voice-agent"}:
        assert (template_dir / "static" / "audio.js").exists()


def test_static_assets_referenced_by_html_exist(template_dir):
    html = (template_dir / "static" / "index.html").read_text()
    refs = set(re.findall(r'(?:href|src)=["\'](/static/[^"\']+)', html))
    assert refs, f"{template_dir.name}: static/index.html should load static assets"
    for ref in refs:
        assert (template_dir / ref.lstrip("/")).exists(), (
            f"{template_dir.name}: static/index.html references missing asset {ref!r}"
        )


def test_codex_edit_points_are_explicit(template_dir):
    notes = (template_dir / "AGENTS.md").read_text()
    app_js = (template_dir / "static" / "app.js").read_text()
    assert "ASSEMBLYAI_API_KEY" in notes
    assert "buildless" in notes
    assert "static/app.js" in notes
    assert "_CONFIG" in app_js


def test_no_committed_dotenv_or_real_key(template_dir):
    assert not (template_dir / ".env").exists(), f"{template_dir.name} ships a real .env"
    assert "your_assemblyai_api_key_here" in (template_dir / "env.example").read_text()


def test_frontend_routes_exist_in_backend(template_dir):
    """Every /api path the page fetches must be a route the backend registers."""
    frontend = (template_dir / "static" / "index.html").read_text()
    frontend += "\n".join(path.read_text() for path in (template_dir / "static").glob("*.js"))
    fetched = set(re.findall(r'fetch\(\s*["\'`](/api/[^"\'`?]+)', frontend))
    # Also catch template-literal paths like fetch(`/api/status/${id}`) and "/api/x/" + id
    fetched |= set(re.findall(r'["\'`](/api/[A-Za-z0-9_\-/]+?)(?:/?\$\{|/?["\'`]\s*\+)', frontend))
    src = "\n".join(path.read_text() for path in (template_dir / "api").glob("*.py"))
    registered = set(re.findall(r'@app\.\w+\(\s*["\']([^"\']+)["\']', src))
    registered_bases = {re.sub(r"/\{[^}]+\}$", "", r).rstrip("/") for r in registered}
    for path in fetched:
        base = path.rstrip("/")
        assert any(base == r or base.startswith(r + "/") for r in registered_bases), (
            f"{template_dir.name}: static/index.html fetches {path!r}, "
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


def test_requirements_pin_versions(template_dir) -> None:
    """Every dependency in requirements.txt carries a version specifier.

    SCA scanners read a starter app's requirements.txt as a lockfile: an unpinned
    line like ``fastapi`` reports as a missing version and blocks vulnerability
    analysis. Require a specifier (``>=`` floor, ``==`` pin, ...) on every line.
    """
    specifier = re.compile(r"(===|==|~=|!=|>=|<=|>|<)")
    unpinned: list[str] = []
    for raw in (template_dir / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not specifier.search(line.split(";", 1)[0]):  # ignore any env marker
            unpinned.append(line)
    assert not unpinned, (
        f"{template_dir.name}: requirements.txt has unpinned dependencies {unpinned}; "
        f"each needs a version specifier so SCA scanners can analyze the lockfile"
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
