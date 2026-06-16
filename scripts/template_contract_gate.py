from __future__ import annotations

import ast
import http.client
import importlib
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import NoReturn

from aai_cli.init import templates

_ROOT = Path("aai_cli/init/templates")
_REQUIRED_FILES = (
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
)
_LOCAL_IMPORTS = {"api"}
_PKG_MAP = {"dotenv": "python-dotenv", "multipart": "python-multipart"}
_STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _template_dirs() -> dict[str, Path]:
    # On-disk dirs are underscore package names; registry ids are kebab. Map each
    # shipped dir back to its kebab id so both sets compare in the id namespace.
    # Templates are now importable packages, so importing them creates __pycache__
    # alongside the template dirs — skip dunder dirs (matches the registry tests).
    dirs = {
        path.name.replace("_", "-"): path
        for path in _ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    registered = set(templates.TEMPLATES)
    shipped = set(dirs)
    missing = registered - shipped
    extra = shipped - registered
    if missing:
        _fail(f"registered templates missing directories: {sorted(missing)}")
    if extra:
        _fail(f"template directories not registered: {sorted(extra)}")
    return {name: dirs[name] for name in templates.TEMPLATE_ORDER}


def _required_files(template: str, path: Path) -> None:
    for rel in _REQUIRED_FILES:
        if not (path / rel).exists():
            _fail(f"{template}: missing {rel}")
    if template in {"live-captions", "voice-agent"} and not (path / "static/audio.js").exists():
        _fail(f"{template}: missing static/audio.js")


def _html_static_refs(template: str, path: Path) -> None:
    html = (path / "static/index.html").read_text(encoding="utf-8")
    refs = set(re.findall(r'(?:href|src)=["\'](/static/[^"\']+)', html))
    if not refs:
        _fail(f"{template}: static/index.html should load static assets")
    for ref in refs:
        if not (path / ref.lstrip("/")).exists():
            _fail(f"{template}: static/index.html references missing asset {ref!r}")


def _frontend_routes(template: str, path: Path) -> None:
    frontend = (path / "static/index.html").read_text(encoding="utf-8")
    frontend += "\n".join(
        asset.read_text(encoding="utf-8") for asset in (path / "static").glob("*.js")
    )
    fetched = set(re.findall(r'fetch\(\s*["\'`](/api/[^"\'`?]+)', frontend))
    fetched |= set(re.findall(r'["\'`](/api/[A-Za-z0-9_\-/]+?)(?:/?\$\{|/?["\'`]\s*\+)', frontend))
    backend = "\n".join(
        source.read_text(encoding="utf-8") for source in (path / "api").glob("*.py")
    )
    registered = set(re.findall(r'@app\.\w+\(\s*["\']([^"\']+)["\']', backend))
    registered_bases = {re.sub(r"/\{[^}]+\}$", "", route).rstrip("/") for route in registered}
    for route in fetched:
        base = route.rstrip("/")
        if not any(
            base == registered or base.startswith(registered + "/")
            for registered in registered_bases
        ):
            _fail(
                f"{template}: frontend fetches {route!r}, "
                f"but backend routes are {sorted(registered_bases)}"
            )


def _requirements_cover_imports(template: str, path: Path) -> None:
    imports: set[str] = set()
    for source in (path / "api").glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.add(node.names[0].name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    third_party = imports - _STDLIB - _LOCAL_IMPORTS
    reqs = (path / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in third_party:
        dist = _PKG_MAP.get(package, package)
        if dist not in reqs:
            _fail(f"{template}: import {package!r} ({dist}) missing from requirements.txt")


_SPECIFIER = re.compile(r"(===|==|~=|!=|>=|<=|>|<)")


def _requirements_pin_versions(template: str, path: Path) -> None:
    """Every requirement must carry a version specifier.

    SCA scanners read a starter app's requirements.txt as a lockfile; an unpinned
    line like ``fastapi`` reports as a missing version and blocks vulnerability
    analysis. Require a specifier (``>=`` floor, ``==`` pin, ...) on every line.
    """
    unpinned: list[str] = []
    for raw in (path / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not _SPECIFIER.search(line.split(";", 1)[0]):  # ignore any env marker
            unpinned.append(line)
    if unpinned:
        _fail(f"{template}: requirements.txt has unpinned dependencies {unpinned}")


@contextmanager
def _template_import_path(path: Path):
    old_path = list(sys.path)
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "api" or name.startswith("api.")
    }
    for name in old_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path.resolve()))
    try:
        yield
    finally:
        for name in [name for name in sys.modules if name == "api" or name.startswith("api.")]:
            sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path


def _import_api(template: str, path: Path) -> None:
    with _template_import_path(path):
        module = importlib.import_module("api.index")
    app = getattr(module, "app", None)
    if app is None:
        _fail(f"{template}: api/index.py does not export app")
    routes = {getattr(route, "path", "") for route in getattr(app, "routes", [])}
    if "/" not in routes:
        _fail(f"{template}: FastAPI app does not register /")


def _parse_python_files(path: Path) -> None:
    for source in (path / "api").glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


# Supported interpreters (see pyproject `requires-python`). runtime.txt tells
# buildpack platforms (Render/Railway/Heroku/Cloud Run) which Python to provision.
_RUNTIME = re.compile(r"^python-3\.(12|13)(\.\d+)?$")


def _runtime_supported(template: str, path: Path) -> None:
    pin = (path / "runtime.txt").read_text(encoding="utf-8").strip()
    if not _RUNTIME.match(pin):
        _fail(f"{template}: runtime.txt pins {pin!r}; must be python-3.12 or python-3.13")


def _web_command(template: str, path: Path) -> str:
    """The Procfile's `web:` process command — the start command every non-Vercel host runs."""
    for raw in (path / "Procfile").read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("web:"):
            command = raw.split("web:", 1)[1].strip()
            if "uvicorn" not in command or "api.index:app" not in command:
                _fail(
                    f"{template}: Procfile web command must run `uvicorn api.index:app`: {command!r}"
                )
            return command
    _fail(f"{template}: Procfile has no `web:` process")
    raise AssertionError  # unreachable; _fail raises. Satisfies the type checker.


def _dockerfile_runs_uvicorn(template: str, path: Path) -> None:
    """The Dockerfile's start command must run the app on the platform's port.

    Fly.io / Railway / Render(Docker) / Cloudflare-Containers build this image instead
    of `fly launch`'s broken auto-generated Dockerfile. It must run uvicorn on the app,
    bind 0.0.0.0 so the platform can reach it, and honor the injected ${PORT}.
    """
    dockerfile = (path / "Dockerfile").read_text(encoding="utf-8")
    for token in ("uvicorn api.index:app", "--host 0.0.0.0", "${PORT"):
        if token not in dockerfile:
            _fail(f"{template}: Dockerfile must contain {token!r}")
    # Fly auto-detects internal_port from EXPOSE; without a match to the CMD's
    # default port, Fly's proxy hits a port the app never binds.
    exposed = re.search(r"^EXPOSE\s+(\d+)\s*$", dockerfile, re.MULTILINE)
    cmd_default = re.search(r"--port \$\{PORT:-(\d+)\}", dockerfile)
    if exposed is None:
        _fail(f"{template}: Dockerfile must declare EXPOSE 8080")
    if cmd_default is None:
        _fail(f"{template}: Dockerfile CMD must default to ${{PORT:-8080}}")
    if exposed.group(1) != "8080" or cmd_default.group(1) != "8080":
        _fail(
            f"{template}: Dockerfile must EXPOSE 8080 and default CMD to ${{PORT:-8080}}; "
            f"got EXPOSE {exposed.group(1)} and ${{PORT:-{cmd_default.group(1)}}}"
        )
    # Container hardening: the image must drop root (Aikido/Checkov CKV_DOCKER_3).
    user = re.search(r"^USER\s+(\S+)\s*$", dockerfile, re.MULTILINE)
    if user is None:
        _fail(f"{template}: Dockerfile must declare a non-root USER (CKV_DOCKER_3)")
    if user.group(1) in {"root", "0"}:
        _fail(f"{template}: Dockerfile USER must not be root; got {user.group(1)!r} (CKV_DOCKER_3)")


def _dockerignore_excludes_env(template: str, path: Path) -> None:
    """`.env` (the real API key) must be excluded from the build context.

    The Dockerfile does `COPY . .` and Fly/Railway upload the local dir as build
    context, so without this the key would be baked into the image.
    """
    lines = {
        line.strip() for line in (path / "dockerignore").read_text(encoding="utf-8").splitlines()
    }
    if ".env" not in lines:
        _fail(f"{template}: dockerignore must list .env so the API key isn't baked into the image")


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _terminate(proc: subprocess.Popen[str]) -> str:
    """Kill the process group (uvicorn + its shell) and return whatever it logged."""
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
    return proc.stdout.read() if proc.stdout else ""


_HTTP_OK = 200


def _serves_root(port: int) -> bool:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        conn.request("GET", "/")
        return conn.getresponse().status == _HTTP_OK
    finally:
        conn.close()


def _procfile_boots(template: str, path: Path) -> None:
    """Run the Procfile's web command for real and confirm the app answers GET / with 200.

    The other checks are static; this is the one that proves a `git push` to Render,
    Railway, Heroku, or Cloud Run actually starts a serving app. PORT is injected the way
    those platforms inject it; the key is unused at boot (settings default it to "").
    """
    command = _web_command(template, path)
    port = _free_port()
    env = {**os.environ, "PORT": str(port), "ASSEMBLYAI_API_KEY": ""}
    # `/bin/sh -c` so the Procfile's ${PORT:-3000} expands as it would on the host;
    # a fresh session group lets _terminate reap the shell and uvicorn together.
    proc = subprocess.Popen(
        ["/bin/sh", "-c", command],
        cwd=path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                _fail(f"{template}: Procfile process exited before serving:\n{_terminate(proc)}")
            try:
                if _serves_root(port):
                    return
            except OSError:
                time.sleep(0.25)  # not up yet; poll again
        _fail(f"{template}: Procfile app did not serve / within 30s:\n{_terminate(proc)}")
    finally:
        _terminate(proc)


def _untracked_template_files() -> None:
    in_worktree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        check=False,
        text=True,
    )
    if in_worktree.returncode != 0:
        return
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", str(_ROOT)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        _fail(result.stderr.strip() or "could not inspect template git tracking")
    files = [line for line in result.stdout.splitlines() if line]
    if files:
        _fail(f"template files are untracked and would be missing from a clean checkout: {files}")


def main() -> int:
    _untracked_template_files()
    for template, path in _template_dirs().items():
        _required_files(template, path)
        _html_static_refs(template, path)
        _frontend_routes(template, path)
        _requirements_cover_imports(template, path)
        _requirements_pin_versions(template, path)
        _parse_python_files(path)
        _import_api(template, path)
        _runtime_supported(template, path)
        _dockerfile_runs_uvicorn(template, path)
        _dockerignore_excludes_env(template, path)
        _procfile_boots(template, path)
    sys.stdout.write(f"validated {len(templates.TEMPLATE_ORDER)} init templates\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
