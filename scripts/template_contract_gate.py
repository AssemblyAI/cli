from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from aai_cli.init import templates

_ROOT = Path("aai_cli/init/templates")
_REQUIRED_FILES = (
    "api/index.py",
    "api/__init__.py",
    "api/settings.py",
    "index.html",
    "static/app.js",
    "static/styles.css",
    "vercel.json",
    "requirements.txt",
    "README.md",
    "AGENTS.md",
    "gitignore",
    "env.example",
)
_LOCAL_IMPORTS = {"api"}
_PKG_MAP = {"dotenv": "python-dotenv", "multipart": "python-multipart"}
_STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _template_dirs() -> dict[str, Path]:
    dirs = {path.name: path for path in _ROOT.iterdir() if path.is_dir()}
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
    if template in {"stream", "agent"} and not (path / "static/audio.js").exists():
        _fail(f"{template}: missing static/audio.js")


def _html_static_refs(template: str, path: Path) -> None:
    html = (path / "index.html").read_text(encoding="utf-8")
    refs = set(re.findall(r'(?:href|src)=["\'](/static/[^"\']+)', html))
    if not refs:
        _fail(f"{template}: index.html should load static assets")
    for ref in refs:
        if not (path / ref.lstrip("/")).exists():
            _fail(f"{template}: index.html references missing asset {ref!r}")


def _frontend_routes(template: str, path: Path) -> None:
    frontend = (path / "index.html").read_text(encoding="utf-8")
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


def _parse_json_files(template: str, path: Path) -> None:
    for source in (path / "api").glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    try:
        json.loads((path / "vercel.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{template}: vercel.json is invalid JSON: {exc}")


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
        _parse_json_files(template, path)
        _import_api(template, path)
    sys.stdout.write(f"validated {len(templates.TEMPLATE_ORDER)} init templates\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
