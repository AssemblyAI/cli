# `aai api` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aai api`, a curl-style authenticated passthrough to the AssemblyAI REST and LLM-gateway APIs, driven by bundled OpenAPI specs, with `list` and an interactive picker.

**Architecture:** A new core library package `aai_cli/openapi/` parses two vendored OpenAPI specs (no live fetch) into typed `Endpoint`/`BodyField` objects and assembles requests; a new `aai_cli/commands/api.py` Typer sub-app resolves the host from `environments.active()` per `--api`, applies spec-derived auth, executes over `httpx2`, and renders results through the existing `output` helpers. A release-time script regenerates the vendored specs; a parametrized contract test keeps them honest.

**Tech Stack:** Python 3.12+, Typer, `httpx2` (existing dep), Rich (command layer only), pytest + pytest-mock + syrupy, uv.

---

## File structure

| File | Responsibility |
|---|---|
| `aai_cli/openapi/__init__.py` | Package marker; re-export public API (`load_spec`, `Endpoint`, `BodyField`, `ApiName`). |
| `aai_cli/openapi/loader.py` | Read vendored JSON, parse to `Endpoint`/`BodyField`, resolve `$ref`/`allOf`, expose security scheme + endpoints. No Rich, no commands. |
| `aai_cli/openapi/request.py` | Pure request assembly: `-F` typed-field parsing, `@file`, `--input`/stdin body, method defaulting, header parsing. No network. |
| `aai_cli/openapi/specs/rest.json` | Vendored REST spec snapshot (force-included in wheel). |
| `aai_cli/openapi/specs/llm-gateway.json` | Vendored LLM-gateway spec snapshot (force-included in wheel). |
| `aai_cli/commands/api.py` | Typer sub-app: passthrough, `list`, picker, host×env resolution, auth, execution, rendering, error mapping, `--show-code`. |
| `scripts/update-openapi-specs.py` | Release-time regeneration of vendored specs from the GitHub repo. |
| `tests/test_openapi_request.py` | Unit tests for `request.py`. |
| `tests/test_openapi_loader.py` | Unit tests for `loader.py`. |
| `tests/test_api_command.py` | Command tests (CliRunner): resolution, auth, errors, paginate, list, picker. |
| `tests/test_openapi_specs_contract.py` | Parametrized contract tests over the vendored specs. |
| `aai_cli/main.py` | Register sub-app + `_COMMAND_ORDER` entry (modify). |
| `aai_cli/help_panels.py` | Add `API` panel constant (modify). |
| `.importlinter` | Add `aai_cli.openapi` to contracts 1 & 3, `aai_cli.commands.api` to contract 2 (modify). |
| `pyproject.toml` | Add `aai_cli/openapi/specs/**` to wheel artifacts (modify). |
| `AGENTS.md` | Document the command + spec-regeneration script (modify). |

---

## Task 1: Vendor the OpenAPI specs + wheel packaging

**Files:**
- Create: `aai_cli/openapi/specs/rest.json`
- Create: `aai_cli/openapi/specs/llm-gateway.json`
- Create: `scripts/update-openapi-specs.py`
- Modify: `pyproject.toml` (wheel `artifacts` list)

- [ ] **Step 1: Write the spec-regeneration script**

Create `scripts/update-openapi-specs.py`:

```python
#!/usr/bin/env python3
"""Regenerate the vendored OpenAPI specs from the AssemblyAI spec repo.

Run at release time. Downloads the REST and LLM-gateway specs, normalizes both
to JSON, and writes them into aai_cli/openapi/specs/. Run via:

    uv run python scripts/update-openapi-specs.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import yaml  # provided by the dev environment (pyyaml)

RAW = "https://raw.githubusercontent.com/AssemblyAI/assemblyai-api-spec/main"
SOURCES = {
    "rest": f"{RAW}/openapi.json",
    "llm-gateway": f"{RAW}/llm-gateway.yml",
}
DEST = Path(__file__).resolve().parent.parent / "aai_cli" / "openapi" / "specs"


def _fetch(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
        raw = resp.read().decode("utf-8")
    return yaml.safe_load(raw)  # yaml.safe_load also parses JSON


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        spec = _fetch(url)
        out = DEST / f"{name}.json"
        out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out.relative_to(DEST.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the vendored specs**

Run: `uv run python scripts/update-openapi-specs.py`
Expected output:
```
wrote aai_cli/openapi/specs/rest.json
wrote aai_cli/openapi/specs/llm-gateway.json
```

- [ ] **Step 3: Verify the generated files parse and have expected shape**

Run:
```bash
uv run python -c "
import json, pathlib
for n in ('rest','llm-gateway'):
    d=json.loads(pathlib.Path(f'aai_cli/openapi/specs/{n}.json').read_text())
    print(n, d['openapi'], len(d['paths']), list(d['components']['securitySchemes']))
"
```
Expected: `rest 3.1.0 8 ['ApiKey']` and `llm-gateway 3.1.0 2 ['ApiKey']`

- [ ] **Step 4: Force-include specs in the wheel**

In `pyproject.toml`, modify the `[tool.hatch.build.targets.wheel] artifacts` list to add the specs glob:

```toml
artifacts = [
    "aai_cli/init/templates/**",
    "aai_cli/skills/**",
    "aai_cli/streaming/macos_system_audio.swift",
    "aai_cli/openapi/specs/**",
]
```

- [ ] **Step 5: Verify the wheel includes the specs**

Run: `uv build 2>/dev/null && uv run python -c "import zipfile,glob; w=sorted(glob.glob('dist/*.whl'))[-1]; names=zipfile.ZipFile(w).namelist(); print([n for n in names if 'openapi/specs' in n])"`
Expected: a list containing `aai_cli/openapi/specs/rest.json` and `aai_cli/openapi/specs/llm-gateway.json`

- [ ] **Step 6: Commit**

```bash
git add scripts/update-openapi-specs.py aai_cli/openapi/specs/ pyproject.toml
git commit -m "feat(api): vendor REST + LLM-gateway OpenAPI specs and packaging"
```

---

## Task 2: Spec loader — `aai_cli/openapi/loader.py`

**Files:**
- Create: `aai_cli/openapi/__init__.py`
- Create: `aai_cli/openapi/loader.py`
- Test: `tests/test_openapi_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_openapi_loader.py`:

```python
from __future__ import annotations

import pytest

from aai_cli.openapi import loader


def test_load_rest_lists_transcript_endpoints():
    spec = loader.load_spec("rest")
    paths = {(e.method, e.path) for e in spec.endpoints}
    assert ("POST", "/v2/transcript") in paths
    assert ("GET", "/v2/transcript/{transcript_id}") in paths


def test_rest_security_scheme_is_raw_apikey_header():
    spec = loader.load_spec("rest")
    assert spec.auth_header_name == "Authorization"
    assert spec.auth_bearer is False


def test_gateway_lists_chat_completions():
    spec = loader.load_spec("llm-gateway")
    paths = {(e.method, e.path) for e in spec.endpoints}
    assert ("POST", "/chat/completions") in paths


def test_post_transcript_body_fields_resolved():
    spec = loader.load_spec("rest")
    ep = next(e for e in spec.endpoints if e.method == "POST" and e.path == "/v2/transcript")
    names = {f.name for f in ep.body_fields}
    assert "audio_url" in names  # required string field on the transcript request


def test_unknown_api_raises():
    with pytest.raises(ValueError):
        loader.load_spec("nope")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openapi_loader.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aai_cli.openapi'`

- [ ] **Step 3: Create the package marker**

Create `aai_cli/openapi/__init__.py`:

```python
from __future__ import annotations

from aai_cli.openapi.loader import ApiName, BodyField, Endpoint, LoadedSpec, load_spec

__all__ = ["ApiName", "BodyField", "Endpoint", "LoadedSpec", "load_spec"]
```

- [ ] **Step 4: Implement the loader**

Create `aai_cli/openapi/loader.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, Literal

ApiName = Literal["rest", "llm-gateway"]
_APIS: tuple[ApiName, ...] = ("rest", "llm-gateway")
_METHODS = ("get", "post", "put", "patch", "delete")


@dataclass(frozen=True)
class BodyField:
    """One property of a JSON request body, flattened from the schema."""

    name: str
    required: bool
    type: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class Endpoint:
    path: str
    method: str  # upper-case
    summary: str = ""
    body_fields: list[BodyField] = field(default_factory=list)


@dataclass(frozen=True)
class LoadedSpec:
    api: ApiName
    endpoints: list[Endpoint]
    auth_header_name: str
    auth_bearer: bool


def load_spec(api: ApiName) -> LoadedSpec:
    """Parse the vendored spec for `api` into endpoints + auth metadata."""
    if api not in _APIS:
        raise ValueError(f"Unknown API spec: {api!r}. Choose from {_APIS}.")
    raw = files("aai_cli.openapi.specs").joinpath(f"{api}.json").read_text("utf-8")
    spec: dict[str, Any] = json.loads(raw)
    header_name, bearer = _auth(spec)
    return LoadedSpec(
        api=api,
        endpoints=_endpoints(spec),
        auth_header_name=header_name,
        auth_bearer=bearer,
    )


def _auth(spec: dict[str, Any]) -> tuple[str, bool]:
    """Read the single security scheme. AssemblyAI uses an apiKey header."""
    schemes = spec.get("components", {}).get("securitySchemes", {})
    for scheme in schemes.values():
        if scheme.get("type") == "apiKey" and scheme.get("in") == "header":
            return scheme.get("name", "Authorization"), False
        if scheme.get("type") == "http" and scheme.get("scheme") == "bearer":
            return "Authorization", True
    return "Authorization", False


def _endpoints(spec: dict[str, Any]) -> list[Endpoint]:
    out: list[Endpoint] = []
    for path, item in spec.get("paths", {}).items():
        for method in _METHODS:
            operation = item.get(method)
            if operation is None:
                continue
            out.append(
                Endpoint(
                    path=path,
                    method=method.upper(),
                    summary=operation.get("summary", "") or item.get("summary", ""),
                    body_fields=_body_fields(spec, operation),
                )
            )
    out.sort(key=lambda e: (e.path, e.method))
    return out


def _body_fields(spec: dict[str, Any], operation: dict[str, Any]) -> list[BodyField]:
    content = operation.get("requestBody", {}).get("content", {})
    schema = content.get("application/json", {}).get("schema")
    resolved = _resolve(spec, schema)
    props = resolved.get("properties", {}) if resolved else {}
    required = set(resolved.get("required", []) if resolved else [])
    fields = [
        BodyField(
            name=name,
            required=name in required,
            type=_resolve(spec, prop).get("type") if _resolve(spec, prop) else None,
            description=(_resolve(spec, prop) or {}).get("description"),
        )
        for name, prop in props.items()
    ]
    fields.sort(key=lambda f: (not f.required, f.name))
    return fields


def _resolve(spec: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve a $ref and merge allOf into a single object schema."""
    if not schema:
        return None
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/components/schemas/"):
            return None
        name = ref.rsplit("/", 1)[-1]
        target = spec.get("components", {}).get("schemas", {}).get(name)
        return _resolve(spec, target)
    if "allOf" in schema:
        merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for sub in schema["allOf"]:
            part = _resolve(spec, sub) or {}
            merged["properties"].update(part.get("properties", {}))
            merged["required"].extend(part.get("required", []))
        return merged
    return schema
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_openapi_loader.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add aai_cli/openapi/__init__.py aai_cli/openapi/loader.py tests/test_openapi_loader.py
git commit -m "feat(api): OpenAPI spec loader"
```

---

## Task 3: Request assembly — `aai_cli/openapi/request.py`

**Files:**
- Create: `aai_cli/openapi/request.py`
- Test: `tests/test_openapi_request.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_openapi_request.py`:

```python
from __future__ import annotations

import pytest

from aai_cli.errors import UsageError
from aai_cli.openapi import request


def test_typed_field_parses_numbers_bools_json():
    body = request.build_body(fields=["n=42", "ok=true", "tags=[1,2]"], raw_fields=[], input_path=None)
    assert body == {"n": 42, "ok": True, "tags": [1, 2]}


def test_raw_field_stays_string():
    body = request.build_body(fields=[], raw_fields=["id=00123"], input_path=None)
    assert body == {"id": "00123"}


def test_field_from_file(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("hello")
    body = request.build_body(fields=[f"note=@{p}"], raw_fields=[], input_path=None)
    assert body == {"note": "hello"}


def test_input_file_is_whole_body(tmp_path):
    p = tmp_path / "b.json"
    p.write_text('{"audio_url": "u"}')
    body = request.build_body(fields=[], raw_fields=[], input_path=str(p))
    assert body == {"audio_url": "u"}


def test_malformed_field_raises_usage_error():
    with pytest.raises(UsageError):
        request.build_body(fields=["noequals"], raw_fields=[], input_path=None)


def test_parse_headers():
    assert request.parse_headers(["X-A: 1", "X-B:2"]) == {"X-A": "1", "X-B": "2"}


def test_default_method_get_without_body():
    assert request.default_method(method=None, has_body=False) == "GET"


def test_default_method_post_with_body():
    assert request.default_method(method=None, has_body=True) == "POST"


def test_explicit_method_wins():
    assert request.default_method(method="delete", has_body=False) == "DELETE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openapi_request.py -q`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` on `request`.

- [ ] **Step 3: Implement request assembly**

Create `aai_cli/openapi/request.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from aai_cli.errors import UsageError


def build_body(
    *, fields: list[str], raw_fields: list[str], input_path: str | None
) -> dict[str, Any] | None:
    """Assemble a JSON request body from --input, -F, and -f.

    --input (whole-body file or stdin) is mutually exclusive with field flags.
    """
    if input_path is not None:
        if fields or raw_fields:
            raise UsageError("Use --input or -F/-f fields, not both.")
        return _load_input(input_path)
    if not fields and not raw_fields:
        return None
    body: dict[str, Any] = {}
    for item in raw_fields:
        key, value = _split(item)
        body[key] = value
    for item in fields:
        key, value = _split(item)
        body[key] = _typed(value)
    return body


def _split(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise UsageError(f"Invalid field {item!r}; expected KEY=VALUE.")
    key, value = item.split("=", 1)
    if not key:
        raise UsageError(f"Invalid field {item!r}; missing key.")
    return key, value


def _typed(value: str) -> Any:
    if value.startswith("@"):
        return Path(value[1:]).read_text("utf-8")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_input(input_path: str) -> dict[str, Any]:
    text = sys.stdin.read() if input_path == "-" else Path(input_path).read_text("utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(f"--input is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise UsageError("--input must be a JSON object.")
    return parsed


def parse_headers(headers: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in headers:
        if ":" not in item:
            raise UsageError(f"Invalid header {item!r}; expected KEY:VALUE.")
        key, value = item.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def default_method(*, method: str | None, has_body: bool) -> str:
    if method is not None:
        return method.upper()
    return "POST" if has_body else "GET"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openapi_request.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add aai_cli/openapi/request.py tests/test_openapi_request.py
git commit -m "feat(api): request body/header/method assembly"
```

---

## Task 4: Import-linter contracts + help panel

**Files:**
- Modify: `.importlinter`
- Modify: `aai_cli/help_panels.py`

- [ ] **Step 1: Add the openapi library modules to contract 1**

In `.importlinter`, under `[importlinter:contract:1]` `source_modules`, add two lines (keeping alphabetical-ish order, after `aai_cli.microphone`):

```
    aai_cli.openapi.loader
    aai_cli.openapi.request
```

- [ ] **Step 2: Add openapi modules to contract 3 (no Rich)**

In `.importlinter`, under `[importlinter:contract:3]` `source_modules`, add:

```
    aai_cli.openapi.loader
    aai_cli.openapi.request
```

- [ ] **Step 3: Add the command to contract 2 (independence)**

In `.importlinter`, under `[importlinter:contract:2]` `modules`, add:

```
    aai_cli.commands.api
```

- [ ] **Step 4: Add the API help panel constant**

In `aai_cli/help_panels.py`, add after the `SETUP` line:

```python
API = "API"  # raw API passthrough: api
```

- [ ] **Step 5: Verify contracts still pass (openapi modules exist, command not yet)**

Run: `uv run lint-imports`
Expected: PASS. (The `aai_cli.commands.api` line references a module created in Task 5; if `lint-imports` errors that the module is missing, defer Step 3 to the start of Task 5. Note this in the commit if so.)

- [ ] **Step 6: Commit**

```bash
git add .importlinter aai_cli/help_panels.py
git commit -m "chore(api): import-linter contracts and API help panel"
```

---

## Task 5: The `aai api` command — passthrough + execution

**Files:**
- Create: `aai_cli/commands/api.py`
- Test: `tests/test_api_command.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_command.py`:

```python
from __future__ import annotations

import json

import httpx2
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


class _FakeResponse:
    def __init__(self, status_code: int, payload: object, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {"content-type": "application/json"}
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self._payload


def _capture(mocker, response: _FakeResponse) -> dict[str, object]:
    seen: dict[str, object] = {}

    def fake_request(method, url, *, headers=None, json=None, **_kw):
        seen["method"] = method
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        return response

    mocker.patch.object(httpx2, "request", side_effect=fake_request)
    return seen


def test_get_sends_auth_header_to_active_host(mocker):
    config.set_api_key("default", "sk_live")
    seen = _capture(mocker, _FakeResponse(200, {"id": "t_1"}))
    result = runner.invoke(app, ["api", "/v2/transcript/t_1"])
    assert result.exit_code == 0
    assert seen["method"] == "GET"
    assert seen["url"] == "https://api.assemblyai.com/v2/transcript/t_1"
    assert seen["headers"]["Authorization"] == "sk_live"
    assert '"id": "t_1"' in result.output


def test_field_triggers_post(mocker):
    config.set_api_key("default", "sk_live")
    seen = _capture(mocker, _FakeResponse(200, {"ok": True}))
    result = runner.invoke(app, ["api", "/v2/transcript", "-F", "audio_url=https://x/a.mp3"])
    assert result.exit_code == 0
    assert seen["method"] == "POST"
    assert seen["json"] == {"audio_url": "https://x/a.mp3"}


def test_sandbox_changes_host(mocker):
    config.set_api_key("default", "sk_live")
    seen = _capture(mocker, _FakeResponse(200, {}))
    result = runner.invoke(app, ["--sandbox", "api", "/v2/transcript/t_1"])
    assert result.exit_code == 0
    assert str(seen["url"]).startswith("https://api.sandbox000.assemblyai-labs.com")


def test_llm_gateway_api_uses_gateway_host(mocker):
    config.set_api_key("default", "sk_live")
    seen = _capture(mocker, _FakeResponse(200, {}))
    result = runner.invoke(app, ["api", "/chat/completions", "--api", "llm-gateway", "-F", "model=m"])
    assert result.exit_code == 0
    assert str(seen["url"]).startswith("https://llm-gateway.assemblyai.com/v1/chat/completions")


def test_auth_error_maps_to_clean_exit_4(mocker):
    config.set_api_key("default", "sk_live")
    _capture(mocker, _FakeResponse(401, {"error": "bad key"}))
    result = runner.invoke(app, ["api", "/v2/transcript/t_1"])
    assert result.exit_code == 4
    assert "Traceback" not in result.output


def test_non_2xx_maps_to_api_error_exit_1(mocker):
    config.set_api_key("default", "sk_live")
    _capture(mocker, _FakeResponse(404, {"error": "not found"}))
    result = runner.invoke(app, ["api", "/v2/transcript/missing"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_endpoint_must_start_with_slash():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["api", "v2/transcript"])
    assert result.exit_code == 2  # UsageError


def test_show_code_emits_curl_without_key(mocker):
    # No key set; --show-code must not require auth.
    spy = mocker.patch.object(httpx2, "request")
    result = runner.invoke(app, ["api", "/v2/transcript/t_1", "--show-code"])
    assert result.exit_code == 0
    assert "curl" in result.output
    assert "ASSEMBLYAI_API_KEY" in result.output
    spy.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_command.py -q`
Expected: FAIL (no `api` command registered).

- [ ] **Step 3: Implement the command body**

Create `aai_cli/commands/api.py`:

```python
from __future__ import annotations

import json as jsonlib
from typing import Any

import httpx2
import typer

from aai_cli import choices, config, environments, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import APIError, UsageError, auth_failure
from aai_cli.help_text import examples_epilog
from aai_cli.openapi import loader, request

app = typer.Typer(help="Make authenticated requests to the AssemblyAI API.")


def _base_url(api: loader.ApiName) -> str:
    env = environments.active()
    return env.api_base if api == "rest" else env.llm_gateway_base


def _curl_snippet(method: str, url: str, header_name: str, body: dict[str, Any] | None) -> str:
    lines = [f"curl -X {method} '{url}' \\", f"  -H '{header_name}: '\"$ASSEMBLYAI_API_KEY\""]
    if body is not None:
        lines[-1] += " \\"
        lines.append("  -H 'Content-Type: application/json' \\")
        lines.append(f"  -d '{jsonlib.dumps(body)}'")
    return "\n".join(lines)


@app.command(
    epilog=examples_epilog(
        [
            ("Get a transcript", "aai api /v2/transcript/5551234-abcd"),
            ("Create a transcript", "aai api /v2/transcript -F audio_url=https://x/a.mp3"),
            ("Delete a transcript", "aai api /v2/transcript/5551234-abcd -X DELETE"),
            ("Show the equivalent curl", "aai api /v2/transcript/5551234-abcd --show-code"),
        ]
    )
)
def api(
    ctx: typer.Context,
    endpoint: str = typer.Argument(..., help="API path starting with / (e.g. /v2/transcript)."),
    method: str | None = typer.Option(None, "-X", "--method", help="HTTP method."),
    fields: list[str] = typer.Option([], "-F", "--field", help="Typed field KEY=VALUE."),
    raw_fields: list[str] = typer.Option([], "-f", "--raw-field", help="String field KEY=VALUE."),
    headers: list[str] = typer.Option([], "-H", "--header", help="Extra header KEY:VALUE."),
    input_path: str | None = typer.Option(None, "--input", help="Body from file (- for stdin)."),
    api_name: choices.ApiSpec = typer.Option(
        choices.ApiSpec.rest, "--api", help="Which API spec/host."
    ),
    include: bool = typer.Option(False, "-i", "--include", help="Include response headers."),
    show_code: bool = typer.Option(
        False, "--show-code", help="Print an equivalent curl command and exit."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Make an authenticated request to an AssemblyAI API endpoint."""
    if not endpoint.startswith("/"):
        raise UsageError("Endpoint must start with '/'.", suggestion="e.g. aai api /v2/transcript")

    api_key_name: loader.ApiName = api_name.value
    body = request.build_body(fields=fields, raw_fields=raw_fields, input_path=input_path)
    http_method = request.default_method(method=method, has_body=body is not None)
    spec = loader.load_spec(api_key_name)
    url = _base_url(api_key_name).rstrip("/") + endpoint

    if show_code:
        output.print_code(
            _curl_snippet(http_method, url, spec.auth_header_name, body), language="bash"
        )
        return

    def run(state: AppState, json_mode: bool) -> None:
        key = config.resolve_api_key(profile=state.profile)
        request_headers = request.parse_headers(headers)
        request_headers[spec.auth_header_name] = f"Bearer {key}" if spec.auth_bearer else key
        try:
            response = httpx2.request(
                http_method, url, headers=request_headers, json=body, timeout=60.0
            )
        except httpx2.HTTPError as exc:
            raise APIError(f"Request failed: {exc}") from exc
        _emit(response, include=include, json_mode=json_mode)

    run_command(ctx, run, json=json_out)


def _emit(response: object, *, include: bool, json_mode: bool) -> None:
    status = getattr(response, "status_code")
    if status in (401, 403):
        raise auth_failure()
    payload = _safe_json(response)
    if status >= 400:
        message = payload if isinstance(payload, str) else jsonlib.dumps(payload)
        raise APIError(f"API returned {status}: {message}")
    if include:
        for key, value in getattr(response, "headers", {}).items():
            output.emit_text(f"{key}: {value}")
        output.emit_text("")
    output.emit(payload, lambda d: jsonlib.dumps(d, indent=2), json_mode=json_mode)


def _safe_json(response: object) -> Any:
    try:
        return response.json()  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        return getattr(response, "text", "")
```

- [ ] **Step 4: Add the `ApiSpec` choices enum**

In `aai_cli/choices.py`, add a new enum (match the existing `str, Enum` style):

```python
class ApiSpec(str, Enum):
    rest = "rest"
    llm_gateway = "llm-gateway"
```

Note: `ApiSpec.llm_gateway.value == "llm-gateway"` matches `loader.ApiName`.

- [ ] **Step 5: Register the command in `main.py`**

In `aai_cli/main.py`: add `api` to the imports from `aai_cli.commands`, add `app.add_typer(api.app, name="api", rich_help_panel=help_panels.API)` near the other `add_typer` calls, and add `"api"` to `_COMMAND_ORDER` right after `"llm"`.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_api_command.py -q`
Expected: PASS (8 passed). If `test_llm_gateway_api_uses_gateway_host` fails on the `api_name.value` typing, ensure `api_key_name` is assigned from `api_name.value` (a `str`) and passed to `load_spec`.

- [ ] **Step 7: Commit**

```bash
git add aai_cli/commands/api.py aai_cli/choices.py aai_cli/main.py tests/test_api_command.py
git commit -m "feat(api): aai api passthrough command"
```

---

## Task 6: `--paginate` support

**Files:**
- Modify: `aai_cli/commands/api.py`
- Test: `tests/test_api_command.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_command.py`:

```python
def test_paginate_follows_next_url(mocker):
    config.set_api_key("default", "sk_live")
    pages = [
        _FakeResponse(200, {"transcripts": [{"id": "a"}], "page_details": {"next_url": "/v2/transcript?after_id=a"}}),
        _FakeResponse(200, {"transcripts": [{"id": "b"}], "page_details": {"next_url": None}}),
    ]
    calls: list[str] = []

    def fake_request(method, url, **_kw):
        calls.append(url)
        return pages[len(calls) - 1]

    mocker.patch.object(httpx2, "request", side_effect=fake_request)
    result = runner.invoke(app, ["api", "/v2/transcript", "--paginate", "--json"])
    assert result.exit_code == 0
    assert len(calls) == 2
    merged = json.loads(result.output)
    assert [t["id"] for t in merged["transcripts"]] == ["a", "b"]


def test_paginate_rejected_for_gateway():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["api", "/chat/completions", "--api", "llm-gateway", "--paginate", "-F", "model=m"])
    assert result.exit_code == 2  # UsageError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_command.py -k paginate -q`
Expected: FAIL (`--paginate` option does not exist).

- [ ] **Step 3: Implement pagination**

In `aai_cli/commands/api.py`, add the option to the `api` signature (after `include`):

```python
    paginate: bool = typer.Option(False, "--paginate", help="Follow page_details.next_url."),
```

Add the guard after the `startswith("/")` check:

```python
    if paginate and api_name == choices.ApiSpec.llm_gateway:
        raise UsageError("--paginate is only supported for the REST API.")
```

Replace the single-request `run` body's request/emit with a paginating helper. Add this module-level function:

```python
def _paginate(http_method: str, base: str, url: str, headers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    next_path: str | None = url
    while next_path:
        response = httpx2.request(http_method, next_path, headers=headers, timeout=60.0)
        if response.status_code in (401, 403):
            raise auth_failure()
        if response.status_code >= 400:
            raise APIError(f"API returned {response.status_code}: {response.text}")
        page = response.json()
        for key, value in page.items():
            if isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            elif key not in merged:
                merged[key] = value
        rel = page.get("page_details", {}).get("next_url")
        next_path = (base.rstrip("/") + rel) if rel else None
    return merged
```

In `run`, branch on `paginate`:

```python
        if paginate:
            merged = _paginate(http_method, _base_url(api_key_name), url, request_headers)
            output.emit(merged, lambda d: jsonlib.dumps(d, indent=2), json_mode=json_mode)
            return
        try:
            response = httpx2.request(...)  # unchanged
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_command.py -k paginate -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/api.py tests/test_api_command.py
git commit -m "feat(api): --paginate follows page_details.next_url"
```

---

## Task 7: `aai api list` subcommand

**Files:**
- Modify: `aai_cli/commands/api.py`
- Test: `tests/test_api_command.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_command.py`:

```python
def test_list_shows_rest_endpoints():
    result = runner.invoke(app, ["api", "list"])
    assert result.exit_code == 0
    assert "/v2/transcript" in result.output


def test_list_json_is_machine_readable():
    result = runner.invoke(app, ["api", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert any(e["path"] == "/v2/transcript" and e["method"] == "POST" for e in data)


def test_list_gateway():
    result = runner.invoke(app, ["api", "list", "--api", "llm-gateway", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert any(e["path"] == "/chat/completions" for e in data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_command.py -k list -q`
Expected: FAIL (no `list` subcommand).

- [ ] **Step 3: Implement `list`**

In `aai_cli/commands/api.py`, add:

```python
@app.command(name="list")
def list_endpoints(
    ctx: typer.Context,
    api_name: choices.ApiSpec = typer.Option(choices.ApiSpec.rest, "--api", help="Which API."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List the endpoints available in the bundled OpenAPI spec."""

    def run(state: AppState, json_mode: bool) -> None:
        spec = loader.load_spec(api_name.value)
        rows = [
            {"method": e.method, "path": e.path, "summary": e.summary} for e in spec.endpoints
        ]
        output.emit(rows, _render_list, json_mode=json_mode)

    run_command(ctx, run, json=json_out, auto_login=False)


def _render_list(rows: list[dict[str, str]]) -> object:
    table = output.data_table("Method", "Path", "Summary")
    for row in rows:
        table.add_row(row["method"], row["path"], row["summary"])
    return table
```

Note `auto_login=False`: `list` reads only the bundled spec and needs no API key.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_command.py -k list -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/api.py tests/test_api_command.py
git commit -m "feat(api): aai api list from bundled spec"
```

---

## Task 8: Interactive endpoint picker

**Files:**
- Modify: `aai_cli/commands/api.py`
- Test: `tests/test_api_command.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_command.py`:

```python
def test_no_endpoint_non_tty_errors(mocker):
    config.set_api_key("default", "sk_live")
    # CliRunner stdin is not a TTY.
    result = runner.invoke(app, ["api"])
    assert result.exit_code == 2  # UsageError
    assert "interactive" in result.output.lower()


def test_picker_selects_endpoint_then_requests(mocker):
    config.set_api_key("default", "sk_live")
    mocker.patch("aai_cli.commands.api._is_tty", return_value=True)
    # Pick the GET /v2/transcript/{id} endpoint, then supply the path param.
    mocker.patch(
        "aai_cli.commands.api._prompt_endpoint",
        return_value=("GET", "/v2/transcript/{transcript_id}"),
    )
    mocker.patch("aai_cli.commands.api._prompt_path_params", return_value="/v2/transcript/t_9")
    seen = _capture(mocker, _FakeResponse(200, {"id": "t_9"}))
    result = runner.invoke(app, ["api"])
    assert result.exit_code == 0
    assert seen["url"] == "https://api.assemblyai.com/v2/transcript/t_9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_command.py -k "picker or non_tty" -q`
Expected: FAIL (endpoint is currently a required argument; no picker).

- [ ] **Step 3: Make `endpoint` optional and add the picker**

In `aai_cli/commands/api.py`:

Change the `endpoint` argument to optional:

```python
    endpoint: str | None = typer.Argument(None, help="API path starting with / (e.g. /v2/transcript)."),
```

Add helpers and the resolution at the top of the `api` function body (before the `startswith` check):

```python
import sys as _sys


def _is_tty() -> bool:
    return _sys.stdin.isatty()


def _prompt_endpoint(spec: loader.LoadedSpec) -> tuple[str, str]:
    choices_list = [f"{e.method} {e.path}" for e in spec.endpoints]
    for index, label in enumerate(choices_list):
        output.emit_text(f"  {index}: {label}")
    raw = typer.prompt("Select an endpoint number")
    selected = spec.endpoints[int(raw)]
    return selected.method, selected.path


def _prompt_path_params(path: str) -> str:
    import re

    def fill(match: "re.Match[str]") -> str:
        return typer.prompt(f"Value for {match.group(1)}")

    return re.sub(r"\{([^}]+)\}", fill, path)
```

In the `api` function, before the `startswith` validation:

```python
    if endpoint is None:
        if not _is_tty():
            raise UsageError(
                "Endpoint is required.", suggestion="Run `aai api` interactively in a terminal."
            )
        spec_for_pick = loader.load_spec(api_name.value)
        picked_method, picked_path = _prompt_endpoint(spec_for_pick)
        endpoint = _prompt_path_params(picked_path)
        if method is None:
            method = picked_method
```

(The existing `startswith("/")` check and the rest of the body run unchanged afterward.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_command.py -k "picker or non_tty" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full command test file**

Run: `uv run pytest tests/test_api_command.py -q`
Expected: PASS (all tests from Tasks 5–8).

- [ ] **Step 6: Commit**

```bash
git add aai_cli/commands/api.py tests/test_api_command.py
git commit -m "feat(api): interactive endpoint picker on a TTY"
```

---

## Task 9: Spec contract tests

**Files:**
- Test: `tests/test_openapi_specs_contract.py`

- [ ] **Step 1: Write the contract test**

Create `tests/test_openapi_specs_contract.py`:

```python
from __future__ import annotations

import pytest

from aai_cli import environments
from aai_cli.openapi import loader

_EXPECTED = {
    "rest": ("POST", "/v2/transcript"),
    "llm-gateway": ("POST", "/chat/completions"),
}


@pytest.mark.parametrize("api", ["rest", "llm-gateway"])
def test_spec_parses_and_has_expected_endpoint(api):
    spec = loader.load_spec(api)
    assert spec.endpoints, f"{api} spec has no endpoints"
    method, path = _EXPECTED[api]
    assert any(e.method == method and e.path == path for e in spec.endpoints)


@pytest.mark.parametrize("api", ["rest", "llm-gateway"])
def test_spec_declares_auth_header(api):
    spec = loader.load_spec(api)
    assert spec.auth_header_name == "Authorization"


@pytest.mark.parametrize("api", ["rest", "llm-gateway"])
def test_each_api_maps_to_a_real_env_host(api):
    env = environments.active()
    host = env.api_base if api == "rest" else env.llm_gateway_base
    assert host.startswith("https://")
```

- [ ] **Step 2: Run the contract test**

Run: `uv run pytest tests/test_openapi_specs_contract.py -q`
Expected: PASS (9 passed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_openapi_specs_contract.py
git commit -m "test(api): parametrized contract tests for bundled specs"
```

---

## Task 10: Snapshot tests + docs + full gate

**Files:**
- Modify: `tests/test_api_command.py` (snapshot of `--show-code` and `list`)
- Modify: `AGENTS.md`
- Regenerate: `tests/__snapshots__/*.ambr`

- [ ] **Step 1: Add snapshot tests**

Append to `tests/test_api_command.py`:

```python
def test_show_code_snapshot(snapshot):
    result = runner.invoke(app, ["api", "/v2/transcript", "-F", "audio_url=https://x/a.mp3", "--show-code"])
    assert result.exit_code == 0
    assert result.output == snapshot


def test_help_snapshot(snapshot):
    result = runner.invoke(app, ["api", "--help"])
    assert result.exit_code == 0
    assert result.output == snapshot
```

- [ ] **Step 2: Generate snapshots**

Run: `uv run pytest tests/test_api_command.py -k snapshot --snapshot-update -q`
Expected: snapshots written; then `uv run pytest tests/test_api_command.py -k snapshot -q` PASSES.

- [ ] **Step 3: Document the command in AGENTS.md**

In `AGENTS.md`, under the "Feature subsystems" list, add a bullet:

```markdown
- **`openapi/`** + **`commands/api.py`** — `aai api` is a curl-style authenticated passthrough to the REST and LLM-gateway APIs, driven by **bundled** OpenAPI specs (`aai_cli/openapi/specs/*.json`, force-included in the wheel). The host is resolved from `environments.active()` per `--api` (so `--sandbox`/`--env` work), and the `Authorization` header is read from the spec's security scheme. Regenerate the vendored specs with `uv run python scripts/update-openapi-specs.py`; the parametrized `tests/test_openapi_specs_contract.py` guards against a stale/malformed spec.
```

- [ ] **Step 4: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` If `vulture` flags an unused field (e.g. `BodyField.description`), either consume it in the picker output or add it to the vulture allowlist as the codebase does elsewhere. If `xenon` flags `api()` complexity, extract the picker/paginate branches into the already-defined helpers (they are).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_command.py tests/__snapshots__/ AGENTS.md
git commit -m "test(api): snapshots; docs: document aai api in AGENTS.md"
```

---

## Self-review notes

- **Spec coverage:** passthrough (T5), multi-host/env (T5), spec-driven auth (T2/T5), `-F/-f/-H/--input/@file/stdin` (T3), `--api` (T5/T7), `--paginate` (T6), `--show-code` (T5), `list` (T7), picker (T8), bundled specs + wheel (T1), regeneration script (T1), contract tests (T9), error mapping (T5), import-linter/help panel/placement (T4/T5), docs (T10) — all covered.
- **Type consistency:** `loader.ApiName` (`"rest"`/`"llm-gateway"`) matches `choices.ApiSpec` values; `LoadedSpec.auth_header_name`/`auth_bearer` used identically in T5; `BodyField`/`Endpoint` fields consistent across tasks.
- **Open follow-up flagged for the executor:** the `auth_bearer` branch in `_emit`/request headers is dead today (both specs use raw apiKey) — kept so the loader stays spec-driven, but `vulture` may flag it; if so, cover it with a tiny unit test in `tests/test_openapi_loader.py` using a synthetic bearer scheme dict rather than deleting the branch.
