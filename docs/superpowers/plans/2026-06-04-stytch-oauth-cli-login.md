# Stytch OAuth CLI Login — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the interactive paste-an-API-key `aai login` with a browser-based Stytch B2B OAuth login that ends by storing a dedicated AssemblyAI API key in the keychain — leaving every other command unchanged.

**Architecture:** New `assemblyai_cli/auth/` package owns a client-side, fixed-loopback-port browser flow: open Stytch B2B OAuth discovery → capture the discovery token on `127.0.0.1:8585` → walk the Accounts Management Service (AMS) `discover → exchange → /v1/auth → projects → tokens` chain to find-or-create an "AssemblyAI CLI" key → store it via the existing `config.set_api_key`. OAuth credentials are transient; only the API key is persisted (store-key-only).

**Tech Stack:** Python, Typer, httpx (HTTP), stdlib `http.server` (loopback), `keyring` (storage), pytest + `typer.testing.CliRunner`.

**Reference spec:** `docs/superpowers/specs/2026-06-04-stytch-oauth-cli-auth-design.md`

---

## File Structure

- Create `assemblyai_cli/auth/__init__.py` — package marker, re-exports `run_login_flow`.
- Create `assemblyai_cli/auth/endpoints.py` — all config constants (env-overridable) + URL helpers.
- Create `assemblyai_cli/auth/discovery.py` — build the Stytch B2B OAuth discovery start URL.
- Create `assemblyai_cli/auth/loopback.py` — fixed-port HTTP server that captures the callback token.
- Create `assemblyai_cli/auth/ams.py` — thin httpx client for the AMS endpoints.
- Create `assemblyai_cli/auth/flow.py` — orchestration: browser → loopback → AMS → api_key.
- Modify `assemblyai_cli/commands/login.py` — `login` runs the OAuth flow (keeps `--api-key` escape hatch); `logout`/`whoami` unchanged in behavior.
- Modify `pyproject.toml` — add `httpx` as a direct dependency.
- Create `tests/test_auth_endpoints.py`, `tests/test_auth_discovery.py`, `tests/test_auth_loopback.py`, `tests/test_auth_ams.py`, `tests/test_auth_flow.py`, and extend `tests/test_login.py`.

**Verified setup (sandbox):** redirect `http://127.0.0.1:8585/callback` registered; Google provider live; public token `public-token-test-79ad7d8d-09df-495e-8eb8-72d18efdafa4`; AMS `https://ams.sandbox000.assemblyai-labs.com`. Defaults below match these so the flow runs against sandbox out of the box.

---

## Task 1: Dependency + package skeleton + endpoints

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Create: `assemblyai_cli/auth/__init__.py`
- Create: `assemblyai_cli/auth/endpoints.py`
- Test: `tests/test_auth_endpoints.py`

- [ ] **Step 1: Add httpx as a direct dependency**

In `pyproject.toml`, inside the `dependencies = [ ... ]` list, add the line (after `"keyring>=25.7.0",`):

```toml
    "httpx>=0.28.1",
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves with `httpx` now a direct dep (it was already present transitively); no errors.

- [ ] **Step 3: Write the failing test for endpoints**

Create `tests/test_auth_endpoints.py`:

```python
import importlib

from assemblyai_cli.auth import endpoints


def test_redirect_uri_is_fixed_loopback():
    assert endpoints.redirect_uri() == "http://127.0.0.1:8585/callback"


def test_defaults_point_at_sandbox():
    assert endpoints.AMS_BASE_URL == "https://ams.sandbox000.assemblyai-labs.com"
    assert endpoints.STYTCH_OAUTH_PROVIDER == "google"
    assert endpoints.CLI_TOKEN_NAME == "AssemblyAI CLI"
    assert endpoints.STYTCH_PUBLIC_TOKEN.startswith("public-token-")


def test_env_overrides_are_honored(monkeypatch):
    monkeypatch.setenv("AAI_AUTH_AMS_URL", "http://localhost:8000")
    monkeypatch.setenv("AAI_AUTH_PORT", "9999")
    importlib.reload(endpoints)
    try:
        assert endpoints.AMS_BASE_URL == "http://localhost:8000"
        assert endpoints.redirect_uri() == "http://127.0.0.1:9999/callback"
    finally:
        monkeypatch.delenv("AAI_AUTH_AMS_URL", raising=False)
        monkeypatch.delenv("AAI_AUTH_PORT", raising=False)
        importlib.reload(endpoints)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_auth_endpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: assemblyai_cli.auth`.

- [ ] **Step 5: Create the package marker**

Create `assemblyai_cli/auth/__init__.py`:

```python
from __future__ import annotations

from assemblyai_cli.auth.flow import run_login_flow

__all__ = ["run_login_flow"]
```

(Note: `flow` is created in Task 5. Until then, import it lazily — temporarily make `__init__.py` empty and add the re-export in Task 5. For now write an empty file.)

Replace with an empty file for this task:

```python
```

- [ ] **Step 6: Create endpoints.py**

Create `assemblyai_cli/auth/endpoints.py`:

```python
from __future__ import annotations

import os

# Stytch B2B project (sandbox defaults; override via env for prod).
STYTCH_PROJECT_DOMAIN = os.environ.get(
    "AAI_AUTH_STYTCH_DOMAIN",
    "https://psychedelic-journey-5884.customers.stytch.dev",
)
STYTCH_PUBLIC_TOKEN = os.environ.get(
    "AAI_AUTH_PUBLIC_TOKEN",
    "public-token-test-79ad7d8d-09df-495e-8eb8-72d18efdafa4",
)
STYTCH_OAUTH_PROVIDER = os.environ.get("AAI_AUTH_PROVIDER", "google")

# Fixed loopback (Stytch does exact-match redirect validation; 8585 is registered).
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = int(os.environ.get("AAI_AUTH_PORT", "8585"))
LOOPBACK_PATH = "/callback"

# Accounts Management Service.
AMS_BASE_URL = os.environ.get(
    "AAI_AUTH_AMS_URL", "https://ams.sandbox000.assemblyai-labs.com"
)

CLI_TOKEN_NAME = "AssemblyAI CLI"


def redirect_uri() -> str:
    """The exact loopback redirect URL registered in Stytch."""
    return f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}{LOOPBACK_PATH}"
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/test_auth_endpoints.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock assemblyai_cli/auth/__init__.py assemblyai_cli/auth/endpoints.py tests/test_auth_endpoints.py
git commit -m "feat(auth): add auth package skeleton and endpoint config"
```

---

## Task 2: Discovery start URL builder

**Files:**
- Create: `assemblyai_cli/auth/discovery.py`
- Test: `tests/test_auth_discovery.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_discovery.py`:

```python
from urllib.parse import parse_qs, urlparse

from assemblyai_cli.auth import discovery, endpoints


def test_build_start_url_targets_b2b_discovery_for_provider():
    url = discovery.build_start_url()
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.path == "/v1/b2b/public/oauth/google/discovery/start"
    assert url.startswith(endpoints.STYTCH_PROJECT_DOMAIN)


def test_build_start_url_includes_public_token_and_redirect():
    url = discovery.build_start_url()
    qs = parse_qs(urlparse(url).query)
    assert qs["public_token"] == [endpoints.STYTCH_PUBLIC_TOKEN]
    assert qs["discovery_redirect_url"] == ["http://127.0.0.1:8585/callback"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_auth_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: assemblyai_cli.auth.discovery`.

- [ ] **Step 3: Create discovery.py**

Create `assemblyai_cli/auth/discovery.py`:

```python
from __future__ import annotations

from urllib.parse import urlencode

from assemblyai_cli.auth import endpoints


def build_start_url() -> str:
    """The Stytch B2B OAuth discovery *start* URL the browser opens.

    Client-side endpoint authenticated by the project's public token. After the
    user authenticates with the provider, Stytch redirects to our loopback
    `discovery_redirect_url` with `?stytch_token_type=discovery_oauth&token=...`.
    No custom state is appended: the redirect is exact-match validated and the
    server is loopback-only, single-shot.
    """
    base = (
        f"{endpoints.STYTCH_PROJECT_DOMAIN}"
        f"/v1/b2b/public/oauth/{endpoints.STYTCH_OAUTH_PROVIDER}/discovery/start"
    )
    params = {
        "public_token": endpoints.STYTCH_PUBLIC_TOKEN,
        "discovery_redirect_url": endpoints.redirect_uri(),
    }
    return f"{base}?{urlencode(params)}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_auth_discovery.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/auth/discovery.py tests/test_auth_discovery.py
git commit -m "feat(auth): build Stytch B2B OAuth discovery start URL"
```

---

## Task 3: Loopback callback server

**Files:**
- Create: `assemblyai_cli/auth/loopback.py`
- Test: `tests/test_auth_loopback.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_loopback.py`:

```python
import threading
import time
import urllib.request

from assemblyai_cli.auth import endpoints, loopback


def _hit(path: str) -> None:
    url = f"http://{endpoints.LOOPBACK_HOST}:{endpoints.LOOPBACK_PORT}{path}"
    # Retry briefly until the server thread is bound.
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except Exception:
            time.sleep(0.05)


def test_capture_returns_token_and_type():
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback(timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    _hit("/callback?stytch_token_type=discovery_oauth&token=tok_abc")
    t.join(timeout=5)

    result = result_box["result"]
    assert result.token == "tok_abc"
    assert result.token_type == "discovery_oauth"
    assert result.error is None


def test_capture_times_out_without_callback():
    result = loopback.capture_callback(timeout=0.3)
    assert result.error == "timeout"
    assert result.token is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_auth_loopback.py -v`
Expected: FAIL with `ModuleNotFoundError: assemblyai_cli.auth.loopback`.

- [ ] **Step 3: Create loopback.py**

Create `assemblyai_cli/auth/loopback.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from assemblyai_cli.auth import endpoints

_SUCCESS_HTML = (
    b"<html><body style='font-family:sans-serif'>"
    b"<h2>Signed in.</h2><p>You can close this tab and return to the terminal.</p>"
    b"</body></html>"
)


@dataclass
class CallbackResult:
    token: str | None = None
    token_type: str | None = None
    error: str | None = None


def capture_callback(timeout: float = 120.0) -> CallbackResult:
    """Bind the fixed loopback port, capture one OAuth callback, return its token.

    Returns a CallbackResult; `error="timeout"` if no callback arrives in time.
    """
    result = CallbackResult()
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            if parsed.path != endpoints.LOOPBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            result.token = (qs.get("token") or [None])[0]
            result.token_type = (qs.get("stytch_token_type") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_SUCCESS_HTML)
            done.set()

        def log_message(self, *args: object) -> None:  # silence stderr logging
            pass

    server = HTTPServer((endpoints.LOOPBACK_HOST, endpoints.LOOPBACK_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not done.wait(timeout):
            result.error = "timeout"
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return result
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_auth_loopback.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/auth/loopback.py tests/test_auth_loopback.py
git commit -m "feat(auth): add fixed-port loopback callback server"
```

---

## Task 4: AMS HTTP client

**Files:**
- Create: `assemblyai_cli/auth/ams.py`
- Test: `tests/test_auth_ams.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_ams.py` (uses httpx's `MockTransport` so no network):

```python
import httpx
import pytest

from assemblyai_cli.auth import ams
from assemblyai_cli.errors import APIError, NotAuthenticated


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ams.httpx, "Client", fake_client)


def test_discover_posts_token_and_type(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "organizations": [{"organization_id": "org_1", "organization_name": "Acme"}],
                "email": "a@b.com",
                "intermediate_session_token": "ist_123",
            },
        )

    _patch_transport(monkeypatch, handler)
    out = ams.discover("tok_abc")
    assert out["intermediate_session_token"] == "ist_123"
    assert "/v2/auth/discover" in seen["url"]
    assert "discovery_oauth" in seen["body"]
    assert "tok_abc" in seen["body"]


def test_get_auth_sends_session_cookie(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, json={"id": 42, "email": "a@b.com"})

    _patch_transport(monkeypatch, handler)
    out = ams.get_auth("sess_jwt_xyz")
    assert out["id"] == 42
    assert "stytch_session_jwt=sess_jwt_xyz" in seen["cookie"]


def test_401_raises_not_authenticated(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid credentials"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated):
        ams.discover("bad")


def test_500_raises_api_error_with_detail(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Something went wrong"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "Something went wrong" in str(exc.value)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_auth_ams.py -v`
Expected: FAIL with `ModuleNotFoundError: assemblyai_cli.auth.ams`.

- [ ] **Step 3: Create ams.py**

Create `assemblyai_cli/auth/ams.py`:

```python
from __future__ import annotations

from typing import Any

import httpx

from assemblyai_cli.auth import endpoints
from assemblyai_cli.errors import APIError, NotAuthenticated

_TIMEOUT = 30.0


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:  # noqa: BLE001 - non-JSON error body
        pass
    return resp.text or f"HTTP {resp.status_code}"


def _json_or_raise(resp: httpx.Response) -> Any:
    if resp.status_code in (401, 403):
        raise NotAuthenticated(f"AMS rejected the login ({resp.status_code}): {_detail(resp)}")
    if resp.status_code >= 400:
        raise APIError(f"AMS request failed ({resp.status_code}): {_detail(resp)}")
    return resp.json()


def discover(token: str) -> dict[str, Any]:
    """POST /v2/auth/discover with a discovery_oauth token -> {orgs, email, IST}."""
    with httpx.Client(base_url=endpoints.AMS_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.post(
            "/v2/auth/discover",
            json={"token": token, "token_type": "discovery_oauth"},
        )
    return _json_or_raise(resp)


def exchange(intermediate_session_token: str, organization_id: str) -> dict[str, Any]:
    """POST /v2/auth/exchange -> SignedInResponse {account, session_jwt, session_token}."""
    with httpx.Client(base_url=endpoints.AMS_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.post(
            "/v2/auth/exchange",
            json={
                "intermediate_session_token": intermediate_session_token,
                "organization_id": organization_id,
            },
        )
    return _json_or_raise(resp)


def get_auth(session_jwt: str) -> dict[str, Any]:
    """GET /v1/auth (session cookie) -> account incl. `id`."""
    with httpx.Client(base_url=endpoints.AMS_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.get("/v1/auth", cookies={"stytch_session_jwt": session_jwt})
    return _json_or_raise(resp)


def list_projects(account_id: int, session_jwt: str) -> list[dict[str, Any]]:
    """GET /v1/users/accounts/{id}/projects -> [{project, tokens[]}]."""
    with httpx.Client(base_url=endpoints.AMS_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.get(
            f"/v1/users/accounts/{account_id}/projects",
            cookies={"stytch_session_jwt": session_jwt},
        )
    return _json_or_raise(resp)


def create_token(
    account_id: int, project_id: int, token_name: str, session_jwt: str
) -> dict[str, Any]:
    """POST /v1/users/accounts/{id}/tokens -> TokenSchema incl. `api_key`."""
    with httpx.Client(base_url=endpoints.AMS_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.post(
            f"/v1/users/accounts/{account_id}/tokens",
            json={"project_id": project_id, "token_name": token_name},
            cookies={"stytch_session_jwt": session_jwt},
        )
    return _json_or_raise(resp)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_auth_ams.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/auth/ams.py tests/test_auth_ams.py
git commit -m "feat(auth): add AMS http client (discover/exchange/auth/projects/tokens)"
```

---

## Task 5: Flow orchestration

**Files:**
- Create: `assemblyai_cli/auth/flow.py`
- Modify: `assemblyai_cli/auth/__init__.py`
- Test: `tests/test_auth_flow.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_flow.py`:

```python
import pytest

from assemblyai_cli.auth import flow, loopback
from assemblyai_cli.errors import APIError


def test_find_or_create_reuses_existing_cli_key(monkeypatch):
    projects = [
        {
            "project": {"id": 7},
            "tokens": [
                {"name": "Default Token", "api_key": "sk_default", "is_disabled": False},
                {"name": "AssemblyAI CLI", "api_key": "sk_cli", "is_disabled": False},
            ],
        }
    ]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)
    monkeypatch.setattr(
        flow.ams, "create_token", lambda *a, **k: pytest.fail("should not create")
    )
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_cli"


def test_find_or_create_creates_when_absent(monkeypatch):
    projects = [{"project": {"id": 7}, "tokens": []}]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)

    created = {}

    def fake_create(account_id, project_id, token_name, session_jwt):
        created.update(project_id=project_id, token_name=token_name)
        return {"api_key": "sk_new"}

    monkeypatch.setattr(flow.ams, "create_token", fake_create)
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_new"
    assert created == {"project_id": 7, "token_name": "AssemblyAI CLI"}


def test_run_login_flow_happy_path(monkeypatch):
    opened = {}
    monkeypatch.setattr(flow, "_open_browser", lambda url: opened.setdefault("url", url))
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [{"organization_id": "org_1", "organization_name": "Acme"}],
            "email": "a@b.com",
            "intermediate_session_token": "ist",
        },
    )
    monkeypatch.setattr(
        flow.ams,
        "exchange",
        lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt", "session_token": "t"},
    )
    monkeypatch.setattr(flow.ams, "get_auth", lambda jwt: {"id": 9})
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    assert flow.run_login_flow() == "sk_final"
    assert opened["url"].startswith("https://")


def test_run_login_flow_timeout_raises(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(flow, "_capture", lambda: loopback.CallbackResult(error="timeout"))
    with pytest.raises(APIError, match="timed out"):
        flow.run_login_flow()


def test_run_login_flow_zero_orgs_raises(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {"organizations": [], "email": "a@b.com", "intermediate_session_token": "ist"},
    )
    with pytest.raises(APIError, match="no AssemblyAI organization"):
        flow.run_login_flow()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_auth_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: assemblyai_cli.auth.flow`.

- [ ] **Step 3: Create flow.py**

Create `assemblyai_cli/auth/flow.py`:

```python
from __future__ import annotations

import webbrowser

from assemblyai_cli import output
from assemblyai_cli.auth import ams, discovery, endpoints, loopback
from assemblyai_cli.errors import APIError


def _open_browser(url: str) -> None:
    """Open the system browser, falling back to printing the URL."""
    output.console.print(f"Opening your browser to sign in:\n  {url}")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - opening a browser is best-effort
        output.console.print(
            "[aai.muted]Could not open a browser; open the URL above manually.[/aai.muted]"
        )


def _capture() -> loopback.CallbackResult:
    return loopback.capture_callback()


def find_or_create_cli_key(account_id: int, session_jwt: str) -> str:
    """Return the existing 'AssemblyAI CLI' key, or create one in the first project."""
    projects = ams.list_projects(account_id, session_jwt)
    for entry in projects:
        for token in entry.get("tokens", []):
            if token.get("name") == endpoints.CLI_TOKEN_NAME and not token.get("is_disabled"):
                return str(token["api_key"])
    if not projects:
        raise APIError("Your account has no project to create an API key in.")
    project_id = projects[0]["project"]["id"]
    created = ams.create_token(
        account_id, project_id, endpoints.CLI_TOKEN_NAME, session_jwt
    )
    return str(created["api_key"])


def run_login_flow() -> str:
    """Drive the full browser + AMS login and return a usable AssemblyAI API key."""
    _open_browser(discovery.build_start_url())
    result = _capture()

    if result.error == "timeout":
        raise APIError("Login timed out waiting for the browser. Run 'aai login' again.")
    if result.token_type != "discovery_oauth" or not result.token:
        raise APIError("Login did not return a valid OAuth token. Run 'aai login' again.")

    disc = ams.discover(result.token)
    organizations = disc.get("organizations") or []
    if not organizations:
        raise APIError(
            "Signed in, but found no AssemblyAI organization for this account."
        )
    organization_id = organizations[0]["organization_id"]

    signed_in = ams.exchange(disc["intermediate_session_token"], organization_id)
    session_jwt = signed_in["session_jwt"]

    account = ams.get_auth(session_jwt)
    return find_or_create_cli_key(int(account["id"]), session_jwt)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_auth_flow.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Wire the package re-export**

Replace the contents of `assemblyai_cli/auth/__init__.py` with:

```python
from __future__ import annotations

from assemblyai_cli.auth.flow import run_login_flow

__all__ = ["run_login_flow"]
```

- [ ] **Step 6: Run the auth test suite**

Run: `uv run pytest tests/test_auth_endpoints.py tests/test_auth_discovery.py tests/test_auth_loopback.py tests/test_auth_ams.py tests/test_auth_flow.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add assemblyai_cli/auth/flow.py assemblyai_cli/auth/__init__.py tests/test_auth_flow.py
git commit -m "feat(auth): orchestrate browser+AMS login into an API key"
```

---

## Task 6: Wire the `login` command to the OAuth flow

**Files:**
- Modify: `assemblyai_cli/commands/login.py:17-48` (the `login` command)
- Test: extend `tests/test_login.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_login.py`:

```python
def test_login_oauth_flow_stores_returned_key(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.login.run_login_flow", lambda: "sk_from_oauth"
    )
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"


def test_login_oauth_flow_failure_exits_nonzero(monkeypatch):
    from assemblyai_cli.errors import APIError

    def boom():
        raise APIError("Login timed out waiting for the browser.")

    monkeypatch.setattr("assemblyai_cli.commands.login.run_login_flow", boom)
    result = runner.invoke(app, ["login"])
    assert result.exit_code != 0
    assert config.get_api_key("default") is None


def test_login_api_key_flag_still_bypasses_oauth(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.login.run_login_flow",
        lambda: (_ for _ in ()).throw(AssertionError("OAuth must not run with --api-key")),
    )
    with patch("assemblyai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag2"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_flag2"
```

Note: the existing `test_login_interactive_prompts_when_no_flag` and `test_login_interactive_survives_browser_failure` tests describe the OLD paste-a-key behavior and are replaced by the OAuth flow. Delete those two tests (they patch `webbrowser`/`typer.prompt`, which the new flow doesn't use for interactive login).

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_login.py -v`
Expected: the three new tests FAIL with `AttributeError: ... has no attribute 'run_login_flow'`; the two deleted-old tests should be removed before running.

- [ ] **Step 3: Rewrite the `login` command**

In `assemblyai_cli/commands/login.py`, change the import block near the top to add `run_login_flow` and drop the now-unused `webbrowser`:

Replace:

```python
import webbrowser

import typer
from rich.markup import escape

from assemblyai_cli import client, config, output
from assemblyai_cli.context import AppState, resolve_profile, run_command
from assemblyai_cli.errors import APIError, NotAuthenticated
```

with:

```python
import typer
from rich.markup import escape

from assemblyai_cli import client, config, output
from assemblyai_cli.auth import run_login_flow
from assemblyai_cli.context import AppState, resolve_profile, run_command
from assemblyai_cli.errors import APIError, NotAuthenticated
```

Then replace the `login` command body (the `def body(...)` inside `login`) with:

```python
    """Authenticate via your browser (Stytch); stores a CLI API key."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        if api_key:
            # Non-interactive escape hatch for CI/automation.
            if not client.validate_key(api_key):
                raise APIError("That API key was rejected (HTTP 401). Check it and retry.")
            key = api_key
        else:
            key = run_login_flow()
        config.set_api_key(profile, key)
        output.emit(
            {"authenticated": True, "profile": profile},
            lambda _d: f"[aai.success]Authenticated[/aai.success] on profile '{escape(profile)}'.",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
```

(Keep the `@app.command()` decorator and the `login(ctx, api_key, json_out)` signature exactly as they are; the docstring at the top of `body`'s function moves to the command — leave the existing parameter declarations untouched.)

- [ ] **Step 4: Run the login tests to verify they pass**

Run: `uv run pytest tests/test_login.py -v`
Expected: PASS (the three new tests + the unchanged flag/profile/whoami/logout tests).

- [ ] **Step 5: Run the full suite + linters**

Run: `uv run pytest -q && uv run ruff check assemblyai_cli tests && uv run mypy assemblyai_cli`
Expected: all green. (If mypy flags `httpx` types, they're shipped; no stub install needed.)

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/commands/login.py tests/test_login.py
git commit -m "feat(auth): aai login uses Stytch OAuth, keeps --api-key escape hatch"
```

---

## Task 7: Manual end-to-end verification against sandbox

**Files:** none (manual smoke test; this is the one link not provable in unit tests — see spec P-NEW).

- [ ] **Step 1: Run a real login against sandbox**

Run: `uv run aai login`
Expected: browser opens to Google; after sign-in, the terminal prints `Authenticated on profile 'default'.`

- [ ] **Step 2: Confirm the key works**

Run: `uv run aai whoami --json`
Expected: `reachable: true`, masked `api_key`.

- [ ] **Step 3: Confirm find-or-create is idempotent**

Run `uv run aai login` a second time; in the Stytch/AssemblyAI dashboard confirm only **one** key named `AssemblyAI CLI` exists (the second login reused it).

- [ ] **Step 4: Note any deviations**

If discover returns 0 orgs (brand-new account), `login` errors with "no AssemblyAI organization" — that confirms the org-creation branch (`/v2/auth/organization`) is needed; capture it as a follow-up (out of scope per spec).

---

## Self-Review

- **Spec coverage:** loopback+discovery (Tasks 2–3), AMS chain incl. find-or-create (Tasks 4–5), store-key-only via `config.set_api_key` (Task 6), `--api-key`/env escape hatches retained (Task 6), error handling for timeout/bad-token/0-orgs/401/500 (Tasks 4–5), config constants env-overridable (Task 1), testing approach (every task). `logout`/`whoami` are intentionally unchanged (spec: behavior unchanged) — no task needed; optional best-effort revoke is a documented follow-up, not built here (YAGNI for v1).
- **Placeholder scan:** none — every code/test step is complete.
- **Type consistency:** `CallbackResult` (loopback) used in flow + tests; `run_login_flow()`/`find_or_create_cli_key()` names match across flow + login + tests; AMS function names (`discover`/`exchange`/`get_auth`/`list_projects`/`create_token`) consistent across `ams.py`, `flow.py`, and tests.

## Known follow-ups (not in this plan)
- 0-orgs → `/v2/auth/organization` create-and-sign-in branch.
- MFA / verify-email responses from `/v2/auth/exchange`.
- `logout` server-side revoke of the CLI token; storing `account_id`/`token_id` for precise revoke.
- Production project/AMS URLs + prod redirect registration (spec P2/O4).
