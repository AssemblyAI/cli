# AMS Account Self-Service Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add account self-service commands to `aai` — `keys` (ls/create/rename), `balance`, `usage`, `limits`, and `sessions` (ls/get) — backed by the AMS API, plus enrich `whoami`.

**Architecture:** Every AMS user endpoint authenticates with the browser-login **Stytch `session_jwt` cookie**, not the API key (verified against the OpenAPI spec — there is no API-key/Bearer scheme). Today `aai login` mints an API key from the session and discards the session. This plan persists the session (`session_jwt` + `session_token` in the keyring, `account_id` in `config.toml`) at browser-login time, exposes new functions on the existing `aai_cli/auth/ams.py` client, adds a `resolve_session()` helper, and adds three command modules that follow the established `transcripts`/`login` patterns. There is **no session-refresh path** in AMS and the CLI holds only Stytch's *public* token, so on an expired session (401) commands raise `NotAuthenticated` telling the user to run `aai login` again. No destructive/delete commands.

**Tech Stack:** Python 3.10+, Typer, Rich, httpx, keyring, pytest, `typer.testing.CliRunner`.

---

## Auth & API facts (verified against the AMS OpenAPI spec)

These shapes are used verbatim in the tasks below. Do not re-derive them.

- **Auth:** all endpoints below take a `stytch_session_jwt` **cookie**. No Bearer/API-key option.
- `POST /v2/auth/exchange` → `SignedInResponse { account: UserAccountResponse, session_jwt: str, session_token: str }`.
- `GET /v1/auth` → `UserAccountResponse` (free-form object; `account["id"]` is the integer account id).
- `GET /v1/users/accounts/{account_id}/projects` → array of `ProjectDetailResponse { project: ProjectSchema, tokens: [TokenSchema] }`.
  - `ProjectSchema { id:int, account_id:int, name:str, is_disabled:bool, created:str, updated:str }`
  - `TokenSchema { id:int, project_id:int, api_key:str, name:str, is_disabled:bool, created:str, updated:str }`
- `POST /v1/users/accounts/{account_id}/tokens` body `CreateTokenRequest { project_id:int, token_name:str }` → `TokenSchema` (incl. `api_key`).
- `PUT /v1/users/accounts/{account_id}/tokens/{token_id}` body `RenameTokenRequest { token_name:str }` → **`200` with empty `{}` body** (do not call `.json()` blindly).
- `GET /v2/billing/balance` → `GetBalanceResponse { account_id:int, metronome_customer_id:str, balance_in_cents:number }`.
- `POST /v2/billing/usage` body `GetUsageRequest { starting_on:str, ending_before:str, window_size?:str }` → `GetUsageResponse { usage_items: [UsageWindow] }`.
  - `UsageWindow { start_timestamp:str, end_timestamp:str, total:number, line_items:[UsageLineItem] }`
- `GET /v1/users/accounts/{account_id}/rate-limits` → `AccountRateLimitsResponse { rate_limits: [AccountRateLimitsDTO] }`.
  - `AccountRateLimitsDTO { account_id:int, service:str, magnitude:number, created:str, updated:str }`
- `GET /v1/users/streaming` (query: `status, limit, offset, project_id, token_id, start, end, …`) → `{ page_details: PageDetailSchema, data: [StreamingSessionSchema] }`.
- `GET /v1/users/streaming/{session_id}` → `StreamingSessionSchema { session_id:str, project_id:int, token_id:int, status:enum(created|completed|error), region?, created_at?, completed_at?, audio_duration_sec?, session_duration_sec?, speech_model?, language_code?, error?, … }`.
- `GET /v2/user/audit-logs` (query: `action_taken, resource_type, actor_type, actor_id, start, end, limit, offset, account_id`) → `ListResponse_AuditLogResponse_ { page_details: PageDetailSchema, data: [AuditLogResponse] }`.
  - `AuditLogResponse { id:int, log_time:str, actor_type:str, actor_id?:int, action_taken:str, resource_type?:str, resource_id?:str, account_id?:int, old_values?:object, new_values?:object, metadata?:object }`

---

## File Structure

**Modify:**
- `aai_cli/config.py` — add session/account-id persistence (`set_session`, `get_session`, `get_account_id`, `clear_session`).
- `aai_cli/auth/flow.py` — `run_login_flow()` returns a `LoginResult` dataclass (api key + session material) instead of a bare key string.
- `aai_cli/auth/ams.py` — add `_raise_for_error` helper + new endpoint functions (`get_balance`, `get_usage`, `get_rate_limits`, `rename_token`, `list_streaming`, `get_streaming`, `list_audit_logs`).
- `aai_cli/commands/login.py` — persist session at browser login; clear it on logout; enrich `whoami`.
- `aai_cli/context.py` — add `resolve_session()` shared by the new commands.
- `aai_cli/main.py` — register the three new typer apps and update `_COMMAND_ORDER`.
- `README.md` — document the new commands.
- `tests/test_login.py` — update mocks for the new `run_login_flow()` return type.

**Create:**
- `aai_cli/commands/keys.py` — `keys` group: `list`, `create`, `rename`.
- `aai_cli/commands/account.py` — top-level `balance`, `usage`, `limits`.
- `aai_cli/commands/sessions.py` — `sessions` group: `list`, `get`.
- `aai_cli/commands/audit.py` — top-level `audit` (lists the current user's audit log).
- `tests/test_config_session.py`, `tests/test_ams_account.py`, `tests/test_keys.py`, `tests/test_account_command.py`, `tests/test_sessions_command.py`, `tests/test_audit_command.py`.

---

## Task 1: Session persistence in config

**Files:**
- Modify: `aai_cli/config.py`
- Test: `tests/test_config_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_session.py`:

```python
from aai_cli import config


def test_set_and_get_session_roundtrips():
    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    assert config.get_session("default") == {"jwt": "jwt_1", "token": "tok_1"}
    assert config.get_account_id("default") == 42


def test_get_session_none_when_absent():
    assert config.get_session("default") is None
    assert config.get_account_id("default") is None


def test_clear_session_removes_jwt_and_account_id():
    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    config.clear_session("default")
    assert config.get_session("default") is None
    assert config.get_account_id("default") is None


def test_clear_session_is_safe_when_absent():
    config.clear_session("never-logged-in")  # must not raise


def test_get_session_returns_none_for_corrupt_blob():
    import keyring

    keyring.set_password(config.KEYRING_SERVICE, "session:default", "not-json")
    assert config.get_session("default") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_session.py -v`
Expected: FAIL with `AttributeError: module 'aai_cli.config' has no attribute 'set_session'`.

- [ ] **Step 3: Implement session persistence**

In `aai_cli/config.py`, add `import json` to the imports (top of file, after `import contextlib`). Then add these functions after `clear_api_key` (around line 101):

```python
SESSION_KEYRING_PREFIX = "session"  # keyring username: f"{prefix}:{profile}"


def _session_username(profile: str) -> str:
    return f"{SESSION_KEYRING_PREFIX}:{profile}"


def set_session(profile: str, *, session_jwt: str, session_token: str, account_id: int) -> None:
    """Persist the browser-login Stytch session (secret) + account id (non-secret).

    AMS self-service endpoints authenticate with this session cookie, not the API
    key. The JWT is short-lived; an expired session surfaces as NotAuthenticated.
    """
    _validate_profile(profile)
    keyring.set_password(
        KEYRING_SERVICE,
        _session_username(profile),
        json.dumps({"jwt": session_jwt, "token": session_token}),
    )
    data = _load()
    data.setdefault("profiles", {}).setdefault(profile, {})["account_id"] = account_id
    _dump(data)


def get_session(profile: str) -> dict[str, str] | None:
    """The stored {'jwt', 'token'} for a profile, or None if absent/corrupt."""
    raw = keyring.get_password(KEYRING_SERVICE, _session_username(profile))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "jwt" not in data:
        return None
    return data


def get_account_id(profile: str) -> int | None:
    """The AMS account id recorded at login for a profile, if any."""
    value = _load().get("profiles", {}).get(profile, {}).get("account_id")
    return int(value) if value is not None else None


def clear_session(profile: str) -> None:
    with contextlib.suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(KEYRING_SERVICE, _session_username(profile))
    data = _load()
    prof = data.get("profiles", {}).get(profile)
    if prof and "account_id" in prof:
        del prof["account_id"]
        _dump(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_session.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/config.py tests/test_config_session.py
git commit -m "feat(config): persist Stytch session + account id per profile"
```

---

## Task 2: `run_login_flow()` returns session material

**Files:**
- Modify: `aai_cli/auth/flow.py`
- Test: `tests/test_auth_flow.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_auth_flow.py`:

```python
def test_run_login_flow_returns_session_material(monkeypatch):
    from aai_cli.auth import flow

    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: flow.loopback.CallbackResult(
            token="tok", token_type="discovery_oauth", error=None
        ),
    )
    monkeypatch.setattr(
        flow.ams, "discover",
        lambda token: {
            "organizations": [{"organization_id": "org_1"}],
            "intermediate_session_token": "ist_1",
        },
    )
    monkeypatch.setattr(
        flow.ams, "exchange",
        lambda ist, org: {"session_jwt": "jwt_1", "session_token": "tok_1",
                          "account": {"id": 99}},
    )
    monkeypatch.setattr(flow.ams, "get_auth", lambda jwt: {"id": 99})
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_key")

    result = flow.run_login_flow()
    assert result.api_key == "sk_key"
    assert result.session_jwt == "jwt_1"
    assert result.session_token == "tok_1"
    assert result.account_id == 99
```

(If `tests/test_auth_flow.py` uses different existing names for `CallbackResult`, match the existing import there; the field names `token`/`token_type`/`error` come from `loopback.CallbackResult` used in `flow.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auth_flow.py::test_run_login_flow_returns_session_material -v`
Expected: FAIL with `AttributeError: 'str' object has no attribute 'api_key'`.

- [ ] **Step 3: Implement the dataclass return**

In `aai_cli/auth/flow.py`, add `from dataclasses import dataclass` to the imports, then add the dataclass after the imports and rewrite `run_login_flow`:

```python
@dataclass
class LoginResult:
    """Everything a successful browser login yields: a usable API key plus the
    Stytch session the AMS self-service commands authenticate with."""

    api_key: str
    session_jwt: str
    session_token: str
    account_id: int
```

Replace the end of `run_login_flow` (from `signed_in = ...` onward) with:

```python
    signed_in = ams.exchange(disc["intermediate_session_token"], organization_id)
    session_jwt = signed_in["session_jwt"]
    session_token = signed_in["session_token"]

    account = ams.get_auth(session_jwt)
    account_id = int(account["id"])
    api_key = find_or_create_cli_key(account_id, session_jwt)
    return LoginResult(
        api_key=api_key,
        session_jwt=session_jwt,
        session_token=session_token,
        account_id=account_id,
    )
```

Update the docstring's return line to: `"""Drive the full browser + AMS login and return a LoginResult."""`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auth_flow.py -v`
Expected: PASS (existing flow tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/auth/flow.py tests/test_auth_flow.py
git commit -m "refactor(auth): run_login_flow returns LoginResult with session"
```

---

## Task 3: Persist session at login, clear on logout

**Files:**
- Modify: `aai_cli/commands/login.py`
- Test: `tests/test_login.py`

- [ ] **Step 1: Update existing tests + add new ones**

In `tests/test_login.py`, the existing OAuth tests mock `run_login_flow` to return a bare string — that no longer matches. Replace those mocks and add coverage. Replace the bodies of `test_login_oauth_flow_stores_returned_key`, `test_login_api_key_flag_still_bypasses_oauth`, `test_login_binds_env_to_profile`, and `test_sandbox_flag_is_shortcut_for_env` so each `run_login_flow` mock returns a `LoginResult`. Add a shared helper at the top of the file (after `runner = CliRunner()`):

```python
from aai_cli.auth.flow import LoginResult


def _fake_login_result(key="sk_from_oauth"):
    return LoginResult(api_key=key, session_jwt="jwt_x", session_token="tok_x", account_id=7)
```

Then change each affected mock from `lambda: "sk_from_oauth"` (or `"sk_x"`) to `lambda: _fake_login_result()` (or `_fake_login_result("sk_x")`). For `test_login_api_key_flag_still_bypasses_oauth`, keep the `throw` lambda unchanged (it must still not run).

Add these new tests:

```python
def test_login_oauth_persists_session(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_session("default") == {"jwt": "jwt_x", "token": "tok_x"}
    assert config.get_account_id("default") == 7


def test_login_api_key_flag_does_not_persist_session():
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_session("default") is None


def test_logout_clears_session(monkeypatch):
    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=7)
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert config.get_session("default") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_login.py -v`
Expected: FAIL — new tests fail (session not persisted/cleared) and the `LoginResult` import path is exercised.

- [ ] **Step 3: Implement persistence in login.py**

In `aai_cli/commands/login.py`, rewrite the `login` command body (the `else` branch + persistence) so the browser path stores the session:

```python
    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        env = environments.active().name
        if api_key:
            # Non-interactive escape hatch for CI/automation: no AMS session is
            # obtained, so account self-service commands won't work for this profile.
            if not client.validate_key(api_key):
                raise APIError("That API key was rejected (HTTP 401). Check it and retry.")
            config.set_api_key(profile, api_key)
            config.set_profile_env(profile, env)
        else:
            result = run_login_flow()
            config.set_api_key(profile, result.api_key)
            config.set_profile_env(profile, env)
            config.set_session(
                profile,
                session_jwt=result.session_jwt,
                session_token=result.session_token,
                account_id=result.account_id,
            )
        output.emit(
            {"authenticated": True, "profile": profile, "env": env},
            lambda _d: f"[aai.success]Authenticated[/aai.success] on profile "
            f"'{escape(profile)}' (env: {escape(env)}).",
            json_mode=json_mode,
        )
```

In the `logout` command body, add session clearing after `config.clear_api_key(profile)`:

```python
        config.clear_api_key(profile)
        config.clear_session(profile)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_login.py -v`
Expected: PASS (all login tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/login.py tests/test_login.py
git commit -m "feat(auth): persist AMS session at browser login, clear on logout"
```

---

## Task 4: `resolve_session()` helper

**Files:**
- Modify: `aai_cli/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_context.py`:

```python
def test_resolve_session_returns_account_and_jwt():
    from aai_cli import config
    from aai_cli.context import AppState, resolve_session

    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    account_id, jwt = resolve_session(AppState())
    assert account_id == 42
    assert jwt == "jwt_1"


def test_resolve_session_raises_when_no_session():
    from aai_cli.context import AppState, resolve_session
    from aai_cli.errors import NotAuthenticated

    import pytest

    with pytest.raises(NotAuthenticated):
        resolve_session(AppState())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_session'`.

- [ ] **Step 3: Implement the helper**

In `aai_cli/context.py`, add after `resolve_environment`:

```python
def resolve_session(state: AppState) -> tuple[int, str]:
    """Account id + Stytch session JWT for AMS self-service commands.

    These endpoints authenticate with the browser-login session, not the API
    key. Raises NotAuthenticated when the active profile has no stored session
    (e.g. it was set up with `aai login --api-key`, or the session expired).
    """
    from aai_cli.errors import NotAuthenticated

    profile = resolve_profile(state)
    session = config.get_session(profile)
    account_id = config.get_account_id(profile)
    if session is None or account_id is None:
        raise NotAuthenticated(
            "These commands need a browser login. Run 'aai login' (without --api-key)."
        )
    return account_id, session["jwt"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_context.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/context.py tests/test_context.py
git commit -m "feat(context): add resolve_session helper for AMS commands"
```

---

## Task 5: AMS client functions

**Files:**
- Modify: `aai_cli/auth/ams.py`
- Test: `tests/test_ams_account.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ams_account.py`:

```python
import httpx
import pytest

from aai_cli.auth import ams
from aai_cli.errors import NotAuthenticated


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ams.httpx, "Client", fake_client)


def test_get_balance_sends_cookie_and_parses(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, json={"account_id": 1, "balance_in_cents": 2500})

    _patch_transport(monkeypatch, handler)
    out = ams.get_balance("jwt_1")
    assert out["balance_in_cents"] == 2500
    assert "/v2/billing/balance" in seen["url"]
    assert "stytch_session_jwt=jwt_1" in seen["cookie"]


def test_get_usage_posts_date_range(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"usage_items": []})

    _patch_transport(monkeypatch, handler)
    ams.get_usage("jwt", "2026-05-01", "2026-06-01", "day")
    assert "2026-05-01" in seen["body"] and "2026-06-01" in seen["body"]
    assert "day" in seen["body"]


def test_get_rate_limits_uses_account_path(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"rate_limits": []})

    _patch_transport(monkeypatch, handler)
    ams.get_rate_limits(42, "jwt")
    assert "/v1/users/accounts/42/rate-limits" in seen["url"]


def test_rename_token_puts_name_and_tolerates_empty_body(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = request.read().decode()
        return httpx.Response(200, content=b"")  # empty body, not JSON

    _patch_transport(monkeypatch, handler)
    ams.rename_token(42, 7, "prod", "jwt")  # must not raise
    assert seen["method"] == "PUT"
    assert "/v1/users/accounts/42/tokens/7" in seen["url"]
    assert "prod" in seen["body"]


def test_list_streaming_passes_filters(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"page_details": {}, "data": []})

    _patch_transport(monkeypatch, handler)
    ams.list_streaming("jwt", limit=5, status="completed")
    assert "limit=5" in seen["url"]
    assert "status=completed" in seen["url"]


def test_get_streaming_by_id(monkeypatch):
    def handler(request):
        assert "/v1/users/streaming/s_1" in str(request.url)
        return httpx.Response(200, json={"session_id": "s_1", "status": "completed"})

    _patch_transport(monkeypatch, handler)
    out = ams.get_streaming("s_1", "jwt")
    assert out["session_id"] == "s_1"


def test_account_4xx_raises_not_authenticated(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"detail": "expired"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated):
        ams.get_balance("jwt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ams_account.py -v`
Expected: FAIL with `AttributeError: module 'aai_cli.auth.ams' has no attribute 'get_balance'`.

- [ ] **Step 3: Implement the new client functions**

In `aai_cli/auth/ams.py`, first refactor the error check out of `_json_or_raise` so non-JSON responses (the empty `rename` body) can reuse it. Replace `_json_or_raise` with:

```python
def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise NotAuthenticated(f"AMS rejected the login ({resp.status_code}): {_detail(resp)}")
    if resp.status_code >= 400:
        raise APIError(f"AMS request failed ({resp.status_code}): {_detail(resp)}")


def _json_or_raise(resp: httpx.Response) -> Any:
    _raise_for_error(resp)
    return resp.json()
```

Then append these functions at the end of the file:

```python
def get_balance(session_jwt: str) -> dict[str, Any]:
    """GET /v2/billing/balance -> {account_id, balance_in_cents, ...}."""
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get("/v2/billing/balance")
    return cast(dict[str, Any], _json_or_raise(resp))


def get_usage(
    session_jwt: str,
    starting_on: str,
    ending_before: str,
    window_size: str | None = None,
) -> dict[str, Any]:
    """POST /v2/billing/usage (ISO dates) -> {usage_items: [...]}."""
    body: dict[str, Any] = {"starting_on": starting_on, "ending_before": ending_before}
    if window_size:
        body["window_size"] = window_size
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.post("/v2/billing/usage", json=body)
    return cast(dict[str, Any], _json_or_raise(resp))


def get_rate_limits(account_id: int, session_jwt: str) -> dict[str, Any]:
    """GET /v1/users/accounts/{id}/rate-limits -> {rate_limits: [...]}."""
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get(f"/v1/users/accounts/{account_id}/rate-limits")
    return cast(dict[str, Any], _json_or_raise(resp))


def rename_token(account_id: int, token_id: int, token_name: str, session_jwt: str) -> None:
    """PUT /v1/users/accounts/{id}/tokens/{token_id}; 200 returns an empty body."""
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.put(
            f"/v1/users/accounts/{account_id}/tokens/{token_id}",
            json={"token_name": token_name},
        )
    _raise_for_error(resp)


def list_streaming(session_jwt: str, **filters: Any) -> dict[str, Any]:
    """GET /v1/users/streaming -> {page_details, data: [StreamingSessionSchema]}."""
    params = {k: v for k, v in filters.items() if v is not None}
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get("/v1/users/streaming", params=params)
    return cast(dict[str, Any], _json_or_raise(resp))


def get_streaming(session_id: str, session_jwt: str) -> dict[str, Any]:
    """GET /v1/users/streaming/{session_id} -> StreamingSessionSchema."""
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get(f"/v1/users/streaming/{session_id}")
    return cast(dict[str, Any], _json_or_raise(resp))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ams_account.py tests/test_auth_ams.py -v`
Expected: PASS (new tests + existing ams tests still green after the `_json_or_raise` refactor).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/auth/ams.py tests/test_ams_account.py
git commit -m "feat(ams): add balance, usage, rate-limits, streaming, rename client fns"
```

---

## Task 6: `keys` command (list / create / rename)

**Files:**
- Create: `aai_cli/commands/keys.py`
- Test: `tests/test_keys.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_keys.py`:

```python
import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def test_keys_list_flattens_tokens():
    _auth()
    projects = [
        {
            "project": {"id": 1, "name": "Default"},
            "tokens": [
                {"id": 10, "name": "ci", "api_key": "sk_abcdef1234", "is_disabled": False}
            ],
        }
    ]
    with patch("aai_cli.commands.keys.ams.list_projects", return_value=projects):
        result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["id"] == 10
    assert "sk_abcdef1234" not in result.output  # api key is masked


def test_keys_list_requires_session():
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 2


def test_keys_create_prints_new_key():
    _auth()
    projects = [{"project": {"id": 1, "name": "Default"}, "tokens": []}]
    created = {"id": 11, "project_id": 1, "name": "ci", "api_key": "sk_newkey9999",
               "is_disabled": False}
    with patch("aai_cli.commands.keys.ams.list_projects", return_value=projects), patch(
        "aai_cli.commands.keys.ams.create_token", return_value=created
    ) as create:
        result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 0
    assert "sk_newkey9999" in result.output
    create.assert_called_once_with(42, 1, "ci", "jwt")


def test_keys_rename_calls_ams():
    _auth()
    with patch("aai_cli.commands.keys.ams.rename_token") as rename:
        result = runner.invoke(app, ["keys", "rename", "10", "prod"])
    assert result.exit_code == 0
    rename.assert_called_once_with(42, 10, "prod", "jwt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keys.py -v`
Expected: FAIL — `keys` is not a registered command (Typer exits non-zero / usage error).

- [ ] **Step 3: Create the command module**

Create `aai_cli/commands/keys.py`:

```python
from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import output
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command
from aai_cli.errors import APIError

app = typer.Typer(help="List, create, and rename your AssemblyAI API keys.", no_args_is_help=True)


def _mask(key: str) -> str:
    return f"{key[:3]}…{key[-4:]}" if len(key) > 7 else "***"


@app.command(name="list")
def list_(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List API keys across your projects (keys shown masked)."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        projects = ams.list_projects(account_id, jwt)
        rows: list[dict[str, object]] = []
        for entry in projects:
            project_name = entry["project"]["name"]
            for token in entry.get("tokens", []):
                rows.append(
                    {
                        "id": token["id"],
                        "name": token["name"],
                        "project": project_name,
                        "key": _mask(str(token["api_key"])),
                        "disabled": token["is_disabled"],
                    }
                )

        def render(data: list[dict[str, object]]) -> Table:
            table = Table("id", "name", "project", "key", "disabled", header_style="aai.heading")
            for row in data:
                table.add_row(
                    str(row["id"]),
                    escape(str(row["name"])),
                    escape(str(row["project"])),
                    escape(str(row["key"])),
                    "yes" if row["disabled"] else "no",
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command()
def create(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="A label for the new key."),
    project_id: int = typer.Option(
        None, "--project", help="Project id to create the key in (defaults to your first)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a new API key. Prints the key value once — copy it now."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        pid = project_id
        if pid is None:
            projects = ams.list_projects(account_id, jwt)
            if not projects:
                raise APIError("Your account has no project to create a key in.")
            pid = projects[0]["project"]["id"]
        created = ams.create_token(account_id, pid, name, jwt)
        output.emit(
            created,
            lambda d: f"Created key '[aai.success]{escape(name)}[/aai.success]': "
            f"{escape(str(d['api_key']))}",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command()
def rename(
    ctx: typer.Context,
    token_id: int = typer.Argument(..., help="The key id (see `aai keys list`)."),
    new_name: str = typer.Argument(..., help="The new label."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Rename an existing API key."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        ams.rename_token(account_id, token_id, new_name, jwt)
        output.emit(
            {"id": token_id, "name": new_name},
            lambda d: f"Renamed key {d['id']} to '{escape(new_name)}'.",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Register and run tests**

Add to `aai_cli/main.py` imports (inside the `from aai_cli.commands import (...)` block, keeping alphabetical-ish order): add `keys,` after `doctor,`. Then after `app.add_typer(claude.app, name="claude")` add:

```python
app.add_typer(keys.app, name="keys")
```

Run: `pytest tests/test_keys.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/keys.py aai_cli/main.py tests/test_keys.py
git commit -m "feat(keys): add 'aai keys' list/create/rename commands"
```

---

## Task 7: `balance`, `usage`, `limits` commands

**Files:**
- Create: `aai_cli/commands/account.py`
- Test: `tests/test_account_command.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_account_command.py`:

```python
import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def test_balance_formats_dollars():
    _auth()
    with patch("aai_cli.commands.account.ams.get_balance",
               return_value={"account_id": 42, "balance_in_cents": 2575}):
        result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    assert "$25.75" in result.output


def test_balance_requires_session():
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 2


def test_usage_defaults_date_range_and_renders(monkeypatch):
    _auth()
    captured = {}

    def fake_usage(jwt, start, end, window):
        captured["start"], captured["end"] = start, end
        return {"usage_items": [
            {"start_timestamp": "2026-05-01", "end_timestamp": "2026-05-02",
             "total": 12.5, "line_items": []}
        ]}

    with patch("aai_cli.commands.account.ams.get_usage", side_effect=fake_usage):
        result = runner.invoke(app, ["usage", "--json"])
    assert result.exit_code == 0
    # both bounds are ISO dates (YYYY-MM-DD), defaulted when not passed
    assert len(captured["start"]) == 10 and len(captured["end"]) == 10
    data = json.loads(result.output)
    assert data["usage_items"][0]["total"] == 12.5


def test_usage_passes_explicit_dates():
    _auth()
    with patch("aai_cli.commands.account.ams.get_usage",
               return_value={"usage_items": []}) as get_usage:
        result = runner.invoke(app, ["usage", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 0
    get_usage.assert_called_once_with("jwt", "2026-01-01", "2026-02-01", None)


def test_limits_renders_services():
    _auth()
    with patch("aai_cli.commands.account.ams.get_rate_limits",
               return_value={"rate_limits": [{"service": "transcript", "magnitude": 200}]}):
        result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "transcript" in result.output and "200" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_account_command.py -v`
Expected: FAIL — `balance`/`usage`/`limits` are not registered commands.

- [ ] **Step 3: Create the command module**

Create `aai_cli/commands/account.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import output
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command

app = typer.Typer(help="Account billing, usage, and limits.")


@app.command()
def balance(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show your remaining account balance."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        data = ams.get_balance(jwt)
        cents = data.get("balance_in_cents", 0) or 0
        output.emit(
            data,
            lambda _d: f"Balance: [aai.success]${cents / 100:,.2f}[/aai.success]",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command()
def usage(
    ctx: typer.Context,
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD). Default: 30d ago."),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD). Default: today."),
    window: str = typer.Option(None, "--window", help="Window size, e.g. 'day' or 'month'."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show usage over a date range (defaults to the last 30 days)."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        today = datetime.now(timezone.utc).date()
        end_date = end or today.isoformat()
        start_date = start or (today - timedelta(days=30)).isoformat()
        data = ams.get_usage(jwt, start_date, end_date, window)

        def render(d: dict) -> Table:
            table = Table("window start", "window end", "total", header_style="aai.heading")
            for item in d.get("usage_items", []):
                table.add_row(
                    escape(str(item["start_timestamp"])),
                    escape(str(item["end_timestamp"])),
                    f"{item['total']:,}",
                )
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command()
def limits(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show your account's rate limits per service."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        data = ams.get_rate_limits(account_id, jwt)

        def render(d: dict) -> Table:
            table = Table("service", "limit", header_style="aai.heading")
            for limit in d.get("rate_limits", []):
                table.add_row(escape(str(limit["service"])), f"{limit['magnitude']:,}")
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Register and run tests**

In `aai_cli/main.py`, add `account,` to the `from aai_cli.commands import (...)` block (before `agent,`). After the `app.add_typer(llm.app)` line add:

```python
app.add_typer(account.app)  # balance, usage, limits
```

Run: `pytest tests/test_account_command.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/account.py aai_cli/main.py tests/test_account_command.py
git commit -m "feat(account): add 'aai balance', 'aai usage', 'aai limits'"
```

---

## Task 8: `sessions` command (list / get)

**Files:**
- Create: `aai_cli/commands/sessions.py`
- Test: `tests/test_sessions_command.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sessions_command.py`:

```python
import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def test_sessions_list_renders_rows():
    _auth()
    payload = {"page_details": {"has_more": False},
               "data": [{"session_id": "s_1", "status": "completed",
                         "created_at": "2026-06-01", "audio_duration_sec": 12.0,
                         "speech_model": "universal"}]}
    with patch("aai_cli.commands.sessions.ams.list_streaming", return_value=payload):
        result = runner.invoke(app, ["sessions", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["session_id"] == "s_1"


def test_sessions_list_passes_status_filter():
    _auth()
    with patch("aai_cli.commands.sessions.ams.list_streaming",
               return_value={"data": []}) as list_streaming:
        result = runner.invoke(app, ["sessions", "list", "--status", "error", "--limit", "5"])
    assert result.exit_code == 0
    list_streaming.assert_called_once_with("jwt", limit=5, status="error")


def test_sessions_get_renders_detail():
    _auth()
    detail = {"session_id": "s_1", "status": "completed", "speech_model": "universal",
              "audio_duration_sec": 30.0, "error": None}
    with patch("aai_cli.commands.sessions.ams.get_streaming", return_value=detail):
        result = runner.invoke(app, ["sessions", "get", "s_1"])
    assert result.exit_code == 0
    assert "s_1" in result.output and "universal" in result.output


def test_sessions_requires_session():
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sessions_command.py -v`
Expected: FAIL — `sessions` is not a registered command.

- [ ] **Step 3: Create the command module**

Create `aai_cli/commands/sessions.py`:

```python
from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from aai_cli import output, theme
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command

app = typer.Typer(help="Browse your past streaming (real-time) sessions.", no_args_is_help=True)

# Fields shown by `sessions get`, in display order.
_DETAIL_FIELDS = (
    "session_id",
    "status",
    "region",
    "created_at",
    "completed_at",
    "audio_duration_sec",
    "session_duration_sec",
    "speech_model",
    "language_code",
    "error",
)


@app.command(name="list")
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many sessions to show."),
    status: str = typer.Option(None, "--status", help="Filter: created, completed, or error."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent streaming sessions."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        payload = ams.list_streaming(jwt, limit=limit, status=status)
        rows = payload.get("data", [])

        def render(data: list[dict]) -> Table:
            table = Table(
                "session id", "status", "created", "audio (s)", "model",
                header_style="aai.heading",
            )
            for s in data:
                status_str = str(s["status"])
                table.add_row(
                    escape(str(s["session_id"])),
                    Text(status_str, style=theme.status_style(status_str)),
                    escape(str(s.get("created_at") or "")),
                    escape(str(s.get("audio_duration_sec") or "")),
                    escape(str(s.get("speech_model") or "")),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command()
def get(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Streaming session id."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show details for one streaming session."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        data = ams.get_streaming(session_id, jwt)

        def render(d: dict) -> Table:
            table = Table(show_header=False)
            for field in _DETAIL_FIELDS:
                value = d.get(field)
                table.add_row(field, escape("" if value is None else str(value)))
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Register and run tests**

In `aai_cli/main.py`, add `sessions,` to the `from aai_cli.commands import (...)` block. After the `app.add_typer(transcripts.app, name="transcripts")` line add:

```python
app.add_typer(sessions.app, name="sessions")
```

Run: `pytest tests/test_sessions_command.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/sessions.py aai_cli/main.py tests/test_sessions_command.py
git commit -m "feat(sessions): add 'aai sessions' list/get for streaming history"
```

---

## Task 9: `audit` command

**Files:**
- Modify: `aai_cli/auth/ams.py` (add `list_audit_logs`)
- Create: `aai_cli/commands/audit.py`
- Test: `tests/test_ams_account.py`, `tests/test_audit_command.py`

- [ ] **Step 1: Write the failing client test**

Append to `tests/test_ams_account.py`:

```python
def test_list_audit_logs_passes_filters(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, json={"page_details": {}, "data": []})

    _patch_transport(monkeypatch, handler)
    ams.list_audit_logs("jwt", limit=5, action_taken="token.create")
    assert "/v2/user/audit-logs" in seen["url"]
    assert "limit=5" in seen["url"]
    assert "action_taken=token.create" in seen["url"]
    assert "stytch_session_jwt=jwt" in seen["cookie"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ams_account.py::test_list_audit_logs_passes_filters -v`
Expected: FAIL with `AttributeError: module 'aai_cli.auth.ams' has no attribute 'list_audit_logs'`.

- [ ] **Step 3: Add the client function**

Append to `aai_cli/auth/ams.py`:

```python
def list_audit_logs(session_jwt: str, **filters: Any) -> dict[str, Any]:
    """GET /v2/user/audit-logs -> {page_details, data: [AuditLogResponse]}."""
    params = {k: v for k, v in filters.items() if v is not None}
    with httpx.Client(
        base_url=endpoints.ams_base(),
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get("/v2/user/audit-logs", params=params)
    return cast(dict[str, Any], _json_or_raise(resp))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ams_account.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing command test**

Create `tests/test_audit_command.py`:

```python
import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def test_audit_renders_rows():
    _auth()
    payload = {
        "page_details": {"has_more": False},
        "data": [
            {
                "id": 1,
                "log_time": "2026-06-01T12:00:00Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "token.create",
                "resource_type": "token",
                "resource_id": "10",
            }
        ],
    }
    with patch("aai_cli.commands.audit.ams.list_audit_logs", return_value=payload):
        result = runner.invoke(app, ["audit", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["action_taken"] == "token.create"


def test_audit_passes_filters():
    _auth()
    with patch("aai_cli.commands.audit.ams.list_audit_logs",
               return_value={"data": []}) as list_logs:
        result = runner.invoke(app, ["audit", "--limit", "5", "--action", "token.create"])
    assert result.exit_code == 0
    list_logs.assert_called_once_with("jwt", limit=5, action_taken="token.create")


def test_audit_human_mode_renders_table():
    _auth()
    payload = {"data": [
        {"id": 1, "log_time": "2026-06-01T12:00:00Z", "actor_type": "member",
         "action_taken": "login", "resource_type": None}
    ]}
    with patch("aai_cli.commands.audit.ams.list_audit_logs", return_value=payload):
        result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "login" in result.output


def test_audit_requires_session():
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 2
```

- [ ] **Step 6: Run command test to verify it fails**

Run: `pytest tests/test_audit_command.py -v`
Expected: FAIL — `audit` is not a registered command.

- [ ] **Step 7: Create the command module**

Create `aai_cli/commands/audit.py`:

```python
from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import output
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command

app = typer.Typer(help="View your account's audit log.")


@app.command()
def audit(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", help="How many entries to show."),
    action: str = typer.Option(None, "--action", help="Filter by action_taken."),
    resource: str = typer.Option(None, "--resource", help="Filter by resource_type."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent audit-log entries for your account."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        payload = ams.list_audit_logs(
            jwt, limit=limit, action_taken=action, resource_type=resource
        )
        rows = payload.get("data", [])

        def render(data: list[dict]) -> Table:
            table = Table(
                "time", "actor", "action", "resource", header_style="aai.heading"
            )
            for entry in data:
                actor = str(entry["actor_type"])
                actor_id = entry.get("actor_id")
                if actor_id is not None:
                    actor = f"{actor}:{actor_id}"
                resource_label = entry.get("resource_type") or ""
                resource_id = entry.get("resource_id")
                if resource_label and resource_id:
                    resource_label = f"{resource_label}:{resource_id}"
                table.add_row(
                    escape(str(entry["log_time"])),
                    escape(actor),
                    escape(str(entry["action_taken"])),
                    escape(str(resource_label)),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 8: Register and run tests**

In `aai_cli/main.py`, add `audit,` to the `from aai_cli.commands import (...)` block. After the `app.add_typer(sessions.app, name="sessions")` line (from Task 8) add:

```python
app.add_typer(audit.app)  # audit
```

> NOTE: `audit.app` has a single command named `audit`; registering the sub-typer with no `name=` merges it as the top-level `aai audit` command (same mechanism as `account.app`'s `balance`/`usage`/`limits`).

Run: `pytest tests/test_audit_command.py tests/test_ams_account.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add aai_cli/auth/ams.py aai_cli/commands/audit.py aai_cli/main.py tests/test_ams_account.py tests/test_audit_command.py
git commit -m "feat(audit): add 'aai audit' to list account audit-log entries"
```

---

## Task 10: Enrich `whoami` + finalize help ordering

**Files:**
- Modify: `aai_cli/commands/login.py`
- Modify: `aai_cli/main.py` (`_COMMAND_ORDER`)
- Test: `tests/test_login.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_login.py`:

```python
def test_whoami_reports_session_and_account():
    import json

    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=77)
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["account_id"] == 77
    assert data["session"] == "stored"


def test_whoami_session_none_without_browser_login():
    import json

    config.set_api_key("default", "sk_1234567890")
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["session"] == "none"
    assert data["account_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_login.py::test_whoami_reports_session_and_account -v`
Expected: FAIL with `KeyError: 'account_id'`.

- [ ] **Step 3: Enrich whoami**

In `aai_cli/commands/login.py`, update the `whoami` body to include session/account info:

```python
    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        key = config.get_api_key(profile)
        if not key:
            raise NotAuthenticated()
        masked = f"{key[:3]}…{key[-4:]}" if len(key) > 7 else "***"
        env = environments.active().name
        reachable = client.validate_key(key)
        session = config.get_session(profile)
        account_id = config.get_account_id(profile)
        output.emit(
            {
                "profile": profile,
                "env": env,
                "api_key": masked,
                "reachable": reachable,
                "account_id": account_id,
                "session": "stored" if session else "none",
            },
            lambda _d: f"profile={escape(profile)} env={escape(env)} "
            f"key={escape(masked)} reachable={reachable} "
            f"account={account_id} session={'stored' if session else 'none'}",
            json_mode=json_mode,
        )
```

- [ ] **Step 4: Update help ordering**

In `aai_cli/main.py`, update `_COMMAND_ORDER` to place the new commands sensibly (account group after `llm`, `keys`/`sessions` near related commands):

```python
_COMMAND_ORDER = (
    "transcribe",
    "stream",
    "transcripts",
    "sessions",
    "agent",
    "llm",
    "balance",
    "usage",
    "limits",
    "keys",
    "audit",
    "login",
    "logout",
    "whoami",
    "doctor",
    "samples",
    "claude",
    "version",
)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_login.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aai_cli/commands/login.py aai_cli/main.py tests/test_login.py
git commit -m "feat(whoami): report account id + session; order new commands in help"
```

---

## Task 11: README docs + full suite + smoke check

**Files:**
- Modify: `README.md`
- Test: full suite

- [ ] **Step 1: Document the new commands**

In `README.md`, after the existing command/usage section, add a subsection:

```markdown
## Account self-service

These commands use your browser login session (run `aai login` without
`--api-key`), not your API key:

```sh
aai keys list                       # list API keys (masked) across projects
aai keys create --name ci-pipeline  # mint a new key (printed once)
aai keys rename 123 "prod"          # relabel a key

aai balance                         # remaining account balance
aai usage --start 2026-05-01 --end 2026-06-01
aai limits                          # rate limits per service

aai sessions list --status completed
aai sessions get <session-id>       # one streaming session's details

aai audit --limit 20                # recent account audit-log entries
aai audit --action token.create     # filter by action
```

If a command reports it needs a browser login, your session has expired — run
`aai login` again. (AMS sessions are short-lived and cannot be refreshed
silently.)
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests, including the new modules and updated login tests).

- [ ] **Step 3: Lint / type-check (match the repo's tooling)**

Run: `ruff check aai_cli tests && ruff format --check aai_cli tests`
Expected: clean. (If the repo uses a different command — check `pyproject.toml`/`Makefile` — run that instead. Fix any findings.)

- [ ] **Step 4: Smoke-check the help output**

Run: `python -m aai_cli --help`
Expected: `balance`, `usage`, `limits`, `keys`, `audit`, `sessions` appear in the command list in the order defined by `_COMMAND_ORDER`.

Run: `python -m aai_cli keys --help` and `python -m aai_cli sessions --help`
Expected: subcommands `list`, `create`, `rename` and `list`, `get` respectively.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document account self-service commands"
```

---

## Notes & open items for the implementer

- **Units of `usage.total` and `balance`:** `balance_in_cents` is unambiguous (rendered as dollars). `UsageWindow.total` has no documented unit in the spec — it is rendered as a raw number. If product confirms a unit, format it then.
- **Expired sessions:** surface as `NotAuthenticated` (exit 2) via `ams._raise_for_error` on 401/403. The message already tells the user to run `aai login`. This is the agreed v1 behavior (persist + re-login on expiry); no silent refresh.
- **`--api-key` logins** never get a session, so account self-service commands will (correctly) report they need a browser login. This is expected and documented.
- **Reused existing code:** `ams.list_projects` and `ams.create_token` already exist (used by the login flow) — `keys` reuses them rather than adding duplicates.
```