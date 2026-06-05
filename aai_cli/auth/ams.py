from __future__ import annotations

from typing import Any, cast

import httpx

from aai_cli.auth import endpoints
from aai_cli.errors import APIError, NotAuthenticated

_TIMEOUT = 30.0


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:  # noqa: BLE001,S110 - non-JSON error body; pass is intentional
        pass
    return resp.text or f"HTTP {resp.status_code}"


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise NotAuthenticated(f"AMS rejected the login ({resp.status_code}): {_detail(resp)}")
    if resp.status_code >= 400:
        raise APIError(f"AMS request failed ({resp.status_code}): {_detail(resp)}")


def _json_or_raise(resp: httpx.Response) -> Any:
    _raise_for_error(resp)
    return resp.json()


def _client(session_jwt: str | None = None) -> httpx.Client:
    """An AMS HTTP client; pass a session JWT to send the authenticated cookie."""
    cookies = {"stytch_session_jwt": session_jwt} if session_jwt else None
    return httpx.Client(base_url=endpoints.ams_base(), timeout=_TIMEOUT, cookies=cookies)


def discover(token: str) -> dict[str, Any]:
    """POST /v2/auth/discover with a discovery_oauth token -> {orgs, email, IST}."""
    with _client() as client:
        resp = client.post(
            "/v2/auth/discover",
            json={"token": token, "token_type": "discovery_oauth"},
        )
    return cast(dict[str, Any], _json_or_raise(resp))


def exchange(intermediate_session_token: str, organization_id: str) -> dict[str, Any]:
    """POST /v2/auth/exchange -> SignedInResponse {account, session_jwt, session_token}."""
    with _client() as client:
        resp = client.post(
            "/v2/auth/exchange",
            json={
                "intermediate_session_token": intermediate_session_token,
                "organization_id": organization_id,
            },
        )
    return cast(dict[str, Any], _json_or_raise(resp))


def list_projects(account_id: int, session_jwt: str) -> list[dict[str, Any]]:
    """GET /v1/users/accounts/{id}/projects -> [{project, tokens[]}]."""
    with _client(session_jwt) as client:
        resp = client.get(f"/v1/users/accounts/{account_id}/projects")
    return cast(list[dict[str, Any]], _json_or_raise(resp))


def create_token(
    account_id: int, project_id: int, token_name: str, session_jwt: str
) -> dict[str, Any]:
    """POST /v1/users/accounts/{id}/tokens -> TokenSchema incl. `api_key`."""
    with _client(session_jwt) as client:
        resp = client.post(
            f"/v1/users/accounts/{account_id}/tokens",
            json={"project_id": project_id, "token_name": token_name},
        )
    return cast(dict[str, Any], _json_or_raise(resp))


def get_balance(session_jwt: str) -> dict[str, Any]:
    """GET /v2/billing/balance -> {account_id, balance_in_cents, ...}."""
    with _client(session_jwt) as client:
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
    with _client(session_jwt) as client:
        resp = client.post("/v2/billing/usage", json=body)
    return cast(dict[str, Any], _json_or_raise(resp))


def get_rate_limits(account_id: int, session_jwt: str) -> dict[str, Any]:
    """GET /v1/users/accounts/{id}/rate-limits -> {rate_limits: [...]}."""
    with _client(session_jwt) as client:
        resp = client.get(f"/v1/users/accounts/{account_id}/rate-limits")
    return cast(dict[str, Any], _json_or_raise(resp))


def rename_token(account_id: int, token_id: int, token_name: str, session_jwt: str) -> None:
    """PUT /v1/users/accounts/{id}/tokens/{token_id}; 200 returns an empty body."""
    with _client(session_jwt) as client:
        resp = client.put(
            f"/v1/users/accounts/{account_id}/tokens/{token_id}",
            json={"token_name": token_name},
        )
    _raise_for_error(resp)


def list_streaming(session_jwt: str, **filters: Any) -> dict[str, Any]:
    """GET /v1/users/streaming -> {page_details, data: [StreamingSessionSchema]}."""
    params = {k: v for k, v in filters.items() if v is not None}
    with _client(session_jwt) as client:
        resp = client.get("/v1/users/streaming", params=params)
    return cast(dict[str, Any], _json_or_raise(resp))


def get_streaming(session_id: str, session_jwt: str) -> dict[str, Any]:
    """GET /v1/users/streaming/{session_id} -> StreamingSessionSchema."""
    with _client(session_jwt) as client:
        resp = client.get(f"/v1/users/streaming/{session_id}")
    return cast(dict[str, Any], _json_or_raise(resp))


def list_audit_logs(session_jwt: str, **filters: Any) -> dict[str, Any]:
    """GET /v2/user/audit-logs -> {page_details, data: [AuditLogResponse]}."""
    params = {k: v for k, v in filters.items() if v is not None}
    with _client(session_jwt) as client:
        resp = client.get("/v2/user/audit-logs", params=params)
    return cast(dict[str, Any], _json_or_raise(resp))
