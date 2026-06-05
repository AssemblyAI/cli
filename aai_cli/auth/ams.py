from __future__ import annotations

import httpx2 as httpx
from pydantic import TypeAdapter, ValidationError

from aai_cli.auth import endpoints
from aai_cli.errors import APIError, NotAuthenticated

_TIMEOUT = 30.0
_HTTP_ERROR_MIN_STATUS = 400
_JSON_OBJECT: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_JSON_OBJECTS: TypeAdapter[list[dict[str, object]]] = TypeAdapter(list[dict[str, object]])


def _detail(resp: httpx.Response) -> str:
    fallback = resp.text or f"HTTP {resp.status_code}"
    try:
        body: object = resp.json()
        mapping = _JSON_OBJECT.validate_python(body)
        if "detail" in mapping:
            return str(mapping["detail"])
    except (TypeError, ValueError, ValidationError):
        return fallback
    return fallback


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise NotAuthenticated(f"AMS rejected the login ({resp.status_code}): {_detail(resp)}")
    if resp.status_code >= _HTTP_ERROR_MIN_STATUS:
        raise APIError(f"AMS request failed ({resp.status_code}): {_detail(resp)}")


def _json_or_raise(resp: httpx.Response) -> object:
    _raise_for_error(resp)
    data: object = resp.json()
    return data


def _json_object_or_raise(resp: httpx.Response) -> dict[str, object]:
    data = _json_or_raise(resp)
    try:
        return _JSON_OBJECT.validate_python(data)
    except ValidationError as exc:
        raise APIError(
            f"AMS request returned unexpected JSON: expected object, got {type(data).__name__}."
        ) from exc


def _json_object_list_or_raise(resp: httpx.Response) -> list[dict[str, object]]:
    data = _json_or_raise(resp)
    try:
        return _JSON_OBJECTS.validate_python(data)
    except ValidationError as exc:
        raise APIError(
            f"AMS request returned unexpected JSON: expected list of objects, got {type(data).__name__}."
        ) from exc


def _client(session_jwt: str | None = None) -> httpx.Client:
    """An AMS HTTP client; pass a session JWT to send the authenticated cookie."""
    cookies = {"stytch_session_jwt": session_jwt} if session_jwt else None
    return httpx.Client(base_url=endpoints.ams_base(), timeout=_TIMEOUT, cookies=cookies)


def discover(token: str) -> dict[str, object]:
    """POST /v2/auth/discover with a discovery_oauth token -> {orgs, email, IST}."""
    with _client() as client:
        resp = client.post(
            "/v2/auth/discover",
            json={"token": token, "token_type": "discovery_oauth"},
        )
    return _json_object_or_raise(resp)


def exchange(intermediate_session_token: str, organization_id: str) -> dict[str, object]:
    """POST /v2/auth/exchange -> SignedInResponse {account, session_jwt, session_token}."""
    with _client() as client:
        resp = client.post(
            "/v2/auth/exchange",
            json={
                "intermediate_session_token": intermediate_session_token,
                "organization_id": organization_id,
            },
        )
    return _json_object_or_raise(resp)


def list_projects(account_id: int, session_jwt: str) -> list[dict[str, object]]:
    """GET /v1/users/accounts/{id}/projects -> [{project, tokens[]}]."""
    with _client(session_jwt) as client:
        resp = client.get(f"/v1/users/accounts/{account_id}/projects")
    return _json_object_list_or_raise(resp)


def create_token(
    account_id: int, project_id: int, token_name: str, session_jwt: str
) -> dict[str, object]:
    """POST /v1/users/accounts/{id}/tokens -> TokenSchema incl. `api_key`."""
    with _client(session_jwt) as client:
        resp = client.post(
            f"/v1/users/accounts/{account_id}/tokens",
            json={"project_id": project_id, "token_name": token_name},
        )
    return _json_object_or_raise(resp)


def get_balance(session_jwt: str) -> dict[str, object]:
    """GET /v2/billing/balance -> {account_id, balance_in_cents, ...}."""
    with _client(session_jwt) as client:
        resp = client.get("/v2/billing/balance")
    return _json_object_or_raise(resp)


def get_usage(
    session_jwt: str,
    starting_on: str,
    ending_before: str,
    window_size: str | None = None,
) -> dict[str, object]:
    """POST /v2/billing/usage (ISO dates) -> {usage_items: [...]}."""
    body: dict[str, object] = {"starting_on": starting_on, "ending_before": ending_before}
    if window_size:
        body["window_size"] = window_size
    with _client(session_jwt) as client:
        resp = client.post("/v2/billing/usage", json=body)
    return _json_object_or_raise(resp)


def get_rate_limits(account_id: int, session_jwt: str) -> dict[str, object]:
    """GET /v1/users/accounts/{id}/rate-limits -> {rate_limits: [...]}."""
    with _client(session_jwt) as client:
        resp = client.get(f"/v1/users/accounts/{account_id}/rate-limits")
    return _json_object_or_raise(resp)


def rename_token(account_id: int, token_id: int, token_name: str, session_jwt: str) -> None:
    """PUT /v1/users/accounts/{id}/tokens/{token_id}; 200 returns an empty body."""
    with _client(session_jwt) as client:
        resp = client.put(
            f"/v1/users/accounts/{account_id}/tokens/{token_id}",
            json={"token_name": token_name},
        )
    _raise_for_error(resp)


def list_streaming(session_jwt: str, **filters: object) -> dict[str, object]:
    """GET /v1/users/streaming -> {page_details, data: [StreamingSessionSchema]}."""
    params = {
        key: value for key, value in filters.items() if isinstance(value, str | int | float | bool)
    }
    with _client(session_jwt) as client:
        resp = client.get("/v1/users/streaming", params=params)
    return _json_object_or_raise(resp)


def get_streaming(session_id: str, session_jwt: str) -> dict[str, object]:
    """GET /v1/users/streaming/{session_id} -> StreamingSessionSchema."""
    with _client(session_jwt) as client:
        resp = client.get(f"/v1/users/streaming/{session_id}")
    return _json_object_or_raise(resp)


def list_audit_logs(session_jwt: str, **filters: object) -> dict[str, object]:
    """GET /v2/user/audit-logs -> {page_details, data: [AuditLogResponse]}."""
    params = {
        key: value for key, value in filters.items() if isinstance(value, str | int | float | bool)
    }
    with _client(session_jwt) as client:
        resp = client.get("/v2/user/audit-logs", params=params)
    return _json_object_or_raise(resp)
