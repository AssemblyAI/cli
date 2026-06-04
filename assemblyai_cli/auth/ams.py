from __future__ import annotations

from typing import Any, cast

import httpx

from assemblyai_cli.auth import endpoints
from assemblyai_cli.errors import APIError, NotAuthenticated

_TIMEOUT = 30.0


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:  # noqa: BLE001,S110 - non-JSON error body; pass is intentional
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
    return cast(dict[str, Any], _json_or_raise(resp))


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
    return cast(dict[str, Any], _json_or_raise(resp))


def get_auth(session_jwt: str) -> dict[str, Any]:
    """GET /v1/auth (session cookie) -> account incl. `id`."""
    with httpx.Client(
        base_url=endpoints.AMS_BASE_URL,
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get("/v1/auth")
    return cast(dict[str, Any], _json_or_raise(resp))


def list_projects(account_id: int, session_jwt: str) -> list[dict[str, Any]]:
    """GET /v1/users/accounts/{id}/projects -> [{project, tokens[]}]."""
    with httpx.Client(
        base_url=endpoints.AMS_BASE_URL,
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.get(f"/v1/users/accounts/{account_id}/projects")
    return cast(list[dict[str, Any]], _json_or_raise(resp))


def create_token(
    account_id: int, project_id: int, token_name: str, session_jwt: str
) -> dict[str, Any]:
    """POST /v1/users/accounts/{id}/tokens -> TokenSchema incl. `api_key`."""
    with httpx.Client(
        base_url=endpoints.AMS_BASE_URL,
        timeout=_TIMEOUT,
        cookies={"stytch_session_jwt": session_jwt},
    ) as client:
        resp = client.post(
            f"/v1/users/accounts/{account_id}/tokens",
            json={"project_id": project_id, "token_name": token_name},
        )
    return cast(dict[str, Any], _json_or_raise(resp))
