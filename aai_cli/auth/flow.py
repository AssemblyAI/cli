from __future__ import annotations

import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass

from rich.markup import escape

from aai_cli import jsonshape, output
from aai_cli.auth import ams, discovery, endpoints, loopback
from aai_cli.errors import APIError


@dataclass
class LoginResult:
    """Everything a successful browser login yields: a usable API key plus the
    Stytch session the AMS self-service commands authenticate with."""

    api_key: str
    session_jwt: str
    session_token: str
    account_id: int


def _as_mapping(value: object) -> dict[str, object] | None:
    return jsonshape.as_mapping(value)


def _mapping_list(value: object) -> list[dict[str, object]]:
    return jsonshape.mapping_list(value)


def _require_int(mapping: Mapping[str, object], key: str, what: str) -> int:
    value = _require(mapping, key, what)
    if isinstance(value, bool):
        raise APIError(
            f"Login failed: the server response was missing {what}. Run 'aai login' again."
        )
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise APIError(
                f"Login failed: the server response was missing {what}. Run 'aai login' again."
            ) from exc
    raise APIError(f"Login failed: the server response was missing {what}. Run 'aai login' again.")


def _require(mapping: Mapping[str, object], key: str, what: str) -> object:
    """Pull a required field out of an AMS response, or raise a clean APIError.

    AMS only returns HTTP errors for outright failures; a 200 with an unexpected
    shape would otherwise KeyError into an ugly traceback, so map that to the same
    "run login again" message the rest of the flow uses. The return stays `object`
    because AMS JSON leaves are narrowed by callers with int()/str().
    """
    value = mapping.get(key)
    if value is None:
        raise APIError(
            f"Login failed: the server response was missing {what}. Run 'aai login' again."
        )
    return value


def _require_mapping(mapping: Mapping[str, object], key: str, what: str) -> dict[str, object]:
    value = _require(mapping, key, what)
    mapped = _as_mapping(value)
    if mapped is None:
        raise APIError(
            f"Login failed: the server response was missing {what}. Run 'aai login' again."
        )
    return mapped


def _open_browser(url: str) -> None:
    """Open the system browser, falling back to printing the URL."""
    output.error_console.print(
        f"Opening your browser to sign in:\n  [aai.url]{escape(url)}[/aai.url]"
    )
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - opening a browser is best-effort
        output.error_console.print(
            "[aai.muted]Could not open a browser; open the URL above manually.[/aai.muted]"
        )


def _capture() -> loopback.CallbackResult:
    return loopback.capture_callback()


def _is_reusable_cli_token(token: dict[str, object]) -> bool:
    """A live 'AssemblyAI CLI' token whose key the list actually exposes."""
    # List endpoints may key the display name as either "name" or "token_name"
    # (the latter matches the create payload); accept either so we don't mint a
    # duplicate every login. A token whose api_key the list omits can't be reused.
    name = token.get("name") or token.get("token_name")
    return (
        name == endpoints.CLI_TOKEN_NAME
        and not token.get("is_disabled")
        and bool(token.get("api_key"))
    )


def find_or_create_cli_key(account_id: int, session_jwt: str) -> str:
    """Return the existing 'AssemblyAI CLI' key, or create one in the first project."""
    projects = ams.list_projects(account_id, session_jwt)
    if not projects:
        raise APIError(
            "Your account has no project to create an API key in.",
            suggestion="Create a project in the AssemblyAI dashboard, then run 'aai login' again.",
        )
    for entry in projects:
        for token in _mapping_list(entry.get("tokens")):
            if _is_reusable_cli_token(token):
                return str(token["api_key"])
    project_id = _require_int(
        _require_mapping(projects[0], "project", "a project"), "id", "a project id"
    )
    created = ams.create_token(account_id, project_id, endpoints.CLI_TOKEN_NAME, session_jwt)
    return str(_require(created, "api_key", "an API key"))


def run_login_flow() -> LoginResult:
    """Drive the full browser + AMS login and return a LoginResult."""
    _open_browser(discovery.build_start_url())
    result = _capture()

    if result.error == "timeout":
        raise APIError(
            "Login timed out waiting for the browser.",
            suggestion="Run 'aai login' again.",
        )
    if result.token_type != "discovery_oauth" or not result.token:  # noqa: S105
        raise APIError(
            "Login did not return a valid OAuth token.",
            suggestion="Run 'aai login' again.",
        )

    disc = ams.discover(result.token)
    organizations = _mapping_list(disc.get("organizations"))
    if not organizations:
        raise APIError(
            "Signed in, but this identity has no AssemblyAI account yet. "
            f"Create one at {endpoints.signup_url()}, then run 'aai login' again."
        )
    if len(organizations) > 1:
        chosen = str(
            organizations[0].get("organization_name")
            or organizations[0].get("organization_id", "the first")
        )
        output.error_console.print(
            f"[aai.muted]Found {len(organizations)} organizations; signing in to "
            f"'{chosen}'.[/aai.muted]"
        )
    organization_id = str(_require(organizations[0], "organization_id", "an organization id"))

    intermediate_session_token = str(
        _require(disc, "intermediate_session_token", "a session token")
    )
    signed_in = ams.exchange(intermediate_session_token, organization_id)
    session_jwt = str(_require(signed_in, "session_jwt", "a session token"))
    session_token = str(_require(signed_in, "session_token", "a session token"))
    # `exchange` already returns the signed-in account, so read the id from it
    # rather than making a second GET /v1/auth round-trip.
    account_id = _require_int(
        _require_mapping(signed_in, "account", "an account"), "id", "an account id"
    )
    api_key = find_or_create_cli_key(account_id, session_jwt)
    return LoginResult(
        api_key=api_key,
        session_jwt=session_jwt,
        session_token=session_token,
        account_id=account_id,
    )
