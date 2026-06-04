from __future__ import annotations

import webbrowser

from aai_cli import output
from aai_cli.auth import ams, discovery, endpoints, loopback
from aai_cli.errors import APIError


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
    if not projects:
        raise APIError("Your account has no project to create an API key in.")
    for entry in projects:
        for token in entry.get("tokens", []):
            if token.get("name") == endpoints.CLI_TOKEN_NAME and not token.get("is_disabled"):
                return str(token["api_key"])
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
    if result.token_type != "discovery_oauth" or not result.token:  # noqa: S105
        raise APIError("Login did not return a valid OAuth token. Run 'aai login' again.")

    disc = ams.discover(result.token)
    organizations = disc.get("organizations") or []
    if not organizations:
        raise APIError(
            "Signed in, but this identity has no AssemblyAI account yet. "
            f"Create one at {endpoints.SIGNUP_URL}, then run 'aai login' again."
        )
    organization_id = organizations[0]["organization_id"]

    signed_in = ams.exchange(disc["intermediate_session_token"], organization_id)
    session_jwt = signed_in["session_jwt"]

    account = ams.get_auth(session_jwt)
    return find_or_create_cli_key(int(account["id"]), session_jwt)
