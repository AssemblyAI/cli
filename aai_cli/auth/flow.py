from __future__ import annotations

import sys
import webbrowser
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter, ValidationError
from rich.markup import escape

from aai_cli.auth import ams, discovery, endpoints, loopback
from aai_cli.core import jsonshape
from aai_cli.core.errors import STDIN_KEY_RECIPE, APIError, NotAuthenticated
from aai_cli.ui import output


@dataclass
class LoginResult:
    """Everything a successful browser login yields: a usable API key plus the
    Stytch session the AMS self-service commands authenticate with."""

    api_key: str
    session_jwt: str
    session_token: str
    account_id: int
    # The signed-in user's email, from AMS discovery. Persisted so the CLI can gate
    # internal-only environments (the sandbox) on the org domain; None if AMS omits it.
    email: str | None = None


# Typed views of the AMS login responses. AMS only returns HTTP errors for outright
# failures; a 200 with an unexpected shape would otherwise KeyError into an ugly
# traceback, so each required field's absence becomes the same clean "run login
# again" APIError via `_parse`. Only the fields below are read; the rest are ignored.
class _Organization(BaseModel):
    organization_id: str
    organization_name: str | None = None


class _Discovery(BaseModel):
    intermediate_session_token: str
    organizations: list[_Organization] = []
    # Top-level email from the discover response; used only to gate sandbox access.
    email: str | None = None


class _Account(BaseModel):
    id: int


class _SignedIn(BaseModel):
    session_jwt: str
    session_token: str
    account: _Account


class _Project(BaseModel):
    id: int


class _Token(BaseModel):
    # List endpoints may key the display name as either "name" or "token_name"
    # (the latter matches the create payload), so accept either.
    name: str | None = None
    token_name: str | None = None
    is_disabled: bool = False
    api_key: str | None = None


class _ProjectEntry(BaseModel):
    project: _Project | None = None
    tokens: list[_Token] = []


class _CreatedToken(BaseModel):
    api_key: str


_DISCOVERY: TypeAdapter[_Discovery] = TypeAdapter(_Discovery)
_SIGNED_IN: TypeAdapter[_SignedIn] = TypeAdapter(_SignedIn)
_CREATED_TOKEN: TypeAdapter[_CreatedToken] = TypeAdapter(_CreatedToken)
_PROJECT_LIST: TypeAdapter[list[_ProjectEntry]] = TypeAdapter(list[_ProjectEntry])


def _parse[T](adapter: TypeAdapter[T], data: object) -> T:
    """Validate an AMS response into a typed view, or raise a clean login error."""
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise APIError(
            "Login failed: the server returned an unexpected response. Run 'assembly login' again."
        ) from exc


def _note(*, json_mode: bool, human: str, hint: str, url: str | None = None) -> None:
    """One stderr progress note for the login flow.

    Humans get Rich prose; ``--json`` mode gets one ``{"hint": …}`` object per note,
    keeping stderr machine-readable (the same contract as ``output.emit_warning``).
    """
    if json_mode:
        payload: dict[str, object] = {"hint": hint}
        if url is not None:
            payload["url"] = url
        sys.stderr.write(jsonshape.dumps(payload) + "\n")
    else:
        output.error_console.print(human)


def _open_browser(url: str, *, json_mode: bool) -> None:
    """Open the system browser, falling back to printing the URL."""
    _note(
        json_mode=json_mode,
        human=f"Opening your browser to sign in:\n  [aai.url]{escape(url)}[/aai.url]",
        hint="Opening your browser to sign in.",
        url=url,
    )
    try:
        opened = webbrowser.open(url)
    except Exception:  # noqa: BLE001 - opening a browser is best-effort
        opened = False
    # webbrowser.open returns False — without raising — on headless boxes with no
    # usable browser, so the fallback must fire on the boolean too; otherwise the
    # user sits out the 120s timeout with no hint that nothing opened.
    if not opened:
        # The OAuth callback lands on this machine's loopback port, so a browser on
        # another machine (the common SSH case) can only complete the flow through a
        # port forward — say so now, with the exact command, instead of letting the
        # user open the URL remotely and watch the callback go nowhere.
        port = endpoints.loopback_port()
        forward = f"ssh -L {port}:{endpoints.LOOPBACK_HOST}:{port} <this-host>"
        _note(
            json_mode=json_mode,
            human=(
                "[aai.muted]Could not open a browser; open the URL above manually.\n"
                f"On a remote/SSH machine, forward the callback port first ({forward}) "
                "and open the URL in your local browser — or skip the browser entirely: "
                f"{STDIN_KEY_RECIPE}.[/aai.muted]"
            ),
            hint=(
                "Could not open a browser; open the URL manually. On a remote/SSH "
                f"machine, forward the callback port first ({forward}) and open the "
                "URL in your local browser — or skip the browser entirely: "
                f"{STDIN_KEY_RECIPE}."
            ),
            url=url,
        )


def _start_capture() -> loopback.CallbackCapture:
    return loopback.start_capture()


def _reusable_cli_key(token: _Token) -> str | None:
    """The api_key of a live 'AssemblyAI CLI' token the list actually exposes, else None.

    A disabled token, or one whose api_key the list omits, can't be reused — we fall
    through to minting a fresh one instead of duplicating or crashing.
    """
    name = token.name or token.token_name
    if name == endpoints.CLI_TOKEN_NAME and not token.is_disabled and token.api_key:
        return token.api_key
    return None


def _no_project_error() -> APIError:
    """The one failure mode behind both guards in ``find_or_create_cli_key``:
    nowhere to mint a key (no project entries at all, or an entry without a project)."""
    return APIError(
        "Your account has no project to create an API key in.",
        suggestion="Create a project in the AssemblyAI dashboard, then run 'assembly login' again.",
    )


def find_or_create_cli_key(account_id: int, session_jwt: str) -> str:
    """Return the existing 'AssemblyAI CLI' key, or create one in the first usable project."""
    projects = _parse(_PROJECT_LIST, ams.list_projects(account_id, session_jwt))
    if not projects:
        raise _no_project_error()
    for entry in projects:
        for token in entry.tokens:
            if key := _reusable_cli_key(token):
                return key
    # Mint into the first entry that actually carries a project — an account whose
    # first membership has no project can still have a usable one later in the list.
    project = next((entry.project for entry in projects if entry.project is not None), None)
    if project is None:
        raise _no_project_error()
    created = ams.create_token(account_id, project.id, endpoints.CLI_TOKEN_NAME, session_jwt)
    return _parse(_CREATED_TOKEN, created).api_key


def run_login_flow(*, json_mode: bool = False) -> LoginResult:
    """Drive the full browser + AMS login and return a LoginResult."""
    # Bind the loopback callback server *before* opening the browser: if the port is
    # taken, fail cleanly now instead of stranding the user mid-OAuth in a flow that
    # can never call back.
    capture = _start_capture()
    _open_browser(discovery.build_start_url(), json_mode=json_mode)
    _note(
        json_mode=json_mode,
        human=(
            "[aai.muted]Waiting up to 2 minutes for you to finish signing in…[/aai.muted]\n"
            f"[aai.muted]No browser here? Run '{STDIN_KEY_RECIPE}' instead.[/aai.muted]"
        ),
        hint=(
            "Waiting up to 2 minutes for you to finish signing in. "
            f"No browser here? Run '{STDIN_KEY_RECIPE}' instead."
        ),
    )
    result = capture.wait()

    if result.error == "timeout":
        raise NotAuthenticated(
            "Login timed out waiting for the browser.",
            suggestion=f"Run 'assembly login' again, or use '{STDIN_KEY_RECIPE}'.",
        )
    if result.token_type != "discovery_oauth" or not result.token:  # noqa: S105
        raise APIError(
            "Login did not return a valid OAuth token.",
            suggestion="Run 'assembly login' again.",
        )

    disc = _parse(_DISCOVERY, ams.discover(result.token))
    if not disc.organizations:
        raise APIError(
            "Signed in, but this identity has no AssemblyAI account yet. "
            f"Create one at {endpoints.signup_url()}, then run 'assembly login' again."
        )
    org = disc.organizations[0]
    if len(disc.organizations) > 1:
        chosen = (
            f"Found {len(disc.organizations)} organizations; signing in to "
            f"'{org.organization_name or org.organization_id}'."
        )
        _note(json_mode=json_mode, human=f"[aai.muted]{chosen}[/aai.muted]", hint=chosen)

    # `exchange` already returns the signed-in account, so read the id from it
    # rather than making a second GET /v1/auth round-trip.
    signed_in = _parse(
        _SIGNED_IN, ams.exchange(disc.intermediate_session_token, org.organization_id)
    )
    api_key = find_or_create_cli_key(signed_in.account.id, signed_in.session_jwt)
    return LoginResult(
        api_key=api_key,
        session_jwt=signed_in.session_jwt,
        session_token=signed_in.session_token,
        account_id=signed_in.account.id,
        email=disc.email,
    )
