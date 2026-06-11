from __future__ import annotations

import os

from aai_cli import environments
from aai_cli.errors import CLIError

# Constant across environments.
STYTCH_OAUTH_PROVIDER = "google"
CLI_TOKEN_NAME = "AssemblyAI CLI"  # noqa: S105 - display name, not a credential

# Fixed loopback (Stytch does exact-match redirect validation; 8585 is registered).
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PATH = "/callback"
_DEFAULT_LOOPBACK_PORT = 8585
_MAX_PORT = 65535  # highest valid TCP port


def _invalid_auth_port(raw: str) -> CLIError:
    return CLIError(
        f"AAI_AUTH_PORT must be a port number in 1-65535, got {raw!r}.",
        error_type="invalid_env",
        exit_code=2,
        suggestion="Unset AAI_AUTH_PORT, or set it to a free port in 1-65535.",
    )


def loopback_port() -> int:
    """The loopback callback port, overridable via ``AAI_AUTH_PORT`` (dev/test only).

    Resolved lazily — never at import — and validated, so a malformed override
    surfaces as a clean CLIError on the login path instead of the raw ``ValueError``
    a module-level ``int(...)`` would raise. This module sits on the CLI's import hot
    path, so that ValueError would otherwise crash *every* ``assembly`` command (even
    ``--help``), not just ``assembly login``.
    """
    raw = os.environ.get("AAI_AUTH_PORT")
    if raw is None:
        return _DEFAULT_LOOPBACK_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise _invalid_auth_port(raw) from exc
    if not 1 <= port <= _MAX_PORT:
        raise _invalid_auth_port(raw)
    return port


# Environment-specific values resolve from the active environment (see
# aai_cli.environments). The B2B OAuth flow starts on Stytch's API domain (not a
# vanity CNAME) so the CSRF state cookie is set on the same domain as the OAuth
# callback — otherwise Stytch returns `oauth_invalid_state`.
def stytch_domain() -> str:
    return environments.active().stytch_domain


def stytch_public_token() -> str:
    return environments.active().stytch_public_token


def ams_base() -> str:
    return environments.active().ams_base


def signup_url() -> str:
    return environments.active().signup_url


def redirect_uri() -> str:
    """The exact loopback redirect URL registered in Stytch."""
    return f"http://{LOOPBACK_HOST}:{loopback_port()}{LOOPBACK_PATH}"
