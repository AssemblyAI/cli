from __future__ import annotations

import os

from aai_cli import environments

# Constant across environments.
STYTCH_OAUTH_PROVIDER = "google"
CLI_TOKEN_NAME = "AssemblyAI CLI"  # noqa: S105 - display name, not a credential

# Fixed loopback (Stytch does exact-match redirect validation; 8585 is registered).
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = int(os.environ.get("AAI_AUTH_PORT", "8585"))
LOOPBACK_PATH = "/callback"


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
    return f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}{LOOPBACK_PATH}"
