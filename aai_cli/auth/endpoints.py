from __future__ import annotations

import os

# Stytch B2B project (sandbox/test defaults; override via env for prod).
# Use Stytch's API domain, NOT the vanity CNAME (…customers.stytch.dev): the OAuth
# callback runs on test.stytch.com, so the CSRF state cookie set by the discovery
# /start must be on the same domain. Starting on the CNAME sets the cookie there,
# the callback can't read it, and Stytch returns `oauth_invalid_state`.
# (Prod equivalent: https://api.stytch.com)
STYTCH_PROJECT_DOMAIN = os.environ.get(
    "AAI_AUTH_STYTCH_DOMAIN",
    "https://test.stytch.com",
)
STYTCH_PUBLIC_TOKEN = os.environ.get(
    "AAI_AUTH_PUBLIC_TOKEN",
    "public-token-test-a161155e-7e9b-4dd1-9d43-493c899b4117",
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

CLI_TOKEN_NAME = "AssemblyAI CLI"  # noqa: S105 - display name, not a credential

# Where to send a first-time user who has no AssemblyAI account yet (sandbox
# default; override via env for prod, e.g. https://www.assemblyai.com/dashboard).
SIGNUP_URL = os.environ.get(
    "AAI_AUTH_SIGNUP_URL",
    "https://dashboard-assemblyai.vercel.app/dashboard/login",
)


def redirect_uri() -> str:
    """The exact loopback redirect URL registered in Stytch."""
    return f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}{LOOPBACK_PATH}"
