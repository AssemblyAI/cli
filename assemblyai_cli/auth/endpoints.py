from __future__ import annotations

import os

# Stytch B2B project (sandbox defaults; override via env for prod).
STYTCH_PROJECT_DOMAIN = os.environ.get(
    "AAI_AUTH_STYTCH_DOMAIN",
    "https://psychedelic-journey-5884.customers.stytch.dev",
)
STYTCH_PUBLIC_TOKEN = os.environ.get(
    "AAI_AUTH_PUBLIC_TOKEN",
    "public-token-test-79ad7d8d-09df-495e-8eb8-72d18efdafa4",
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


def redirect_uri() -> str:
    """The exact loopback redirect URL registered in Stytch."""
    return f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}{LOOPBACK_PATH}"
