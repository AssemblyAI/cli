from __future__ import annotations

from urllib.parse import urlencode

from assemblyai_cli.auth import endpoints


def build_start_url() -> str:
    """The Stytch B2B OAuth discovery *start* URL the browser opens.

    Client-side endpoint authenticated by the project's public token. After the
    user authenticates with the provider, Stytch redirects to our loopback
    `discovery_redirect_url` with `?stytch_token_type=discovery_oauth&token=...`.
    No custom state is appended: the redirect is exact-match validated and the
    server is loopback-only, single-shot.
    """
    base = (
        f"{endpoints.STYTCH_PROJECT_DOMAIN}"
        f"/v1/b2b/public/oauth/{endpoints.STYTCH_OAUTH_PROVIDER}/discovery/start"
    )
    params = {
        "public_token": endpoints.STYTCH_PUBLIC_TOKEN,
        "discovery_redirect_url": endpoints.redirect_uri(),
    }
    return f"{base}?{urlencode(params)}"
