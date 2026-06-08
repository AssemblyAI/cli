from __future__ import annotations

from urllib.parse import urlencode

from aai_cli.auth import endpoints


def build_start_url() -> str:
    """The Stytch B2B OAuth discovery *start* URL the browser opens.

    Client-side endpoint authenticated by the project's public token. After the
    user authenticates with the provider, Stytch redirects to our loopback
    `discovery_redirect_url` with `?stytch_token_type=discovery_oauth&token=...`.
    The redirect URL is the bare loopback path Stytch validates by
    scheme/host/port/path, with no extra query parameters (so the redirect needs
    no wildcard registered in the Stytch dashboard).
    """
    base = (
        f"{endpoints.stytch_domain()}"
        f"/v1/b2b/public/oauth/{endpoints.STYTCH_OAUTH_PROVIDER}/discovery/start"
    )
    params = {
        "public_token": endpoints.stytch_public_token(),
        "discovery_redirect_url": endpoints.redirect_uri(),
    }
    return f"{base}?{urlencode(params)}"
