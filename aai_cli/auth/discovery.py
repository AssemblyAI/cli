from __future__ import annotations

from urllib.parse import urlencode

from aai_cli.auth import endpoints


def build_start_url(state: str) -> str:
    """The Stytch B2B OAuth discovery *start* URL the browser opens.

    Client-side endpoint authenticated by the project's public token. After the
    user authenticates with the provider, Stytch redirects to our loopback
    `discovery_redirect_url` with `?stytch_token_type=discovery_oauth&token=...`.

    `state` is a single-use random nonce carried as a query parameter *on the
    redirect URL*. Stytch validates the redirect URL by scheme/host/port/path and
    preserves query parameters through the flow, so the nonce comes back on the
    callback unchanged; `loopback.capture_callback` rejects any callback whose
    `state` doesn't match. This stops a malicious page from completing someone
    else's login at the loopback server (login CSRF / account confusion) by
    replaying an attacker-minted discovery token while `aai login` is waiting.
    """
    base = (
        f"{endpoints.stytch_domain()}"
        f"/v1/b2b/public/oauth/{endpoints.STYTCH_OAUTH_PROVIDER}/discovery/start"
    )
    redirect_with_state = f"{endpoints.redirect_uri()}?{urlencode({'state': state})}"
    params = {
        "public_token": endpoints.stytch_public_token(),
        "discovery_redirect_url": redirect_with_state,
    }
    return f"{base}?{urlencode(params)}"
