from __future__ import annotations

from aai_cli import config
from aai_cli.errors import NotAuthenticated


def resolve_optional_api_key(*, profile: str | None) -> str | None:
    """The CLI's key chain (env -> keyring), but None instead of raising when absent.

    `aai init` scaffolds even without a key (writing a placeholder), so it must not
    fail the way run commands do.
    """
    try:
        return config.resolve_api_key(profile=profile)
    except NotAuthenticated:
        return None
