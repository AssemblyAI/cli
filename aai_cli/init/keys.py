from __future__ import annotations

import os

from aai_cli import config


def resolve_optional_api_key(*, profile: str | None) -> tuple[str | None, str | None]:
    """The CLI's key chain (env -> keyring) plus which source supplied the key.

    Returns ``(key, source)`` with source ``"environment"`` or ``"keyring"``, or
    ``(None, None)`` when absent. `assembly init` scaffolds even without a key
    (writing a placeholder), so it must not fail the way run commands do; the
    source feeds the report's ``key`` row.
    """
    key = config.resolve_api_key_optional(profile=profile)
    if key is None:
        return None, None
    source = "environment" if os.environ.get(config.ENV_API_KEY) else "keyring"
    return key, source
