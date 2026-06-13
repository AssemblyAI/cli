from __future__ import annotations

from aai_cli.core import config, env


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
    # Mirror resolve_api_key's whitespace handling: a blank env var is "unset",
    # so a key that actually came from the keyring must not report "environment".
    env_value = (env.get(config.ENV_API_KEY) or "").strip()
    source = "environment" if env_value else "keyring"
    return key, source
