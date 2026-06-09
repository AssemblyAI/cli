from __future__ import annotations

from aai_cli import config


def resolve_optional_api_key(*, profile: str | None) -> str | None:
    """The CLI's key chain (env -> keyring), but None instead of raising when absent.

    `aai init` scaffolds even without a key (writing a placeholder), so it must not
    fail the way run commands do.
    """
    return config.resolve_api_key_optional(profile=profile)
