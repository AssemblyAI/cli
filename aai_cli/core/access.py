"""Who may select the internal-only environments (the sandbox).

The sandbox runs on internal infrastructure, so it's gated on the login email
captured at browser login (persisted per profile by ``config``), not the API key —
an API-key-only profile (CI, ``ASSEMBLYAI_API_KEY``) therefore reads as external.
The root callback rejects an internal environment for an external account, and the
root ``--help`` hides the sandbox flags/commands from it.
"""

from __future__ import annotations

from aai_cli.core import config
from aai_cli.core.errors import CLIError

# Login emails in this domain unlock the internal-only environments.
INTERNAL_EMAIL_DOMAIN = "assemblyai.com"


def is_internal_email(email: str | None) -> bool:
    """Whether ``email`` belongs to the AssemblyAI org (gates sandbox access).

    The ``@`` anchors the domain boundary so a look-alike like
    ``user@evil-assemblyai.com`` is rejected; matching is case-insensitive.
    """
    return email is not None and email.strip().lower().endswith("@" + INTERNAL_EMAIL_DOMAIN)


def profile_is_internal(profile: str | None = None) -> bool:
    """Whether a profile's stored login email is an AssemblyAI address.

    Reads the active profile when ``profile`` is None. Fails closed: an unreadable
    or corrupt config reads as external rather than raising, so the gate never
    accidentally grants access (or crashes ``--help``) on a broken config.toml.
    """
    try:
        name = profile or config.get_active_profile()
        return is_internal_email(config.get_profile_email(name))
    except CLIError:
        return False
