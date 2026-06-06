from __future__ import annotations

import os
from dataclasses import dataclass

from aai_cli.errors import CLIError


@dataclass(frozen=True)
class Environment:
    """The full set of backend hosts for one deployment (prod, sandbox, …).

    A credential is only valid against its own environment's hosts, so the
    environment is bound to the profile that minted it (see commands/login).
    """

    name: str
    api_base: str  # SDK base_url for /v2/upload + /v2/transcript
    streaming_host: str  # StreamingClientOptions.api_host (SDK builds wss://host/v3/ws)
    agents_host: str  # Voice Agent host; the agent client builds wss://host/v1/ws
    llm_gateway_base: str  # OpenAI base_url for the LLM Gateway (…/v1)
    ams_base: str  # Accounts Management Service
    stytch_domain: str  # Stytch API domain for B2B OAuth discovery
    stytch_public_token: str  # client-side public token (safe to ship)
    signup_url: str  # where a first-time user creates an account


ENVIRONMENTS: dict[str, Environment] = {
    "production": Environment(
        name="production",
        api_base="https://api.assemblyai.com",
        streaming_host="streaming.assemblyai.com",
        agents_host="agents.assemblyai.com",
        llm_gateway_base="https://llm-gateway.assemblyai.com/v1",
        # NOTE: production Stytch is not provisioned yet (see the REPLACE_ME
        # token), which is why DEFAULT_ENV stays "sandbox000". Tracked under
        # spec O4.
        ams_base="https://ams.internal.assemblyai-labs.com",
        stytch_domain="https://api.stytch.com",
        stytch_public_token="public-token-live-REPLACE_ME",  # noqa: S106 - public token, safe to ship
        signup_url="https://www.assemblyai.com/dashboard",
    ),
    "sandbox000": Environment(
        name="sandbox000",
        api_base="https://api.sandbox000.assemblyai-labs.com",
        streaming_host="streaming.sandbox000.assemblyai-labs.com",
        agents_host="agents.sandbox000.assemblyai-labs.com",
        llm_gateway_base="https://llm-gateway.sandbox000.assemblyai-labs.com/v1",
        ams_base="https://ams.sandbox000.assemblyai-labs.com",
        stytch_domain="https://test.stytch.com",
        stytch_public_token="public-token-test-a161155e-7e9b-4dd1-9d43-493c899b4117",  # noqa: S106 - public token, safe to ship
        signup_url="https://dashboard-assemblyai.vercel.app/dashboard/login",
    ),
}

# Shipped default when nothing selects an environment. Flip to "production" at
# release once the production Stytch value above is real.
DEFAULT_ENV = "sandbox000"

# The environment in effect for this process, set once at CLI startup (like
# aai.settings). Resolved from --env / AAI_ENV / the profile's stored env.
_active: Environment | None = None


def get(name: str) -> Environment:
    """The named environment, or a clean CLIError if it's unknown."""
    env = ENVIRONMENTS.get(name)
    if env is None:
        raise CLIError(
            f"Unknown environment {name!r}. Known: {', '.join(ENVIRONMENTS)}.",
            error_type="invalid_environment",
            exit_code=2,
        )
    return env


def resolve(flag: str | None, profile_env: str | None) -> Environment:
    """Pick the environment by precedence: --env flag > AAI_ENV > profile > default."""
    name = flag or os.environ.get("AAI_ENV") or profile_env or DEFAULT_ENV
    return get(name)


def set_active(env: Environment) -> None:
    global _active
    _active = env


def active() -> Environment:
    """The environment in effect; falls back to the default if startup didn't set one."""
    return _active if _active is not None else get(DEFAULT_ENV)
