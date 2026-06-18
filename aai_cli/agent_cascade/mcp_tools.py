"""Load tools from user-configured MCP servers for the `assembly live` agent.

The live voice agent's brain is a deepagents graph, so any Model Context Protocol
server's tools can be threaded into it through ``langchain-mcp-adapters`` — the same
adapter `docs_mcp.py` uses for the hosted AssemblyAI docs. This lets a spoken
conversation reach real tools (clock, weather, memory, a notes folder, …), bringing
`assembly live` toward Gemini-Live / ChatGPT-voice parity.

Two entry points feed the brain:

- :func:`default_servers` returns a curated, zero/low-auth set (time, fetch, memory,
  filesystem, weather) that every live session loads out of the box.
- :func:`parse_mcp_config` reads one or more standard ``mcpServers`` JSON files — the
  exact shape Claude Desktop / Claude Code use — so an existing config drops in
  unchanged and can extend or override the defaults.

Launching a server is **best-effort per server**: a missing ``npx``/``uvx`` or an
offline run skips that one server (the others still load) rather than aborting the
session — a single broken tool can't sink a live demo.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.core import jsonshape
from aai_cli.core.errors import UsageError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.sessions import Connection

# One MCP server's launch spec, as it appears under "mcpServers" in a standard config:
# stdio servers carry {command, args, env}; remote servers carry {url}.
ServerSpec = Mapping[str, object]
# A loader maps (server name, adapter connection dict) -> the server's tools. Injected in
# tests so the per-server orchestration runs without subprocesses or sockets.
Loader = Callable[[str, "Connection"], "list[BaseTool]"]


def default_servers(filesystem_root: Path) -> dict[str, ServerSpec]:
    """The curated server set every live session loads: zero/low-auth, fast, speakable.

    Every entry is a published reference server runnable with no API key:
    ``time``/``fetch`` over ``uvx`` (PyPI), ``memory``/``filesystem`` over ``npx`` (npm),
    and an NWS-backed ``weather`` server. ``filesystem`` is rooted at ``filesystem_root``
    (the working directory) so "summarize my notes file" stays scoped to one folder.
    """
    return {
        "time": {"command": "uvx", "args": ["mcp-server-time"]},
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "memory": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(filesystem_root)],
        },
        "weather": {"command": "npx", "args": ["-y", "@h1deya/mcp-server-weather"]},
    }


def parse_mcp_config(paths: Sequence[Path]) -> dict[str, ServerSpec]:
    """Merge the ``mcpServers`` maps from one or more standard MCP config JSON files.

    Each file must be ``{"mcpServers": {name: spec, …}}`` (the Claude Desktop / Claude
    Code shape). Later files win on a name clash. A malformed file, a missing
    ``mcpServers`` key, or a spec with neither ``command`` nor ``url`` is a usage error,
    surfaced before any audio device opens.
    """
    servers: dict[str, ServerSpec] = {}
    for path in paths:
        try:
            data = jsonshape.as_mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise UsageError(f"Could not read MCP config {str(path)!r}: {exc}") from exc
        entries = jsonshape.as_mapping(data.get("mcpServers")) if data is not None else None
        if entries is None:
            raise UsageError(
                f"MCP config {str(path)!r} has no 'mcpServers' object.",
                suggestion='Expected {"mcpServers": {"name": {"command": "…"}}}.',
            )
        for name, spec in entries.items():
            servers[name] = _validate_spec(name, spec)
    return servers


def _validate_spec(name: str, spec: object) -> dict[str, object]:
    """Return the spec as a mapping, or reject one naming neither a ``command`` nor ``url``."""
    mapping = jsonshape.as_mapping(spec)
    if mapping is None or ("command" not in mapping and "url" not in mapping):
        raise UsageError(
            f"MCP server {name!r} needs a 'command' or 'url'.",
            suggestion='e.g. {"command": "uvx", "args": ["mcp-server-time"]}.',
        )
    return mapping


def _to_connection(spec: ServerSpec) -> Connection:
    """Translate a standard ``mcpServers`` spec into a langchain-mcp-adapters connection.

    A ``url`` spec becomes a ``streamable_http`` transport; otherwise it's a ``stdio``
    transport launched from ``command``/``args`` (passing ``env`` through when present).
    """
    if "url" in spec:
        return {"transport": "streamable_http", "url": str(spec["url"])}
    args = [str(arg) for arg in jsonshape.object_list(spec.get("args"))]
    env_map = jsonshape.as_mapping(spec.get("env"))
    env = {str(k): str(v) for k, v in env_map.items()} if env_map is not None else None
    return {"transport": "stdio", "command": str(spec["command"]), "args": args, "env": env}


def _load_server(name: str, conn: Connection) -> list[BaseTool]:
    """Connect to one MCP server and return its tools (drives the async adapter)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def _fetch() -> list[BaseTool]:
        client = MultiServerMCPClient({name: conn})
        return await client.get_tools()

    return asyncio.run(_fetch())


def _safe_load(loader: Loader, name: str, spec: ServerSpec) -> list[BaseTool]:
    """One server's tools, or ``[]`` if it won't start — so a failure is never fatal."""
    try:
        return loader(name, _to_connection(spec))
    except Exception:
        return []


def load_mcp_tools(
    servers: Mapping[str, ServerSpec], *, loader: Loader = _load_server
) -> list[BaseTool]:
    """Load the tools from every configured MCP server, skipping any that fail to start.

    Each server is launched independently so one unreachable server (npx not installed,
    an offline host) drops only its own tools — the rest still load. ``loader`` is the
    only network/subprocess seam, injected in tests.
    """
    tools: list[BaseTool] = []
    for name, spec in servers.items():
        tools.extend(_safe_load(loader, name, spec))
    return tools
