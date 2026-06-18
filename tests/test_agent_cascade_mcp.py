"""Tests for the MCP-server toolset behind `assembly live --mcp-config/--demo-tools`.

The only network/subprocess seam is the per-server ``loader``, injected here so the
config parsing, connection translation, and best-effort per-server loading all run with
no sockets or `npx`/`uvx` subprocesses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aai_cli.agent_cascade import mcp_tools
from aai_cli.commands.agent_cascade import _exec
from aai_cli.core.errors import UsageError

# --- demo_servers ------------------------------------------------------------


def test_demo_servers_curated_set_and_filesystem_root():
    servers = mcp_tools.demo_servers(Path("/notes/dir"))
    # The five curated, no-auth servers, each with a real launch command.
    assert set(servers) == {"time", "fetch", "memory", "filesystem", "weather"}
    assert servers["time"] == {"command": "uvx", "args": ["mcp-server-time"]}
    assert servers["memory"]["args"] == ["-y", "@modelcontextprotocol/server-memory"]
    # The filesystem server is scoped to the passed-in root directory.
    assert servers["filesystem"]["args"] == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/notes/dir",
    ]


# --- parse_mcp_config --------------------------------------------------------


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_mcp_config_reads_and_merges_later_file_wins(tmp_path):
    a = _write(tmp_path / "a.json", {"mcpServers": {"time": {"command": "uvx"}, "x": {"url": "u"}}})
    b = _write(tmp_path / "b.json", {"mcpServers": {"time": {"command": "npx"}}})
    servers = mcp_tools.parse_mcp_config([a, b])
    # Both files' servers are present; the later file overrides a clashing name.
    assert set(servers) == {"time", "x"}
    assert servers["time"] == {"command": "npx"}


def test_parse_mcp_config_empty_paths_is_empty():
    assert mcp_tools.parse_mcp_config([]) == {}


def test_parse_mcp_config_malformed_json_is_usage_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(UsageError, match="Could not read MCP config"):
        mcp_tools.parse_mcp_config([bad])


def test_parse_mcp_config_missing_mcpservers_key_is_usage_error(tmp_path):
    path = _write(tmp_path / "c.json", {"servers": {}})
    with pytest.raises(UsageError, match="no 'mcpServers'"):
        mcp_tools.parse_mcp_config([path])


def test_parse_mcp_config_spec_without_command_or_url_is_usage_error(tmp_path):
    path = _write(tmp_path / "d.json", {"mcpServers": {"bad": {"args": ["x"]}}})
    with pytest.raises(UsageError, match="needs a 'command' or 'url'"):
        mcp_tools.parse_mcp_config([path])


# --- _validate_spec ----------------------------------------------------------


def test_validate_spec_accepts_command_or_url():
    # Both shapes are valid and the spec is returned narrowed to a mapping.
    assert mcp_tools._validate_spec("a", {"command": "uvx"}) == {"command": "uvx"}
    assert mcp_tools._validate_spec("b", {"url": "https://x"}) == {"url": "https://x"}


def test_validate_spec_rejects_non_mapping():
    with pytest.raises(UsageError):
        mcp_tools._validate_spec("a", ["not", "a", "mapping"])


# --- _to_connection ----------------------------------------------------------


def test_to_connection_stdio_carries_command_args_and_env():
    conn = mcp_tools._to_connection({"command": "npx", "args": ["-y", "pkg"], "env": {"K": "V"}})
    assert conn == {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "pkg"],
        "env": {"K": "V"},
    }


def test_to_connection_stdio_without_args_or_env_defaults():
    conn = mcp_tools._to_connection({"command": "uvx"})
    assert conn == {"transport": "stdio", "command": "uvx", "args": [], "env": None}


def test_to_connection_url_becomes_streamable_http():
    conn = mcp_tools._to_connection({"url": "https://host/mcp"})
    assert conn == {"transport": "streamable_http", "url": "https://host/mcp"}


# --- load_mcp_tools / _safe_load ---------------------------------------------


def test_load_mcp_tools_combines_tools_from_each_server():
    def loader(name, conn) -> list:
        del conn
        return [f"{name}-tool"]

    tools = mcp_tools.load_mcp_tools({"a": {"command": "x"}, "b": {"command": "y"}}, loader=loader)
    assert tools == ["a-tool", "b-tool"]


def test_load_mcp_tools_skips_a_server_that_fails_to_start():
    def loader(name, conn) -> list:
        del conn
        if name == "broken":
            raise RuntimeError("npx not found")
        return [f"{name}-tool"]

    tools = mcp_tools.load_mcp_tools(
        {"broken": {"command": "x"}, "ok": {"command": "y"}}, loader=loader
    )
    # The broken server contributes nothing; the working server's tool still loads.
    assert tools == ["ok-tool"]


def test_load_mcp_tools_empty_servers_is_empty():
    # No servers -> the loader is never reached and the result is empty.
    assert mcp_tools.load_mcp_tools({}) == []


def test_safe_load_returns_empty_on_failure():
    def boom(name, conn) -> list:
        raise RuntimeError("down")

    assert mcp_tools._safe_load(boom, "s", {"command": "x"}) == []


# --- _resolve_mcp_servers (the --demo-tools / --mcp-config merge) -------------


def test_resolve_mcp_servers_demo_only():
    servers = _exec._resolve_mcp_servers(demo_tools=True, mcp_config=())
    # The curated demo set, with no config files.
    assert {"time", "weather", "memory", "fetch", "filesystem"} <= set(servers)


def test_resolve_mcp_servers_config_only(tmp_path):
    path = tmp_path / "servers.json"
    path.write_text(
        '{"mcpServers": {"custom": {"command": "uvx", "args": ["x"]}}}', encoding="utf-8"
    )
    servers = _exec._resolve_mcp_servers(demo_tools=False, mcp_config=(path,))
    assert servers == {"custom": {"command": "uvx", "args": ["x"]}}


def test_resolve_mcp_servers_config_overrides_demo_by_name(tmp_path):
    path = tmp_path / "servers.json"
    path.write_text('{"mcpServers": {"time": {"command": "my-time"}}}', encoding="utf-8")
    servers = _exec._resolve_mcp_servers(demo_tools=True, mcp_config=(path,))
    # An explicit config entry overrides the demo entry of the same name.
    assert servers["time"] == {"command": "my-time"}


def test_resolve_mcp_servers_none_is_empty():
    assert _exec._resolve_mcp_servers(demo_tools=False, mcp_config=()) == {}


def test_load_server_drives_the_adapter_with_a_one_server_client(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, connections):
            captured["connections"] = connections

        async def get_tools(self):
            return ["tool-a"]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient, raising=True
    )
    conn = mcp_tools._to_connection({"command": "uvx", "args": ["mcp-server-time"]})
    tools = mcp_tools._load_server("time", conn)
    # The named server's connection is handed to the adapter and its tools returned.
    assert tools == ["tool-a"]
    assert captured["connections"] == {"time": conn}
