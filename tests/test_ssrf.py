"""The SSRF guard (`aai_cli/core/ssrf.py`): IP classification and URL validation.

DNS is stubbed (`_resolve_host`) so these stay hermetic under the socket-blocked
suite while still exercising the real `ipaddress`-based classification.
"""

from __future__ import annotations

import pytest

from aai_cli.core import ssrf
from aai_cli.core.errors import UsageError


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.0.0.1",  # RFC 1918 private
        "192.168.1.1",  # RFC 1918 private
        "172.16.0.1",  # RFC 1918 private
        "169.254.169.254",  # link-local — the cloud-metadata address
        "0.0.0.1",  # the unspecified 0.0.0.0/8 range
        "224.0.0.1",  # multicast
        "::1",  # IPv6 loopback
        "fd00::1",  # IPv6 unique-local
        "fe80::1",  # IPv6 link-local
        "::ffff:169.254.169.254",  # IPv4-mapped IPv6 of the metadata address
        "::ffff:127.0.0.1",  # IPv4-mapped IPv6 loopback
    ],
)
def test_internal_ips_are_blocked(ip):
    assert ssrf._is_blocked_ip(ip) is True


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
def test_public_ips_are_allowed(ip):
    assert ssrf._is_blocked_ip(ip) is False


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_resolve_host_returns_canonical_ips():
    # A numeric literal resolves locally (no DNS), exercising the real getaddrinfo
    # path the rest of the suite stubs out.
    assert ssrf._resolve_host("127.0.0.1") == ["127.0.0.1"]


def test_assert_public_url_allows_public_host(monkeypatch):
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["93.184.216.34"])
    # Returns without raising for a host that resolves to a public address.
    ssrf.assert_public_url("https://example.com/page")


def test_assert_public_url_blocks_host_resolving_internal(monkeypatch):
    # A perfectly ordinary-looking hostname that resolves to an internal IP is the
    # DNS-based bypass the string regex missed.
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["169.254.169.254"])
    with pytest.raises(ssrf.BlockedURLError):
        ssrf.assert_public_url("http://metadata.example.com/")


def test_assert_public_url_rejects_non_http_scheme(monkeypatch):
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["93.184.216.34"])
    with pytest.raises(ssrf.BlockedURLError):
        ssrf.assert_public_url("file:///etc/passwd")


def test_assert_public_url_rejects_missing_host(monkeypatch):
    # Even if resolution would pass, a URL with no host is refused before resolving.
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["93.184.216.34"])
    with pytest.raises(ssrf.BlockedURLError):
        ssrf.assert_public_url("http:///just-a-path")


def test_blocked_url_error_is_a_usage_error():
    # So it renders as a clean exit-2 message and existing `except UsageError`
    # handlers (e.g. the read_url tool) catch it uniformly.
    assert issubclass(ssrf.BlockedURLError, UsageError)
