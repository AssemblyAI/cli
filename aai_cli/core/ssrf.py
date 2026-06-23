"""SSRF guard for outbound URL fetches.

The CLI fetches URLs that originate from untrusted sources — a page the live
agent's ``read_url`` tool is steered to by web content it just read, a
``speak --url`` argument, a podcast feed URL. A bare string check (``startswith
"127."``) is not enough: a hostname can *resolve* to a private address, a public
URL can *redirect* to one, and ``127.0.0.1`` has many literal spellings
(decimal ``2130706433``, hex, IPv4-mapped IPv6). So the guard resolves the host
and inspects the resulting IPs, and callers re-run it on **every redirect hop**.

A residual gap is DNS rebinding (the name resolves to a public IP here, then to a
private one when httpx connects); closing it fully needs IP-pinned connections,
which is out of scope — this raises the bar from "trivially bypassable" to
"resolves and is checked", which is the meaningful win for a single-user CLI.

Stdlib-only (``socket``/``ipaddress``) plus :mod:`aai_cli.core.errors`, so both
``core.webpage`` and ``app.transcribe.feed`` import it without pulling httpx onto
their startup path.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

from aai_cli.core.errors import UsageError

# Cap how many redirect hops a caller follows before giving up — also the loop
# bound that stops a redirect cycle from spinning forever.
MAX_REDIRECTS = 5  # pragma: no mutate -- tuning knob; a +-1 shift is behaviorally equivalent
# The HTTP status codes that carry a Location to follow (callers re-validate each).
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})  # pragma: no mutate -- standard codes


class BlockedURLError(UsageError):
    """A URL was refused because it isn't an http(s) address that resolves to a
    public host (the SSRF guard). A ``UsageError`` so it renders as a clean exit-2
    message and callers that already catch ``UsageError`` handle it uniformly."""


def _resolve_host(host: str) -> list[str]:
    """The IP strings ``host`` resolves to (split out so tests can stub DNS).

    Uses ``getaddrinfo``, which resolves the same way httpx's connection will —
    including libc's acceptance of non-dotted-decimal IPv4 literals — so the IPs
    we inspect are the IPs that would actually be connected to. Each sockaddr's
    address is normalized through ``ip_address`` to a canonical string.
    """
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [ipaddress.ip_address(info[4][0]).compressed for info in infos]


def _is_blocked_ip(ip: str) -> bool:
    """True when ``ip`` is a non-public address an outbound fetch must never reach.

    ``is_private`` already covers loopback, link-local (incl. the
    ``169.254.169.254`` cloud-metadata address), unique-local, reserved, and the
    RFC 1918 ranges for both families; multicast is the one internal class it
    doesn't, so it is checked explicitly.
    """
    addr = ipaddress.ip_address(ip)
    # Normalize an IPv4-mapped IPv6 address (``::ffff:a.b.c.d``) to its v4 form so the
    # v4 rules apply regardless of how the Python version classifies the mapped form.
    mapped = addr.ipv4_mapped if isinstance(addr, ipaddress.IPv6Address) else None
    if mapped is not None:  # pragma: no mutate -- cross-version v4-mapped normalization
        addr = mapped  # pragma: no mutate
    return addr.is_private or addr.is_multicast


def assert_public_url(url: str) -> None:
    """Raise :class:`BlockedURLError` unless ``url`` is an http(s) URL whose host
    resolves only to public addresses.

    Call this for the initial URL *and* for every redirect target, since a public
    URL can 30x-redirect to an internal one.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedURLError(
            f"Refused to fetch a non-web URL: {url}",
            suggestion="Only http(s) URLs are fetched.",
        )
    host = parts.hostname
    if not host:
        raise BlockedURLError(f"Refused to fetch a URL with no host: {url}")
    if any(_is_blocked_ip(ip) for ip in _resolve_host(host)):
        raise BlockedURLError(
            f"Refused to fetch a private or internal address: {url}",
            suggestion="The URL resolves to a loopback, private, link-local, or internal address.",
        )
