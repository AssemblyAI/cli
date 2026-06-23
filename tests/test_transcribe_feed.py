"""Podcast RSS/Atom feed expansion for `assembly transcribe`.

A feed URL becomes a batch over its episode enclosures. These tests cover the
parser and fetcher in `app/transcribe/feed.py` directly, the `expand_sources`
seam that routes a feed URL into batch mode, and the end-to-end CLI run with the
network fetch faked (the suite is socket-blocked).
"""

import json

import httpx2 as httpx
import pytest
from typer.testing import CliRunner

from aai_cli.app.transcribe import feed
from aai_cli.app.transcribe import sources as transcribe_sources
from aai_cli.core import config, ssrf
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    # The SSRF guard resolves the feed host; stub DNS to a public IP so the
    # socket-blocked suite stays hermetic. Guard-specific tests override this.
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["93.184.216.34"])


_TRANSCRIBE = "aai_cli.app.transcribe.run.client.transcribe"

# A minimal but realistic RSS 2.0 podcast feed: two episodes, newest first, with
# an &amp;-escaped query string on the second enclosure URL.
_RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Example Show</title>
    <item>
      <title>Episode 2</title>
      <enclosure url="https://cdn.example.com/ep2.mp3" length="1" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 1</title>
      <enclosure type="audio/mpeg" url="https://cdn.example.com/ep1.mp3?token=a&amp;b=2"/>
    </item>
  </channel>
</rss>
"""


# --- parsing ------------------------------------------------------------------


def test_episode_urls_parses_rss_in_feed_order_and_unescapes():
    assert feed._episode_urls(_RSS) == [
        "https://cdn.example.com/ep2.mp3",
        "https://cdn.example.com/ep1.mp3?token=a&b=2",  # &amp; decoded, item order kept
    ]


def test_episode_urls_parses_atom_enclosure_links_ignoring_other_links():
    atom = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <link rel="alternate" href="https://example.com/page"/>
        <link href="https://cdn.example.com/a.mp3" rel="enclosure" type="audio/mpeg"/>
      </entry>
    </feed>"""
    # rel/href in either order; the non-enclosure <link> is skipped.
    assert feed._episode_urls(atom) == ["https://cdn.example.com/a.mp3"]


def test_episode_urls_dedupes_preserving_order():
    rss = """<rss><channel>
      <item><enclosure url="https://x/a.mp3"/></item>
      <item><enclosure url="https://x/b.mp3"/></item>
      <item><enclosure url="https://x/a.mp3"/></item>
    </channel></rss>"""
    assert feed._episode_urls(rss) == ["https://x/a.mp3", "https://x/b.mp3"]


def test_episode_urls_returns_none_when_not_a_feed():
    # An ordinary HTML page that merely contains the word enclosure is not a feed.
    html_page = '<html><body><enclosure url="https://x/a.mp3"/></body></html>'
    assert feed._episode_urls(html_page) is None


def test_episode_urls_returns_none_for_feed_without_enclosures():
    assert feed._episode_urls("<rss><channel><title>No media</title></channel></rss>") is None


def test_episode_urls_ignores_empty_url_attribute():
    assert feed._episode_urls('<rss><channel><enclosure url=""/></channel></rss>') is None


# --- media-extension gate -----------------------------------------------------


@pytest.mark.parametrize(
    ("url", "feed_shaped"),
    [
        ("https://feeds.example.com/show.xml", True),
        ("https://feeds.example.com/show.RSS?fmt=1", True),  # case-insensitive, query ignored
        ("https://feeds.example.com/feed.atom", True),
        ("https://feeds.example.com/54nAGcIl", True),  # extensionless feed
        ("https://cdn.example.com/show/ep.mp3", False),  # direct media — never probed
        ("https://example.com/notes.txt", False),
    ],
)
def test_looks_like_feed_url(url, feed_shaped):
    assert feed._looks_like_feed_url(url) is feed_shaped


# --- feed_episode_urls (gate + fetch + parse) ---------------------------------


def test_feed_episode_urls_skips_direct_media_without_fetching(monkeypatch):
    def _boom(url):
        raise AssertionError("a direct media URL must not be fetched")

    monkeypatch.setattr(feed, "_fetch", _boom)
    assert feed.feed_episode_urls("https://cdn.example.com/ep.mp3") is None


def test_feed_episode_urls_skips_ytdlp_page_without_fetching(monkeypatch):
    def _boom(url):
        raise AssertionError("a yt-dlp page URL must not be fetched")

    monkeypatch.setattr(feed, "_fetch", _boom)
    assert feed.feed_episode_urls("https://youtu.be/abc") is None


def test_feed_episode_urls_returns_episodes_for_a_feed(monkeypatch):
    monkeypatch.setattr(feed, "_fetch", lambda url: _RSS)
    assert feed.feed_episode_urls("https://feeds.example.com/show") == [
        "https://cdn.example.com/ep2.mp3",
        "https://cdn.example.com/ep1.mp3?token=a&b=2",
    ]


def test_feed_episode_urls_returns_none_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(feed, "_fetch", lambda url: None)
    assert feed.feed_episode_urls("https://feeds.example.com/show") is None


def test_feed_episode_urls_returns_none_for_non_feed_body(monkeypatch):
    monkeypatch.setattr(feed, "_fetch", lambda url: "<html>not a feed</html>")
    assert feed.feed_episode_urls("https://example.com/page") is None


# --- _fetch (httpx, faked offline) --------------------------------------------


class _FakeStream:
    def __init__(
        self, *, status=200, content_type="application/rss+xml", chunks=(b"<rss/>",), location=None
    ):
        self.status_code = status
        self.is_success = 200 <= status < 300  # mirror httpx.Response.is_success
        self.headers = {"content-type": content_type}
        if location is not None:
            self.headers["location"] = location
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        yield from self._chunks


class _FakeClient:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url):
        return self._stream


def _patch_client(monkeypatch, stream):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return _FakeClient(stream)

    monkeypatch.setattr(httpx, "Client", factory)
    return captured


class _SequenceClient:
    """A client whose successive `stream` calls return the given streams in order,
    so a redirect hop and its destination can be distinguished."""

    def __init__(self, streams):
        self._streams = list(streams)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url):
        return self._streams.pop(0)


def test_fetch_follows_redirect_to_final_body(monkeypatch):
    # A 301 to a public CDN is followed to the real feed body — proves the status is
    # recognized as a redirect (not parsed as the body itself).
    streams = [
        _FakeStream(status=301, location="https://cdn.example.com/final"),
        _FakeStream(chunks=(b"<rss>real</rss>",)),
    ]
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: _SequenceClient(streams))
    assert feed._fetch("https://feeds.example.com/show") == "<rss>real</rss>"


def test_fetch_returns_decoded_body(monkeypatch):
    captured = _patch_client(monkeypatch, _FakeStream(chunks=(b"<rss>", b"</rss>")))
    assert feed._fetch("https://feeds.example.com/show") == "<rss></rss>"
    # Redirects are followed manually (per-hop SSRF check), so the client itself
    # must not auto-follow.
    assert captured["follow_redirects"] is False
    assert captured["timeout"] == feed._FETCH_TIMEOUT_SECONDS


def test_fetch_returns_none_on_http_error_status(monkeypatch):
    _patch_client(monkeypatch, _FakeStream(status=404))
    assert feed._fetch("https://feeds.example.com/missing") is None


def test_fetch_treats_exactly_400_as_error(monkeypatch):
    # The >= 400 boundary: a 400 is an error, not a body to parse.
    _patch_client(monkeypatch, _FakeStream(status=400))
    assert feed._fetch("https://feeds.example.com/bad") is None


@pytest.mark.parametrize("content_type", ["audio/mpeg", "video/mp4", "image/png"])
def test_fetch_skips_binary_media_content_types(monkeypatch, content_type):
    _patch_client(monkeypatch, _FakeStream(content_type=content_type, chunks=(b"\x00\x01",)))
    assert feed._fetch("https://cdn.example.com/file") is None


def test_fetch_truncates_at_the_byte_cap(monkeypatch):
    monkeypatch.setattr(feed, "_MAX_FEED_BYTES", 4)
    _patch_client(monkeypatch, _FakeStream(chunks=(b"aaa", b"bbb", b"ccc")))
    # Reads until the running total reaches the cap, then stops — never the third chunk.
    assert feed._fetch("https://feeds.example.com/big") == "aaabbb"


def test_fetch_returns_none_on_network_error(monkeypatch):
    def _raise(**kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", _raise)
    assert feed._fetch("https://feeds.example.com/show") is None


def test_fetch_returns_none_for_internal_host(monkeypatch):
    # A feed URL whose host resolves to an internal address is refused (None) so the
    # local fetch never reaches it — it falls through to the API's server-side fetch.
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["169.254.169.254"])
    _patch_client(monkeypatch, _FakeStream())
    assert feed._fetch("https://feeds.internal.example/show") is None


def test_fetch_returns_none_on_redirect_to_internal_host(monkeypatch):
    # A public feed URL that redirects to an internal address is caught on the hop.
    monkeypatch.setattr(
        ssrf,
        "_resolve_host",
        lambda host: ["169.254.169.254"] if "internal" in host else ["93.184.216.34"],
    )
    _patch_client(monkeypatch, _FakeStream(status=301, location="http://internal.host/meta"))
    assert feed._fetch("https://feeds.example.com/show") is None


def test_fetch_returns_none_on_redirect_loop(monkeypatch):
    # A redirect that never resolves is bounded by the hop cap and gives up (None).
    _patch_client(monkeypatch, _FakeStream(status=301, location="https://feeds.example.com/again"))
    assert feed._fetch("https://feeds.example.com/show") is None


# --- expand_sources seam ------------------------------------------------------


def test_expand_sources_routes_feed_url_to_batch(monkeypatch):
    monkeypatch.setattr(
        feed, "feed_episode_urls", lambda url: ["https://x/a.mp3", "https://x/b.mp3"]
    )
    assert transcribe_sources.expand_sources(
        ["https://feeds.example.com/show"], from_stdin=False, sample=False
    ) == ["https://x/a.mp3", "https://x/b.mp3"]


def test_expand_sources_skips_feed_probe_when_detect_feeds_false(monkeypatch):
    def _boom(url):
        raise AssertionError("feed detection must be skipped when detect_feeds is False")

    monkeypatch.setattr(feed, "feed_episode_urls", _boom)
    assert (
        transcribe_sources.expand_sources(
            ["https://feeds.example.com/show"], from_stdin=False, sample=False, detect_feeds=False
        )
        is None
    )


# --- end-to-end CLI -----------------------------------------------------------


def _auth():
    config.set_api_key("default", "sk_live")


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _patch_transcribe(mocker, monkeypatch):
    seen = []

    def fake(api_key, audio, *, config):
        seen.append(audio)
        t = mocker.MagicMock()
        t.id = f"t_{audio}"
        t.text = f"text of {audio}"
        t.status = "completed"
        t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
        return t

    monkeypatch.setattr(_TRANSCRIBE, fake)
    return seen


def test_transcribe_feed_url_batches_every_episode(mocker, monkeypatch):
    _auth()
    monkeypatch.setattr(feed, "_fetch", lambda url: _RSS)
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "https://feeds.example.com/show", "--json"])
    assert result.exit_code == 0
    # Each episode enclosure was transcribed directly (the API fetches the URL — no
    # yt-dlp download), and the feed XML itself was never sent as a source. Batch
    # workers finish in any order, so compare as a set.
    assert sorted(seen) == sorted(
        [
            "https://cdn.example.com/ep2.mp3",
            "https://cdn.example.com/ep1.mp3?token=a&b=2",
        ]
    )
    statuses = [json.loads(line)["status"] for line in result.output.splitlines()]
    assert statuses == ["completed", "completed"]


def test_transcribe_non_feed_url_stays_single_source(mocker, monkeypatch):
    # A direct audio URL is passed straight to the API, not expanded into a batch.
    _auth()
    monkeypatch.setattr(
        feed, "_fetch", lambda url: (_ for _ in ()).throw(AssertionError("no fetch"))
    )
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "https://example.com/episode.mp3", "-o", "id"])
    assert result.exit_code == 0
    assert seen == ["https://example.com/episode.mp3"]
    assert result.output.strip() == "t_https://example.com/episode.mp3"


def test_transcribe_feed_url_show_code_does_not_fetch(monkeypatch):
    _auth()

    def _boom(url):
        raise AssertionError("--show-code must not touch the network")

    monkeypatch.setattr(feed, "_fetch", _boom)
    result = runner.invoke(app, ["transcribe", "https://feeds.example.com/show", "--show-code"])
    assert result.exit_code == 0
    assert "import assemblyai as aai" in result.output  # generated SDK code, no network probe
