from aai_cli.init import tunnel


def test_tunnel_command_shape():
    assert tunnel.tunnel_command(3000) == [
        "cloudflared",
        "tunnel",
        "--url",
        "http://localhost:3000",
    ]


def test_find_url_matches_cloudflared_banner():
    line = "2026-06-09T12:00:00Z INF |  https://happy-cat-tree.trycloudflare.com  |"
    assert tunnel.find_url(line) == "https://happy-cat-tree.trycloudflare.com"


def test_find_url_none_when_absent():
    assert tunnel.find_url("INF Registered tunnel connection") is None


def test_await_url_found_immediately(tmp_path):
    log = tmp_path / "cf.log"
    log.write_text("starting\nhttps://abc-def.trycloudflare.com\n")
    assert tunnel.await_url(log, timeout=5.0) == "https://abc-def.trycloudflare.com"


def test_await_url_times_out(tmp_path):
    log = tmp_path / "cf.log"
    log.write_text("no url yet")
    assert tunnel.await_url(log, timeout=0.0) is None


def test_await_url_polls_until_written(tmp_path):
    log = tmp_path / "cf.log"
    log.write_text("")
    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        log.write_text("https://later-slug.trycloudflare.com")

    url = tunnel.await_url(log, timeout=5.0, sleep=fake_sleep)
    assert url == "https://later-slug.trycloudflare.com"
    assert calls["n"] == 1
