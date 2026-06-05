from aai_cli import follow, output


def test_json_mode_emits_ndjson_per_refresh(monkeypatch):
    emitted = []
    monkeypatch.setattr(output, "emit_ndjson", lambda obj: emitted.append(obj))

    with follow.FollowRenderer(json_mode=True) as r:
        r("first answer", 1)
        r("second answer", 2)

    assert emitted == [
        {"turns": 1, "output": "first answer"},
        {"turns": 2, "output": "second answer"},
    ]


def test_json_mode_does_not_start_a_live_region():
    r = follow.FollowRenderer(json_mode=True)
    with r:
        assert r._live is None


class _FakeLive:
    def __init__(self, *args, **kwargs):
        self.started = False
        self.stopped = False
        self.updates = []

    def start(self):
        self.started = True

    def update(self, renderable, refresh):
        self.updates.append(renderable)

    def stop(self):
        self.stopped = True


def test_terminal_mode_renders_panels_and_prints_final(monkeypatch):
    fake = _FakeLive()
    monkeypatch.setattr(follow, "Live", lambda *a, **k: fake)
    printed = []
    monkeypatch.setattr(output.console, "print", lambda renderable: printed.append(renderable))

    with follow.FollowRenderer(json_mode=False) as r:
        r("hello", 1)
        r("world", 3)
        assert fake.started is True
        assert len(fake.updates) == 2

    assert fake.stopped is True
    # the final panel is reprinted to the normal screen as scrollback
    assert printed == [fake.updates[-1]]


def test_terminal_mode_panel_title_pluralizes_turns(monkeypatch):
    fake = _FakeLive()
    monkeypatch.setattr(follow, "Live", lambda *a, **k: fake)
    monkeypatch.setattr(output.console, "print", lambda renderable: None)

    with follow.FollowRenderer(json_mode=False) as r:
        r("a", 1)  # singular
        r("b", 2)  # plural

    assert fake.updates[0].title == "scribe · 1 turn"
    assert fake.updates[1].title == "scribe · 2 turns"


def test_terminal_mode_empty_answer_shows_placeholder(monkeypatch):
    fake = _FakeLive()
    monkeypatch.setattr(follow, "Live", lambda *a, **k: fake)
    monkeypatch.setattr(output.console, "print", lambda renderable: None)

    with follow.FollowRenderer(json_mode=False) as r:
        r("", 1)

    # an empty answer renders the "…" placeholder rather than a blank panel
    assert "…" in fake.updates[0].renderable
