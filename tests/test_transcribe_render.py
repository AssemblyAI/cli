from types import SimpleNamespace

from rich.console import Console

from aai_cli.app import transcribe_render as tr
from aai_cli.ui import theme


def _render(transcript) -> str:
    console = Console(width=80, force_terminal=False)
    with console.capture() as cap:
        tr.render_transcript_result(transcript, console)
    return cap.get()


def _render_styled(transcript) -> str:
    # Themed console (as the CLI uses) + force_terminal so Rich emits ANSI to assert on.
    console = theme.make_console(force_terminal=True, color_system="truecolor")
    with console.capture() as cap:
        tr.render_transcript_result(transcript, console)
    return cap.get()


def test_fmt_ms_formats_minutes_and_seconds():
    # Pins the mm:ss math: values straddling a 60s boundary distinguish // from %
    # and the literal 60 (e.g. 120s is 02:00, not 01:xx).
    assert tr._fmt_ms(0) == "00:00"
    assert tr._fmt_ms(133000) == "02:13"
    assert tr._fmt_ms(120000) == "02:00"
    assert tr._fmt_ms(605000) == "10:05"


def test_per_speaker_lines_are_styled():
    transcript = SimpleNamespace(
        text="flat",
        utterances=[SimpleNamespace(speaker="A", text="hello")],
    )
    out = _render_styled(transcript)
    assert "\x1b[" in out  # speaker label is color-styled, not plain text


def test_flat_text_with_brackets_is_not_parsed_as_markup():
    # Transcript content containing [tags] must render literally, not as Rich markup.
    out = _render(SimpleNamespace(text="say [bold]hi[/bold] now"))
    assert "[bold]hi[/bold]" in out


def test_renders_text_only_when_no_analysis():
    out = _render(SimpleNamespace(text="hello world"))
    assert "hello world" in out
    assert "Summary" not in out


def test_render_helpers_handle_invalid_dynamic_values():
    # Numeric coercion now lives in jsonshape.as_int/as_float (see test_jsonshape).
    assert tr._mapping("bad") == {}


def test_renders_per_speaker_utterances():
    transcript = SimpleNamespace(
        text="flat",
        utterances=[
            SimpleNamespace(speaker="A", text="hello"),
            SimpleNamespace(speaker="B", text="hi"),
        ],
    )
    out = _render(transcript)
    assert "Speaker A: hello" in out
    assert "Speaker B: hi" in out
    assert "flat" not in out


def test_renders_summary_and_chapters():
    transcript = SimpleNamespace(
        text="t",
        summary="A short summary.",
        chapters=[SimpleNamespace(start=0, end=133000, headline="Intro", gist="i", summary="s")],
    )
    out = _render(transcript)
    assert "Summary:" in out
    assert "A short summary." in out
    assert "Chapters:" in out
    assert "Intro" in out
    assert "00:00" in out and "02:13" in out  # 133000ms -> 02:13


def test_renders_sentiment_aggregate():
    transcript = SimpleNamespace(
        text="t",
        sentiment_analysis=[
            SimpleNamespace(text="a", sentiment=SimpleNamespace(value="POSITIVE")),
            SimpleNamespace(text="b", sentiment=SimpleNamespace(value="POSITIVE")),
            SimpleNamespace(text="c", sentiment=SimpleNamespace(value="NEGATIVE")),
        ],
    )
    out = _render(transcript).lower()
    assert "sentiment:" in out
    # Largest-remainder rounding makes the percentages sum to exactly 100:
    # 2 of 3 positive -> 67% (66.67 takes the leftover point), 1 of 3 -> 33%.
    assert "67% positive" in out
    assert "33% negative" in out


def test_renders_entities_topics_content_safety_highlights():
    transcript = SimpleNamespace(
        text="t",
        entities=[SimpleNamespace(entity_type=SimpleNamespace(value="person_name"), text="Ada")],
        iab_categories=SimpleNamespace(summary={"Technology": 0.91, "Health": 0.12}),
        content_safety=SimpleNamespace(summary={"profanity": 0.40, "alcohol": 0.10}),
        auto_highlights=SimpleNamespace(
            results=[SimpleNamespace(text="key phrase", count=3, rank=0.9)]
        ),
    )
    out = _render(transcript)
    assert "Entities:" in out and "Ada" in out
    assert "Topics:" in out and "Technology" in out
    assert "Content Safety:" in out and "profanity" in out
    assert "Highlights:" in out and "key phrase" in out
    # Relevance/confidence are rendered to 2 decimals...
    assert "Technology (0.91)" in out
    assert "profanity (0.40)" in out
    # ...and both lists are sorted most-relevant-first (pins reverse=True).
    assert out.index("Technology") < out.index("Health")
    assert out.index("profanity") < out.index("alcohol")


def test_fmt_ms_rolls_over_to_hours():
    # Sub-hour stays MM:SS; at an hour the format grows an hours field so a
    # 1h chapter reads 1:00:00 instead of the ambiguous 60:00.
    from aai_cli.app.transcribe_render import _fmt_ms

    assert _fmt_ms(0) == "00:00"
    assert _fmt_ms(59_000) == "00:59"
    assert _fmt_ms(3_599_000) == "59:59"
    assert _fmt_ms(3_600_000) == "1:00:00"
    assert _fmt_ms(4_530_000) == "1:15:30"
    assert _fmt_ms(3_720_000) == "1:02:00"  # minutes division boundary past the hour


def test_sentiment_percentages_always_sum_to_100():
    # Largest-remainder rounding: three equal counts split 34/33/33, not 33x3=99.
    from collections import Counter

    from aai_cli.app.transcribe_render import _percentages

    pcts = _percentages(Counter(positive=1, neutral=1, negative=1))
    assert sum(pcts.values()) == 100
    assert sorted(pcts.values()) == [33, 33, 34]
    assert _percentages(Counter()) == {}
    assert _percentages(Counter(positive=2, negative=1)) == {"positive": 67, "negative": 33}
    assert _percentages(Counter(positive=4)) == {"positive": 100}
    # The leftover point goes to the largest *fraction* (positive, .714), which here
    # is not the largest share -- pins the remainder ordering.
    assert _percentages(Counter(positive=5, negative=9)) == {"positive": 36, "negative": 64}
