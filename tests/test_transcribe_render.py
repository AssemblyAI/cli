from types import SimpleNamespace

from rich.console import Console

from aai_cli import theme
from aai_cli import transcribe_render as tr


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
    assert tr._mapping("bad") == {}
    assert tr._int_value(True) == 0
    assert tr._int_value(12) == 12
    assert tr._int_value(12.9) == 12
    assert tr._int_value("13") == 13
    assert tr._int_value("bad") == 0
    assert tr._int_value(object()) == 0
    assert tr._float_value(True) == 0.0
    assert tr._float_value(1) == 1.0
    assert tr._float_value("1.5") == 1.5
    assert tr._float_value("bad") == 0.0
    assert tr._float_value(object()) == 0.0


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
    out = _render(transcript)
    assert "Sentiment:" in out
    assert "positive" in out.lower()


def test_renders_entities_topics_content_safety_highlights():
    transcript = SimpleNamespace(
        text="t",
        entities=[SimpleNamespace(entity_type=SimpleNamespace(value="person_name"), text="Ada")],
        iab_categories=SimpleNamespace(summary={"Technology": 0.91}),
        content_safety=SimpleNamespace(summary={"profanity": 0.4}),
        auto_highlights=SimpleNamespace(
            results=[SimpleNamespace(text="key phrase", count=3, rank=0.9)]
        ),
    )
    out = _render(transcript)
    assert "Entities:" in out and "Ada" in out
    assert "Topics:" in out and "Technology" in out
    assert "Content Safety:" in out and "profanity" in out
    assert "Highlights:" in out and "key phrase" in out
