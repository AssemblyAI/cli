from aai_cli.agent import voices


def test_voices_includes_default():
    assert "ivy" in voices.VOICES


def test_voices_are_unique_and_nonempty():
    assert voices.VOICES
    assert len(voices.VOICES) == len(set(voices.VOICES))


def test_format_voice_list_mentions_voices():
    out = voices.format_voice_list()
    assert "ivy" in out
    assert "james" in out


def test_default_voice_is_in_voices():
    assert voices.DEFAULT_VOICE in voices.VOICES


def test_format_voice_list_contains_all_voices():
    out = voices.format_voice_list()
    for v in voices.VOICES:
        assert v in out
