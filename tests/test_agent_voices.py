import dataclasses

import pytest

from aai_cli.agent import voices

# The catalog pinned exactly, split by language group: --list-voices renders these
# groups and the names back the --voice typo check, so a drifted entry is a bug.
_ENGLISH_NAMES = [
    "ivy",
    "james",
    "tyler",
    "winter",
    "sam",
    "mia",
    "bella",
    "david",
    "jack",
    "kyle",
    "helen",
    "martha",
    "river",
    "emma",
    "victor",
    "eleanor",
    "sophie",
    "oliver",
]
_MULTILINGUAL_NAMES = [
    "arjun",
    "ethan",
    "dmitri",
    "lukas",
    "lena",
    "pierre",
    "mina",
    "ren",
    "mei",
    "joon",
    "giulia",
    "luca",
    "lucia",
    "hana",
    "mateo",
    "diego",
]


def test_voice_catalog_matches_known_groups():
    assert [v.name for v in voices.VOICES if v.language == voices.ENGLISH] == _ENGLISH_NAMES
    assert [v.name for v in voices.VOICES if v.language == voices.MULTILINGUAL] == (
        _MULTILINGUAL_NAMES
    )
    # Every voice belongs to exactly one of the two known groups.
    assert {v.language for v in voices.VOICES} == {"English", "Multilingual"}


def test_voice_entries_are_immutable():
    # frozen=True: the catalog is shared module state, so entries must not be mutable.
    field = "name"  # via setattr: a literal `voice.name = …` is rejected by type checkers
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(voices.VOICES[0], field, "hacked")


def test_voice_names_are_unique_and_in_catalog_order():
    assert voices.VOICE_NAMES == _ENGLISH_NAMES + _MULTILINGUAL_NAMES
    assert len(voices.VOICE_NAMES) == len(set(voices.VOICE_NAMES))


def test_default_voice_is_in_voices():
    assert voices.DEFAULT_VOICE == "ivy"
    assert voices.DEFAULT_VOICE in voices.VOICE_NAMES


def test_format_voice_list_groups_by_language():
    blocks = voices.format_voice_list().split("\n\n")
    assert [block.splitlines()[0] for block in blocks] == ["English:", "Multilingual:"]
    english, multilingual = blocks
    # Names are indented under their group header, one per line, in catalog order.
    assert english.splitlines()[1:] == [f"  {name}" for name in _ENGLISH_NAMES]
    assert multilingual.splitlines()[1:] == [f"  {name}" for name in _MULTILINGUAL_NAMES]
