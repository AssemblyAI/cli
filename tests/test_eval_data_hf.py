"""Hugging Face dataset loading for `assembly eval` (`aai_cli.eval_data`).

Runs against an httpx MockTransport (the test_auth_ams.py pattern), so
pytest-socket stays armed; local-manifest paths live in
test_eval_data_manifest.py.
"""

import dataclasses

import httpx2 as httpx
import pytest

from aai_cli import der, eval_data
from aai_cli.errors import APIError, UsageError

# ------------------------------------------------------- Hugging Face datasets


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(eval_data.httpx, "Client", fake_client)


def _audio_cell(url="https://hf.example/audio/0.wav"):
    return [{"src": url, "type": "audio/wav"}]


def _hf_handler(monkeypatch, *, splits, rows, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/splits":
            return httpx.Response(200, json={"splits": splits})
        return httpx.Response(200, json={"rows": rows})

    _patch_transport(monkeypatch, handler)


_ONE_SPLIT = [{"dataset": "org/ds", "config": "default", "split": "test"}]


def _hf_row(idx=0, **cells):
    cells.setdefault("audio", _audio_cell())
    cells.setdefault("text", "hello world")
    return {"row_idx": idx, "row": cells, "truncated_cells": []}


def test_hf_happy_path_resolves_columns_and_requests(monkeypatch):
    seen = []
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row()], seen=seen)
    data = eval_data.load("org/ds", limit=5)
    assert data.label == "org/ds · default/test"
    item = data.items[0]
    assert item.item_id == "test[0]"
    assert item.audio == "https://hf.example/audio/0.wav"
    assert item.reference == "hello world"
    splits_req, rows_req = seen
    assert splits_req.url.path == "/splits"
    assert splits_req.url.params["dataset"] == "org/ds"
    assert "authorization" not in splits_req.headers
    params = rows_req.url.params
    assert params["config"] == "default"
    assert params["split"] == "test"
    assert params["offset"] == "0"
    assert params["length"] == "5"


def test_hf_token_sent_when_set(monkeypatch):
    seen = []
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row()], seen=seen)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    eval_data.load("org/ds", limit=1)
    assert seen[0].headers["authorization"] == "Bearer hf_secret"


def test_hf_bare_string_audio_cell(monkeypatch):
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row(audio="https://hf.example/a.mp3")])
    assert eval_data.load("org/ds", limit=1).items[0].audio == "https://hf.example/a.mp3"


def test_hf_audio_cell_skips_sources_without_src(monkeypatch):
    cell = [{"type": "audio/wav"}, {"src": "https://hf.example/b.wav"}]
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row(audio=cell)])
    assert eval_data.load("org/ds", limit=1).items[0].audio == "https://hf.example/b.wav"


def test_hf_unusable_audio_cell_errors(monkeypatch):
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row(audio=123)])
    with pytest.raises(APIError) as exc:
        eval_data.load("org/ds", limit=1)
    assert "test[0]" in exc.value.message and "no audio URL" in exc.value.message


def test_hf_speaker_rows(monkeypatch):
    row = _hf_row(speakers=["alice"], timestamps_start=[0.5], timestamps_end=[2.0])
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[row])
    data = eval_data.load("org/ds", limit=1, with_speakers=True)
    assert data.items[0].turns == [der.Turn(speaker="alice", start=0.5, end=2.0)]


def test_hf_single_named_config_auto_picked(monkeypatch):
    seen = []
    splits = [{"config": "clean", "split": "test"}]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()], seen=seen)
    assert eval_data.load("org/ds", limit=1).label == "org/ds · clean/test"
    assert seen[1].url.params["config"] == "clean"


def test_hf_default_config_wins_among_many(monkeypatch):
    splits = [
        {"config": "clean", "split": "test"},
        {"config": "default", "split": "test"},
    ]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()])
    assert eval_data.load("org/ds", limit=1).label == "org/ds · default/test"


def test_hf_many_configs_without_default_require_subset(monkeypatch):
    splits = [{"config": "clean", "split": "test"}, {"config": "other", "split": "test"}]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()])
    with pytest.raises(UsageError) as exc:
        eval_data.load("org/ds", limit=1)
    assert "clean, other" in exc.value.message
    assert exc.value.suggestion is not None and "--subset" in exc.value.suggestion


def test_hf_explicit_subset(monkeypatch):
    seen = []
    splits = [{"config": "clean", "split": "test"}, {"config": "other", "split": "test"}]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()], seen=seen)
    eval_data.load("org/ds", subset="other", limit=1)
    assert seen[1].url.params["config"] == "other"


def test_hf_unknown_subset_lists_options(monkeypatch):
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row()])
    with pytest.raises(UsageError, match="no subset 'nope'"):
        eval_data.load("org/ds", subset="nope", limit=1)


def test_hf_test_split_preferred(monkeypatch):
    seen = []
    splits = [
        {"config": "default", "split": "train"},
        {"config": "default", "split": "test"},
    ]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()], seen=seen)
    eval_data.load("org/ds", limit=1)
    assert seen[1].url.params["split"] == "test"


def test_hf_single_split_used_when_no_test(monkeypatch):
    splits = [{"config": "default", "split": "validation"}]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()])
    assert eval_data.load("org/ds", limit=1).label == "org/ds · default/validation"


def test_hf_many_splits_without_test_require_split(monkeypatch):
    splits = [
        {"config": "default", "split": "train"},
        {"config": "default", "split": "validation"},
    ]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()])
    with pytest.raises(UsageError) as exc:
        eval_data.load("org/ds", limit=1)
    assert "train, validation" in exc.value.message
    assert exc.value.suggestion is not None and "--split" in exc.value.suggestion


def test_hf_explicit_split_must_exist_for_the_subset(monkeypatch):
    # 'train' exists, but only under the other config — split filtering is per-config.
    splits = [
        {"config": "default", "split": "test"},
        {"config": "other", "split": "train"},
    ]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()])
    with pytest.raises(UsageError, match="no 'train' split in subset 'default'"):
        eval_data.load("org/ds", split="train", limit=1)


def test_hf_explicit_split_used(monkeypatch):
    seen = []
    splits = [
        {"config": "default", "split": "train"},
        {"config": "default", "split": "test"},
    ]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()], seen=seen)
    eval_data.load("org/ds", split="train", limit=1)
    assert seen[1].url.params["split"] == "train"


def test_hf_empty_splits_payload(monkeypatch):
    _hf_handler(monkeypatch, splits=[], rows=[])
    with pytest.raises(APIError, match="no splits"):
        eval_data.load("org/ds", limit=1)


def test_hf_no_rows(monkeypatch):
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[])
    with pytest.raises(APIError, match="returned no rows"):
        eval_data.load("org/ds", limit=1)


@pytest.mark.parametrize("status", [401, 403])
def test_hf_auth_failure_suggests_hf_token(monkeypatch, status):
    _patch_transport(monkeypatch, lambda request: httpx.Response(status, json={"error": "gated"}))
    with pytest.raises(APIError) as exc:
        eval_data.load("org/gated", limit=1)
    assert str(status) in exc.value.message
    assert exc.value.suggestion is not None and "HF_TOKEN" in exc.value.suggestion


def test_hf_404_is_a_usage_error_with_detail(monkeypatch):
    _patch_transport(
        monkeypatch, lambda request: httpx.Response(404, json={"error": "Dataset not on the Hub"})
    )
    with pytest.raises(UsageError) as exc:
        eval_data.load("org/nope", limit=1)
    assert "org/nope" in exc.value.message
    assert "Dataset not on the Hub" in exc.value.message
    # The detail is extracted from the JSON body, not the raw body pasted in.
    assert '"error"' not in exc.value.message


def test_hf_server_error_with_plain_text_body(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(500, text="upstream down"))
    with pytest.raises(APIError) as exc:
        eval_data.load("org/ds", limit=1)
    assert "500" in exc.value.message and "upstream down" in exc.value.message


def test_hf_server_error_with_non_object_json_falls_back_to_text(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(500, json=["weird"]))
    with pytest.raises(APIError) as exc:
        eval_data.load("org/ds", limit=1)
    assert '["weird"]' in exc.value.message


def test_hf_invalid_json_success_body(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(200, text="<html>"))
    with pytest.raises(APIError, match="invalid JSON"):
        eval_data.load("org/ds", limit=1)


def test_hf_non_object_json_success_body(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(200, json=["nope"]))
    with pytest.raises(APIError, match="expected an object"):
        eval_data.load("org/ds", limit=1)


def test_hf_network_failure_wrapped(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("boom")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match="Could not reach the Hugging Face datasets server"):
        eval_data.load("org/ds", limit=1)


@pytest.mark.parametrize("bad_id", ["not a dataset", "a/b/c", "org/ds?x=1"])
def test_non_hf_looking_ids_rejected_before_any_request(bad_id):
    # No transport patch: pytest-socket would fail loudly if a request were made.
    with pytest.raises(UsageError, match="neither a local"):
        eval_data.load(bad_id, limit=1)


def test_hf_empty_reference_text_rejected(monkeypatch):
    _hf_handler(monkeypatch, splits=_ONE_SPLIT, rows=[_hf_row(text="…")])
    with pytest.raises(UsageError) as exc:
        eval_data.load("org/ds", limit=1)
    assert "test[0]" in exc.value.message and "empty reference" in exc.value.message


# ----------------------------------------------------------- benchmark aliases


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("librispeech", eval_data.Alias("openslr/librispeech_asr", subset="clean")),
        ("librispeech-other", eval_data.Alias("openslr/librispeech_asr", subset="other")),
        ("tedlium", eval_data.Alias("sanchit-gandhi/tedlium-data")),
        ("earnings22", eval_data.Alias("sanchit-gandhi/earnings22_robust_split")),
        ("spgispeech", eval_data.Alias("kensho/spgispeech", subset="test")),
        ("ami", eval_data.Alias("edinburghcstr/ami", subset="ihm")),
        ("ami-sdm", eval_data.Alias("edinburghcstr/ami", subset="sdm")),
        ("gigaspeech", eval_data.Alias("fixie-ai/gigaspeech", subset="dev", split="dev")),
        ("peoples", eval_data.Alias("fixie-ai/peoples_speech", subset="clean")),
        ("commonvoice", eval_data.Alias("fixie-ai/common_voice_17_0", subset="en")),
        ("voxpopuli", eval_data.Alias("facebook/voxpopuli", subset="en")),
        ("switchboard", eval_data.Alias("hhoangphuoc/switchboard", split="validation")),
        ("expresso", eval_data.Alias("ylacombe/expresso")),
        (
            "loquacious",
            eval_data.Alias("speechbrain/LoquaciousSet", subset="small", audio_column="wav"),
        ),
        ("callhome", eval_data.Alias("talkbank/callhome", subset="eng")),
    ],
)
def test_alias_table_pins_each_benchmark(alias, expected):
    assert eval_data.ALIASES[alias] == expected


def test_alias_table_has_no_unpinned_entries():
    # Every entry is asserted exactly in the parametrized test above.
    assert len(eval_data.ALIASES) == 15


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_alias_entries_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(eval_data.ALIASES["tedlium"], "dataset", "org/other")


def test_alias_expands_to_hub_id_and_defaults(monkeypatch):
    seen = []
    splits = [{"config": "dev", "split": "dev"}, {"config": "dev", "split": "test"}]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()], seen=seen)
    data = eval_data.load("gigaspeech", limit=1)
    assert data.label == "fixie-ai/gigaspeech · dev/dev"
    assert seen[0].url.params["dataset"] == "fixie-ai/gigaspeech"
    params = seen[1].url.params
    assert params["config"] == "dev"
    assert params["split"] == "dev"  # the alias split beats the usual 'test' preference


def test_alias_audio_column_beats_autodetect(monkeypatch):
    # The row also carries an 'audio' column (the auto-detect favorite); the
    # alias's audio_column must still win.
    row = _hf_row(wav=_audio_cell("https://hf.example/from-wav.wav"))
    splits = [{"config": "small", "split": "test"}]
    _hf_handler(monkeypatch, splits=splits, rows=[row])
    item = eval_data.load("loquacious", limit=1).items[0]
    assert item.audio == "https://hf.example/from-wav.wav"


def test_explicit_audio_column_overrides_alias(monkeypatch):
    row = _hf_row(wav=_audio_cell("https://hf.example/from-wav.wav"))
    splits = [{"config": "small", "split": "test"}]
    _hf_handler(monkeypatch, splits=splits, rows=[row])
    item = eval_data.load("loquacious", audio_column="audio", limit=1).items[0]
    assert item.audio == "https://hf.example/audio/0.wav"


def test_explicit_subset_and_split_override_alias(monkeypatch):
    seen = []
    splits = [
        {"config": "en", "split": "test"},
        {"config": "fr", "split": "test"},
        {"config": "fr", "split": "validation"},
    ]
    _hf_handler(monkeypatch, splits=splits, rows=[_hf_row()], seen=seen)
    eval_data.load("commonvoice", subset="fr", split="validation", limit=1)
    params = seen[1].url.params
    assert params["dataset"] == "fixie-ai/common_voice_17_0"
    assert params["config"] == "fr"
    assert params["split"] == "validation"
