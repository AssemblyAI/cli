"""Batch-mode `assembly transcribe --llm`: the per-source LLM chain, its sidecar
`transform` record, and chain-only resume (see test_transcribe_batch.py for the
core batch/sidecar behavior).
"""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from aai_cli.app.transcribe import batch as transcribe_batch
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()

_TRANSCRIBE = "aai_cli.app.transcribe.run.client.transcribe"


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    # Batch sources and sidecars are resolved relative to the working directory;
    # isolate each test in its own tmp cwd.
    monkeypatch.chdir(tmp_path)


def _auth():
    config.set_api_key("default", "sk_live")


def _fake_transcript(mocker, source="x"):
    t = mocker.MagicMock()
    t.id = f"t_{source}"
    t.text = f"text of {source}"
    t.status = "completed"
    t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
    return t


def _patch_transcribe(mocker, monkeypatch):
    """Patch client.transcribe with a fake that records the audio args it saw."""
    seen = []

    def fake(api_key, audio, *, config):
        seen.append(audio)
        return _fake_transcript(mocker, audio)

    monkeypatch.setattr(_TRANSCRIBE, fake)
    return seen


def _ndjson(result):
    return [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]


# --- the --llm chain in batch mode ------------------------------------------------

_TRANSFORM = "aai_cli.core.llm.transform_transcript"


def _patch_transform(monkeypatch):
    """Patch the gateway call under run_chain_steps, recording each call's args."""
    calls = []

    def fake(api_key, *, prompt, model, max_tokens, transcript_id=None, transcript_text=None):
        calls.append(
            {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "transcript_id": transcript_id,
            }
        )
        return f"resp:{prompt}"

    monkeypatch.setattr(_TRANSFORM, fake)
    return calls


def _sidecar_record(tmp_path, name, data, transform=None):
    record = {
        "source": name,
        "id": "t_old",
        "status": "completed",
        "transcript": {"id": "t_old"},
        "source_sha256": hashlib.sha256(data).hexdigest(),
    }
    if transform is not None:
        record["transform"] = transform
    (tmp_path / f"{name}.aai.json").write_text(json.dumps(record))


def test_batch_llm_stores_chain_steps_in_each_sidecar(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "b.mp3").write_bytes(b"bbb")
    _patch_transcribe(mocker, monkeypatch)
    calls = _patch_transform(monkeypatch)
    result = runner.invoke(
        app,
        [
            "transcribe",
            "*.mp3",
            "--llm",
            "Summarize",
            "--model",
            "gw-x",
            "--max-tokens",
            "7",
            "--json",
        ],
    )
    assert result.exit_code == 0
    # One chain per source, against that source's transcript id, with the gateway
    # flags passed through (non-default on purpose).
    assert {(c["transcript_id"], c["prompt"], c["model"], c["max_tokens"]) for c in calls} == {
        ("t_a.mp3", "Summarize", "gw-x", 7),
        ("t_b.mp3", "Summarize", "gw-x", 7),
    }
    sidecar = json.loads((tmp_path / "a.mp3.aai.json").read_text())
    assert sidecar["transform"] == {
        "model": "gw-x",
        "prompts": ["Summarize"],
        "steps": [{"prompt": "Summarize", "output": "resp:Summarize"}],
    }
    assert sidecar["transcript"]["id"] == "t_a.mp3"  # transcription payload kept alongside
    records = {r["source"]: r for r in _ndjson(result)}
    assert records["a.mp3"] == {
        "type": "result",
        "source": "a.mp3",
        "status": "completed",
        "id": "t_a.mp3",
        "sidecar": "a.mp3.aai.json",
    }


def test_failed_llm_chain_leaves_resumable_transcription(tmp_path, mocker, monkeypatch):
    from aai_cli.core.errors import APIError

    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    seen = _patch_transcribe(mocker, monkeypatch)

    def boom(api_key, **kwargs):
        raise APIError("gateway exploded")

    monkeypatch.setattr(_TRANSFORM, boom)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--llm", "Summarize", "--json"])
    assert result.exit_code == 1
    records = {r["source"]: r for r in _ndjson(result) if "source" in r}
    assert records["a.mp3"]["status"] == "failed"
    assert records["a.mp3"]["error"] == "gateway exploded"
    # The transcription itself was recorded before the chain ran...
    sidecar = json.loads((tmp_path / "a.mp3.aai.json").read_text())
    assert sidecar["status"] == "completed"
    assert "transform" not in sidecar

    # ...so the retry pays only for the LLM step: no second transcription, the
    # chain anchored on the recorded transcript id, and the transcription payload
    # kept alongside the new transform.
    seen.clear()
    calls = _patch_transform(monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--llm", "Summarize", "--json"])
    assert result.exit_code == 0
    assert seen == []
    assert [c["transcript_id"] for c in calls] == ["t_a.mp3"]
    sidecar = json.loads((tmp_path / "a.mp3.aai.json").read_text())
    assert sidecar["transform"]["steps"] == [{"prompt": "Summarize", "output": "resp:Summarize"}]
    assert sidecar["transcript"] == {
        "id": "t_a.mp3",
        "text": "text of a.mp3",
        "status": "completed",
    }
    assert sidecar["source_sha256"] == hashlib.sha256(b"aaa").hexdigest()
    assert {r["source"]: r["status"] for r in _ndjson(result)} == {"a.mp3": "completed"}


def test_rerun_with_same_llm_chain_skips_entirely(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _sidecar_record(
        tmp_path,
        "a.mp3",
        b"aaa",
        transform={"model": "gw-x", "prompts": ["Summarize"], "steps": []},
    )
    seen = _patch_transcribe(mocker, monkeypatch)
    calls = _patch_transform(monkeypatch)
    result = runner.invoke(
        app, ["transcribe", "*.mp3", "--llm", "Summarize", "--model", "gw-x", "--json"]
    )
    assert result.exit_code == 0
    assert seen == []
    assert calls == []
    assert _ndjson(result) == [
        {
            "type": "result",
            "source": "a.mp3",
            "status": "skipped",
            "id": "t_old",
            "sidecar": "a.mp3.aai.json",
        }
    ]


@pytest.mark.parametrize(
    "stored",
    [
        {"model": "gw-x", "prompts": ["old prompt"], "steps": []},  # prompts differ
        {"model": "gw-other", "prompts": ["Summarize"], "steps": []},  # model differs
        None,  # pre---llm sidecar with no transform at all
    ],
)
def test_changed_llm_chain_replays_llm_only(tmp_path, mocker, monkeypatch, stored):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _sidecar_record(tmp_path, "a.mp3", b"aaa", transform=stored)
    seen = _patch_transcribe(mocker, monkeypatch)
    calls = _patch_transform(monkeypatch)
    result = runner.invoke(
        app, ["transcribe", "*.mp3", "--llm", "Summarize", "--model", "gw-x", "--json"]
    )
    assert result.exit_code == 0
    assert seen == []  # the transcription is already paid for
    assert [c["transcript_id"] for c in calls] == ["t_old"]
    sidecar = json.loads((tmp_path / "a.mp3.aai.json").read_text())
    assert sidecar["transform"] == {
        "model": "gw-x",
        "prompts": ["Summarize"],
        "steps": [{"prompt": "Summarize", "output": "resp:Summarize"}],
    }
    assert sidecar["transcript"] == {"id": "t_old"}  # the resumed record's payload survives


def test_resumed_record_without_id_retranscribes_for_llm(tmp_path, mocker, monkeypatch):
    # A completed sidecar that never recorded a transcript id can't anchor a
    # server-side chain: the source is transcribed again.
    _auth()
    url = "https://example.com/ep.mp3"
    transcribe_batch.sidecar_path(url).write_text(json.dumps({"status": "completed"}))
    seen = _patch_transcribe(mocker, monkeypatch)
    calls = _patch_transform(monkeypatch)
    result = runner.invoke(
        app, ["transcribe", "--from-stdin", "--llm", "Summarize", "--json"], input=url + "\n"
    )
    assert result.exit_code == 0
    assert seen == [url]
    assert [c["transcript_id"] for c in calls] == [f"t_{url}"]


def test_force_with_llm_retranscribes_and_reruns_chain(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _sidecar_record(
        tmp_path,
        "a.mp3",
        b"aaa",
        transform={"model": "gw-x", "prompts": ["Summarize"], "steps": []},
    )
    seen = _patch_transcribe(mocker, monkeypatch)
    calls = _patch_transform(monkeypatch)
    result = runner.invoke(
        app, ["transcribe", "*.mp3", "--force", "--llm", "Summarize", "--model", "gw-x", "--json"]
    )
    assert result.exit_code == 0
    assert seen == ["a.mp3"]
    assert [c["transcript_id"] for c in calls] == ["t_a.mp3"]
