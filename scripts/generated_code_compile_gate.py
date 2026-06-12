from __future__ import annotations

import json
import sys
from pathlib import Path

import typer.main
from click.testing import CliRunner

from aai_cli.main import app

_ARG_COUNT = 2
_USAGE_EXIT = 2

# Compile exactly what `assembly … --show-code > script.py` would capture: stdout
# only (stderr carries human chrome like warnings), with telemetry disabled so a
# gate run never mints a device id or spawns a flusher on the host.
_ENV = {"AAI_TELEMETRY_DISABLED": "1"}


def _write_fixture(
    runner: CliRunner,
    out_dir: Path,
    name: str,
    args: tuple[str, ...],
) -> None:
    command = typer.main.get_command(app)
    result = runner.invoke(command, list(args), env=_ENV)
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.output.strip() or str(result.exception)
        raise RuntimeError(f"{name}: {' '.join(args)} failed: {detail}")
    code = result.output
    if not code.strip():
        raise RuntimeError(f"{name}: {' '.join(args)} produced no code")
    (out_dir / f"{name}.py").write_text(code, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != _ARG_COUNT:
        sys.stderr.write("usage: generated_code_compile_gate.py OUT_DIR\n")
        return _USAGE_EXIT
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    transcribe_config = out_dir / "transcribe-config.json"
    transcribe_config.write_text(
        json.dumps({"speaker_labels": True, "summarization": True, "word_boost": ["CLI"]}),
        encoding="utf-8",
    )
    stream_config = out_dir / "stream-config.json"
    stream_config.write_text(
        json.dumps({"sample_rate": 16000, "format_turns": True, "keyterms_prompt": ["CLI"]}),
        encoding="utf-8",
    )
    prompt_file = out_dir / "agent-prompt.txt"
    prompt_file.write_text('Be terse, quote "edge cases", and handle newlines.\n', encoding="utf-8")

    cases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("transcribe-basic", ("transcribe", "audio.mp3", "--show-code")),
        (
            "transcribe-config-llm",
            (
                "transcribe",
                "--sample",
                "--config-file",
                str(transcribe_config),
                "--llm",
                "summarize action items",
                "--show-code",
            ),
        ),
        (
            "transcribe-youtube-download-sections",
            (
                "transcribe",
                "https://youtu.be/dtp6b76pMak",
                "--download-sections",
                "*0:00-5:00",
                "--show-code",
            ),
        ),
        ("stream-basic", ("stream", "--show-code")),
        (
            "stream-config-llm",
            (
                "stream",
                "--config-file",
                str(stream_config),
                "--llm",
                "summarize the finalized turns",
                "--show-code",
            ),
        ),
        (
            "agent-basic",
            ("agent", "--voice", "ivy", "--greeting", "Hello there", "--show-code"),
        ),
        (
            "agent-prompt-file",
            (
                "agent",
                "--voice",
                "james",
                "--system-prompt-file",
                str(prompt_file),
                "--show-code",
            ),
        ),
    )

    runner = CliRunner(mix_stderr=False)
    for name, args in cases:
        _write_fixture(runner, out_dir, name, args)

    sys.stdout.write(f"generated {len(cases)} show-code fixtures in {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
