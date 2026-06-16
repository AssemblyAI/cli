"""Run-logic shared by the two voice commands (`agent` and `agent-framework`).

Both build a live terminal conversation and resolve the persona the same way, so
the shared piece lives in the `app/` layer (the `doctor_checks`/`setup_exec`
precedent) rather than being copied between the two command packages.
"""

from __future__ import annotations

from pathlib import Path

from aai_cli.core.errors import CLIError


def resolve_system_prompt(system_prompt: str, system_prompt_file: Path | None) -> str:
    """The persona text: a --system-prompt-file (if given) overrides --system-prompt."""
    if system_prompt_file is None:
        return system_prompt
    try:
        return system_prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise CLIError(
            f"Could not read --system-prompt-file {system_prompt_file}: {exc}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Check the path and that the file is readable.",
        ) from exc
