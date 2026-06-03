from __future__ import annotations

from assemblyai.streaming.v3 import SpeechModel


def py_literal(value: object) -> str:
    """Render a coerced config value as Python source.

    Handles SDK enums (SpeechModel.<name>) and plain JSON-ish types. repr() yields
    valid Python for str/bool/int/float/list/dict with string keys.
    """
    if isinstance(value, SpeechModel):
        return f"SpeechModel.{value.name}"
    return repr(value)


def config_kwarg_lines(merged: dict[str, object], indent: int) -> list[str]:
    """Render a merged kwargs dict as indented `field=value,` source lines."""
    pad = " " * indent
    return [f"{pad}{key}={py_literal(val)}," for key, val in merged.items()]
