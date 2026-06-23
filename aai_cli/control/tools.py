"""The control actions as OpenAI function-calling tool definitions.

The LLM Gateway is OpenAI-compatible, so each macOS action is exposed to the
model as a ``function`` tool; the model picks one and supplies JSON arguments,
which :func:`aai_cli.control.actions.validate` turns into an executable
:class:`~aai_cli.control.actions.Action`. The required-argument set comes
straight from :data:`actions.ACTION_SPECS` so the advertised tools and the
executable vocabulary cannot drift (the tests assert the two agree).
"""

from __future__ import annotations

from aai_cli.control import actions

# Human-readable, imperative one-liners the model sees for each tool.
_DESCRIPTIONS: dict[str, str] = {
    "type_text": "Type literal text at the current cursor/focus",
    "key_combo": "Press a key chord, e.g. ['cmd','s'] to save or ['cmd','tab'] to switch apps",
    "click": "Click an accessibility element by id (from get_ui_tree), or raw screen x/y",
    "launch_app": "Launch (or activate) an application by name, e.g. 'Safari'",
    "focus_app": "Bring an already-running application to the foreground by name",
    "get_ui_tree": "Read the focused app's accessibility tree: labeled, clickable elements",
    "screenshot": "Capture the current screen so you can see what is on it",
}

# JSON-schema property definitions per action. Required-ness is layered on from
# ACTION_SPECS in tool_definitions(), so this only describes the shape of each arg.
_PROPERTIES: dict[str, dict[str, dict[str, object]]] = {
    "type_text": {"text": {"type": "string", "description": "The exact text to type"}},
    "key_combo": {
        "keys": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Modifier/key names pressed together, lowercased",
        }
    },
    "click": {
        "element": {"type": "string", "description": "Accessibility element id from get_ui_tree"},
        "x": {"type": "integer", "description": "Screen x coordinate (use instead of element)"},
        "y": {"type": "integer", "description": "Screen y coordinate (use instead of element)"},
    },
    "launch_app": {"name": {"type": "string", "description": "Application name"}},
    "focus_app": {"name": {"type": "string", "description": "Application name"}},
    "get_ui_tree": {},
    "screenshot": {},
}


def tool_names() -> tuple[str, ...]:
    """The advertised tool names, sorted — must equal the executable action set."""
    return tuple(sorted(actions.ACTION_SPECS))


def _function_schema(name: str) -> dict[str, object]:
    """The ``function`` tool schema for one action, with its required args marked."""
    properties = _PROPERTIES[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(actions.ACTION_SPECS[name]),
                "additionalProperties": False,
            },
        },
    }


def tool_definitions() -> list[dict[str, object]]:
    """Every control action as an OpenAI ``tools`` entry, in stable (sorted) order."""
    return [_function_schema(name) for name in tool_names()]
