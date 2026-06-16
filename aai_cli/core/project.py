from __future__ import annotations

from aai_cli.core import jsonshape
from aai_cli.core.errors import UsageError

# Columns within one projected record are tab-separated, so a multi-field row
# (`-o id,status`) stays parseable with `cut -f` and pastes into a spreadsheet.
COLUMN_SEP = "\t"


def parse_fields(spec: str) -> list[str]:
    """Parse a comma-separated ``-o`` field spec into a list of field paths.

    Whitespace around each name is trimmed and empty segments dropped, so
    ``-o "id, status"`` and ``-o id,status`` are equivalent. An all-empty spec
    (e.g. ``-o ,``) is a usage error rather than a silently empty projection.
    """
    fields = [part.strip() for part in spec.split(",")]
    fields = [field for field in fields if field]
    if not fields:
        raise UsageError(
            "No fields given to -o.",
            suggestion="Pass one or more field names, e.g. -o id or -o id,status.",
        )
    return fields


def _lookup(record: dict[str, object], path: str) -> object:
    """Resolve a dotted ``path`` against a JSON object, or ``None`` if a step is missing.

    Dotted access (``a.b``) descends into nested objects, so a top-level field like
    ``session_id`` and a nested one like ``transform.model`` both work; a path that
    runs off a non-object (or names a missing key) yields ``None``, rendered as an
    empty column rather than raising.
    """
    value: object = record
    for key in path.split("."):
        mapping = jsonshape.as_mapping(value)
        if mapping is None or key not in mapping:
            return None
        value = mapping[key]
    return value


def render_value(value: object) -> str:
    """Render one projected value as a pipe-friendly scalar.

    ``None`` becomes an empty column, JSON booleans render lowercased
    (``true``/``false``) so they read like the ``--json`` payload, and a nested
    object/list is re-serialized as compact JSON rather than Python ``repr``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return jsonshape.dumps(value)


def project_record(record: dict[str, object], fields: list[str]) -> str:
    """One tab-separated line of the selected ``fields`` from a single JSON object."""
    return COLUMN_SEP.join(render_value(_lookup(record, field)) for field in fields)


def project_rows(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    """One projected line per object in ``rows`` (a list result)."""
    return [project_record(row, fields) for row in rows]


def project_any(data: object, fields: list[str]) -> list[str]:
    """Project ``fields`` from a JSON value, dispatching on its shape.

    A single object yields one line; a list yields one line per object (non-object
    items drop out); anything else (a bare scalar) yields no lines. Lets the output
    layer stay shape-agnostic — it just prints whatever lines come back.
    """
    mapping = jsonshape.as_mapping(data)
    if mapping is not None:
        return [project_record(mapping, fields)]
    return project_rows(jsonshape.mapping_list(data), fields)
