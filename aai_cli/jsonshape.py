from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

_JSON_OBJECT: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_OBJECT_LIST: TypeAdapter[list[object]] = TypeAdapter(list[object])
_OBJECT_DICT_LIST: TypeAdapter[list[dict[str, object]]] = TypeAdapter(list[dict[str, object]])
_INT: TypeAdapter[int] = TypeAdapter(int)
_FLOAT: TypeAdapter[float] = TypeAdapter(float)


def as_mapping(value: object) -> dict[str, object] | None:
    """``value`` as a ``dict[str, object]`` if it is a JSON object, else None."""
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def object_list(value: object) -> list[object]:
    """``value`` as a list if it is one, else ``[]`` — for iterating an untyped payload."""
    try:
        return _OBJECT_LIST.validate_python(value)
    except ValidationError:
        return []


def as_object_list(value: object) -> list[dict[str, object]] | None:
    """Validate ``value`` as a list of JSON objects, or None if it isn't one.

    Unlike ``mapping_list`` (which silently drops non-object items), this rejects
    the whole value when any element isn't an object — for callers that must tell a
    wrong-shaped response apart from an empty list.
    """
    try:
        return _OBJECT_DICT_LIST.validate_python(value)
    except ValidationError:
        return None


def mapping_list(value: object) -> list[dict[str, object]]:
    """The object items of ``value`` as dicts, silently dropping any non-object element."""
    return [mapped for item in object_list(value) if (mapped := as_mapping(item)) is not None]


def as_int(value: object, default: int = 0) -> int:
    """Coerce an untyped JSON scalar to int, returning ``default`` on failure.

    ``bool`` is treated as non-numeric (a JSON ``true``/``false`` is not a count),
    overriding pydantic's lax ``True`` -> ``1`` coercion.
    """
    if isinstance(value, bool):
        return default
    try:
        return _INT.validate_python(value)
    except ValidationError:
        return default


def as_float(value: object, default: float = 0.0) -> float:
    """Coerce an untyped JSON scalar to float, returning ``default`` on failure.

    ``bool`` is treated as non-numeric (a JSON ``true``/``false`` is not a count),
    overriding pydantic's lax ``True`` -> ``1.0`` coercion.
    """
    if isinstance(value, bool):
        return default
    try:
        return _FLOAT.validate_python(value)
    except ValidationError:
        return default


def dumps(obj: object) -> str:
    """Serialize ``obj`` to a JSON string the way the whole CLI does it.

    ``default=str`` is the one safety the CLI relies on everywhere it emits JSON:
    pydantic/SDK models and ``datetime``\\s that aren't natively serializable fall
    back to ``str(...)`` instead of raising. Centralized here so every emission
    path (``output``'s stdout/stderr writers, the realtime ``BaseRenderer``, the
    ``--out`` and ``-o json`` field renderers) shares one serialization policy.
    """
    return json.dumps(obj, default=str)


def compact(mapping: dict[str, object]) -> dict[str, object]:
    """Return ``mapping`` without the keys whose value is ``None``.

    For JSON payloads where an absent optional field should be omitted entirely
    rather than serialized as ``null`` — the build-then-``if x is not None``
    idiom repeated across the error and realtime-event payloads.
    """
    return {key: value for key, value in mapping.items() if value is not None}
