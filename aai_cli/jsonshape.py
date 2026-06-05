from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

_JSON_OBJECT: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_OBJECT_LIST: TypeAdapter[list[object]] = TypeAdapter(list[object])
_OBJECT_DICT_LIST: TypeAdapter[list[dict[str, object]]] = TypeAdapter(list[dict[str, object]])
_INT: TypeAdapter[int] = TypeAdapter(int)
_FLOAT: TypeAdapter[float] = TypeAdapter(float)


def as_mapping(value: object) -> dict[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def object_list(value: object) -> list[object]:
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
    valid: list[dict[str, object]] = []
    for item in object_list(value):
        mapped = as_mapping(item)
        if mapped is not None:
            valid.append(mapped)
    return valid


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
