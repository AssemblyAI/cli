from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

_JSON_OBJECT: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_OBJECT_LIST: TypeAdapter[list[object]] = TypeAdapter(list[object])


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


def mapping_list(value: object) -> list[dict[str, object]]:
    valid: list[dict[str, object]] = []
    for item in object_list(value):
        mapped = as_mapping(item)
        if mapped is not None:
            valid.append(mapped)
    return valid
