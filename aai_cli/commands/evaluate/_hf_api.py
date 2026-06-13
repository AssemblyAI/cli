"""The Hugging Face datasets-server REST client behind `assembly eval`.

Thin httpx wrappers (split discovery, subset/split selection, row fetching)
with the error translation `eval_data` relies on: auth/gating denials get an
``HF_TOKEN`` hint only when the response body actually reads like gating, and
everything else surfaces the server's own detail verbatim.
"""

from __future__ import annotations

from http import HTTPStatus

import httpx2 as httpx

from aai_cli.core import env, jsonshape
from aai_cli.core.errors import APIError, UsageError

_DATASETS_SERVER = "https://datasets-server.huggingface.co"
_TIMEOUT = 30.0  # pragma: no mutate (request timeout; nothing observable to assert)


def _error_detail(resp: httpx.Response) -> str:
    try:
        body: object = resp.json()
    except ValueError:
        return resp.text
    mapping = jsonshape.as_mapping(body)
    if mapping is not None and "error" in mapping:
        return str(mapping["error"])
    return resp.text


# A 401/403 body that mentions one of these reads like HF auth/gating, where a token
# can actually help; anything else (e.g. a sandbox proxy's "Host not in allowlist")
# gets the body verbatim instead of a misleading HF_TOKEN hint.
_GATING_HINTS = ("gated", "private", "auth", "token")


def _looks_gating_related(detail: str) -> bool:
    lowered = detail.lower()
    return not detail or any(hint in lowered for hint in _GATING_HINTS)


def _denied_access_error(resp: httpx.Response, *, dataset: str) -> APIError:
    detail = _error_detail(resp)
    message = f"Hugging Face denied access to '{dataset}' (HTTP {resp.status_code})"
    if detail:
        message += f": {detail}"
    return APIError(
        message,
        suggestion=(
            "Gated or private dataset? Set HF_TOKEN to a token that has access."
            if _looks_gating_related(detail)
            else None
        ),
    )


def _checked_payload(resp: httpx.Response, *, dataset: str) -> dict[str, object]:
    if resp.status_code in (401, 403):
        raise _denied_access_error(resp, dataset=dataset)
    if resp.status_code == HTTPStatus.NOT_FOUND:
        raise UsageError(
            f"Hugging Face dataset '{dataset}' was not found: {_error_detail(resp)}",
            suggestion="Check the dataset id, e.g. 'distil-whisper/meanwhile'.",
        )
    if resp.status_code != HTTPStatus.OK:
        raise APIError(
            f"Hugging Face datasets server error (HTTP {resp.status_code}): {_error_detail(resp)}"
        )
    try:
        data: object = resp.json()
    except ValueError as exc:
        raise APIError("Hugging Face datasets server returned invalid JSON.") from exc
    mapping = jsonshape.as_mapping(data)
    if mapping is None:
        raise APIError(
            "Hugging Face datasets server returned unexpected JSON (expected an object)."
        )
    return mapping


def fetch_json(endpoint: str, params: dict[str, str | int], *, dataset: str) -> dict[str, object]:
    token = env.get("HF_TOKEN")
    headers = {"authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=_DATASETS_SERVER, timeout=_TIMEOUT, headers=headers) as client:
            resp = client.get(endpoint, params=params)
    except httpx.HTTPError as exc:
        raise APIError(f"Could not reach the Hugging Face datasets server: {exc}") from exc
    return _checked_payload(resp, dataset=dataset)


def split_entries(dataset: str) -> list[dict[str, object]]:
    payload = fetch_json("/splits", {"dataset": dataset}, dataset=dataset)
    entries = jsonshape.mapping_list(payload.get("splits"))
    if not entries:
        raise APIError(f"Hugging Face reports no splits for '{dataset}'.")
    return entries


def pick_subset(entries: list[dict[str, object]], subset: str | None, dataset: str) -> str:
    configs = list(dict.fromkeys(str(entry.get("config")) for entry in entries))
    if subset is not None:
        if subset in configs:
            return subset
        raise UsageError(f"'{dataset}' has no subset '{subset}' (subsets: {', '.join(configs)}).")
    if len(configs) == 1:
        return configs[0]
    if "default" in configs:
        return "default"
    raise UsageError(
        f"'{dataset}' has multiple subsets: {', '.join(configs)}.",
        suggestion="Pick one with --subset.",
    )


def pick_split(
    entries: list[dict[str, object]], config: str, split: str | None, dataset: str
) -> str:
    splits = [str(entry.get("split")) for entry in entries if str(entry.get("config")) == config]
    if split is not None:
        if split in splits:
            return split
        raise UsageError(
            f"'{dataset}' has no '{split}' split in subset '{config}' "
            f"(splits: {', '.join(splits)})."
        )
    if "test" in splits:
        return "test"
    if len(splits) == 1:
        return splits[0]
    raise UsageError(
        f"'{dataset}' has several splits in subset '{config}': {', '.join(splits)}.",
        suggestion="Pick one with --split (eval sets usually score 'test').",
    )
