"""The CLIError hierarchy and the exit-code contract scripts can rely on.

Exit codes (stable; mirrored in the README "Exit codes" table):

* ``0`` — success.
* ``1`` — a generic runtime failure: an API/network error (:class:`APIError`),
  a missing dependency, or an unexpected internal error.
* ``2`` — a usage/validation error (:class:`UsageError` and the validation
  ``CLIError``\\s): bad flags, a bad path, a malformed id, or an unusable config.
* ``4`` — not authenticated (:class:`NotAuthenticated`): no usable credential,
  a rejected key, or a self-service command that needs a browser login. The
  gh-style split keeps "you're not signed in" distinct from a usage error.
* ``130`` — cancelled with Ctrl-C.

A subprocess the CLI shells out to (``aai deploy``/``dev``) propagates that
process's own exit code unchanged. Each :class:`CLIError` carries an
``error_type`` (the JSON ``error.type``) paired 1:1 with its ``exit_code``.
"""

from __future__ import annotations


class CLIError(Exception):
    """Base error carrying an exit code, a machine-readable type, and an optional
    human suggestion for how to fix it."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "error",
        exit_code: int = 1,
        transcript_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.exit_code = exit_code
        self.transcript_id = transcript_id
        self.suggestion = suggestion

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {"type": self.error_type, "message": self.message}
        if self.suggestion is not None:
            body["suggestion"] = self.suggestion
        if self.transcript_id is not None:
            body["transcript_id"] = self.transcript_id
        return {"error": body}


class NotAuthenticated(CLIError):
    # Exit code 4 (not 2) so scripts can tell "you're not signed in" apart from a
    # usage error: UsageError keeps the conventional 2, auth gets its own code, the
    # same split gh uses.
    def __init__(
        self,
        message: str = "You're not signed in.",
        *,
        suggestion: str | None = (
            "Run 'aai onboard' for guided setup, 'aai login' if you have an account, "
            "or set ASSEMBLYAI_API_KEY."
        ),
    ) -> None:
        super().__init__(
            message, error_type="not_authenticated", exit_code=4, suggestion=suggestion
        )


class APIError(CLIError):
    def __init__(
        self,
        message: str,
        *,
        transcript_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_type="api_error",
            exit_code=1,
            transcript_id=transcript_id,
            suggestion=suggestion,
        )


class UsageError(CLIError):
    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, error_type="usage_error", exit_code=2, suggestion=suggestion)


# Word-level phrases that mark a failure as "the credentials were rejected" rather
# than a generic network/protocol error. Matched case-insensitively against str(exc).
# Deliberately NOT bare numbers like "401"/"403"/"1008": those match unrelated text
# (transcript ids, byte counts, ports). Numeric status/close codes are matched
# structurally on exception attributes by `_has_auth_status_code` instead.
_AUTH_FAILURE_HINTS = (
    "unauthorized",
    "forbidden",
    "authentication",
    "api token",
    "invalid api key",
    "invalid key",
)

REJECTED_KEY_MESSAGE = "Your API key was rejected."
REJECTED_KEY_SUGGESTION = "Run 'aai login' with a valid key, or set ASSEMBLYAI_API_KEY."
_VOICE_AGENT_POLICY_VIOLATION_CODE = 1008


def _has_auth_status_code(exc: object) -> bool:
    """True when an exception carries a numeric signal that credentials were rejected.

    Reads exception attributes, never the message text, so a transcript id or byte
    count that happens to contain "401"/"1008" can't trip it. Covers the two structured
    cases: the Voice Agent closes a bad-key WebSocket with code 1008 (on ``.code``, or
    on the received close frame ``.rcvd.code``), and a pre-upgrade HTTP rejection
    carries 401/403 on ``.response.status_code``.
    """
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(getattr(exc, "rcvd", None), "code", None)
    if code == _VOICE_AGENT_POLICY_VIOLATION_CODE:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in (401, 403)


def is_auth_failure(exc: object) -> bool:
    """Does this exception/error indicate rejected/invalid credentials?

    Checks structured numeric signals first (WebSocket 1008, HTTP 401/403), then falls
    back to a case-insensitive text heuristic. Shared by every backend path — REST
    (client.py), v3 streaming, and the Voice Agent — so a rejected key yields the same
    exit code no matter which transport surfaced the failure.
    """
    if _has_auth_status_code(exc):
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _AUTH_FAILURE_HINTS)


def auth_failure() -> NotAuthenticated:
    """A NotAuthenticated for the 'key present but rejected by the server' case."""
    return NotAuthenticated(REJECTED_KEY_MESSAGE, suggestion=REJECTED_KEY_SUGGESTION)
