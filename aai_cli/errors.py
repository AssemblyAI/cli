"""The CLIError hierarchy and the exit-code contract scripts can rely on.

Exit codes (stable; mirrored in the REFERENCE.md "Exit codes" table):

* ``0`` — success.
* ``1`` — a generic runtime failure: an API/network error (:class:`APIError`),
  a missing dependency, or an unexpected internal error.
* ``2`` — a usage/validation error (:class:`UsageError` and the validation
  ``CLIError``\\s): bad flags, a bad path, a malformed id, or an unusable config.
* ``4`` — not authenticated (:class:`NotAuthenticated`): no usable credential,
  a rejected key, or a self-service command that needs a browser login. The
  gh-style split keeps "you're not signed in" distinct from a usage error.
* ``130`` — cancelled with Ctrl-C.

A subprocess the CLI shells out to (``assembly deploy``/``dev``) propagates that
process's own exit code unchanged. Each :class:`CLIError` carries an
``error_type`` (the JSON ``error.type``) paired 1:1 with its ``exit_code``.
"""

from __future__ import annotations

from aai_cli import jsonshape


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
        # suggestion/transcript_id are omitted entirely when unset (not null).
        body = jsonshape.compact(
            {
                "type": self.error_type,
                "message": self.message,
                "suggestion": self.suggestion,
                "transcript_id": self.transcript_id,
            }
        )
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
            "Run 'assembly onboard' for guided setup, 'assembly login' if you have an account, "
            "or set ASSEMBLYAI_API_KEY."
        ),
        rejected_key: bool = False,
    ) -> None:
        super().__init__(
            message, error_type="not_authenticated", exit_code=4, suggestion=suggestion
        )
        # Structured marker for "a key was presented and the server rejected it" (vs
        # "no credential at all"), so callers like auto-login don't match message text.
        self.rejected_key = rejected_key


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


def mutually_exclusive(*flags: tuple[str, object], suggestion: str | None = None) -> None:
    """Raise a :class:`UsageError` naming the conflicting flags when more than one is set.

    The shared primitive behind every "these flags conflict" check (gh's
    ``MutuallyExclusive``), so each call site is one declaration instead of a bespoke
    validator. A flag counts as given when its value is truthy — call sites pass the
    raw option values (``None``/``False``/``[]`` all mean "not passed").
    """
    given = [name for name, value in flags if value]
    if not given[1:]:  # zero or one flag set: no conflict
        return
    *head, last = given
    joined = ", ".join(head)
    listed = f"{joined}, and {last}" if given[2:] else f"{joined} and {last}"
    raise UsageError(f"{listed} can't be combined.", suggestion=suggestion)


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
REJECTED_KEY_SUGGESTION = "Run 'assembly login' with a valid key, or set ASSEMBLYAI_API_KEY."
# The non-interactive sign-in recipe (stdin-only so the key never reaches shell
# history or `ps`); referenced by login's flags and the browser-flow fallbacks.
STDIN_KEY_RECIPE = "printenv ASSEMBLYAI_API_KEY | assembly login --with-api-key"
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
    return NotAuthenticated(
        REJECTED_KEY_MESSAGE, suggestion=REJECTED_KEY_SUGGESTION, rejected_key=True
    )
