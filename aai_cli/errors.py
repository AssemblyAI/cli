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
    def __init__(
        self,
        message: str = "Not authenticated.",
        *,
        suggestion: str | None = "Run 'aai login'.",
    ) -> None:
        super().__init__(
            message, error_type="not_authenticated", exit_code=2, suggestion=suggestion
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
# (transcript ids, byte counts, ports). HTTP status codes and the Voice Agent's 1008
# close are detected structurally at the call site instead (see agent/session.py).
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


def is_auth_failure(exc: object) -> bool:
    """Heuristic: does this exception/error indicate rejected/invalid credentials?"""
    text = str(exc).lower()
    return any(hint in text for hint in _AUTH_FAILURE_HINTS)


def auth_failure() -> NotAuthenticated:
    """A NotAuthenticated for the 'key present but rejected by the server' case."""
    return NotAuthenticated(REJECTED_KEY_MESSAGE, suggestion=REJECTED_KEY_SUGGESTION)
