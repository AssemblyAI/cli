from __future__ import annotations


class CLIError(Exception):
    """Base error carrying an exit code and a machine-readable type."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "error",
        exit_code: int = 1,
        transcript_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.exit_code = exit_code
        self.transcript_id = transcript_id

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {"type": self.error_type, "message": self.message}
        if self.transcript_id is not None:
            body["transcript_id"] = self.transcript_id
        return {"error": body}


class NotAuthenticated(CLIError):
    def __init__(self, message: str = "Not authenticated. Run 'aai login'.") -> None:
        super().__init__(message, error_type="not_authenticated", exit_code=2)


class APIError(CLIError):
    def __init__(self, message: str, *, transcript_id: str | None = None) -> None:
        super().__init__(message, error_type="api_error", exit_code=1, transcript_id=transcript_id)


class UsageError(CLIError):
    def __init__(self, message: str) -> None:
        super().__init__(message, error_type="usage_error", exit_code=2)


# Substrings that mark a failure as "the credentials were rejected" rather than a
# generic network/protocol error. Matched case-insensitively against str(exc).
# Includes WebSocket close 1008 (policy violation), which is how the Voice Agent
# server signals a bad API key.
_AUTH_FAILURE_HINTS = (
    "unauthorized",
    "forbidden",
    "authentication",
    "api token",
    "invalid api key",
    "invalid key",
    "401",
    "403",
    "1008",
    "policy violation",
)

REJECTED_KEY_MESSAGE = (
    "Your API key was rejected. Run 'aai login' with a valid key (or set ASSEMBLYAI_API_KEY)."
)


def is_auth_failure(exc: object) -> bool:
    """Heuristic: does this exception/error indicate rejected/invalid credentials?"""
    text = str(exc).lower()
    return any(hint in text for hint in _AUTH_FAILURE_HINTS)


def auth_failure() -> NotAuthenticated:
    """A NotAuthenticated for the 'key present but rejected by the server' case."""
    return NotAuthenticated(REJECTED_KEY_MESSAGE)
