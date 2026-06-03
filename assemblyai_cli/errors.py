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
