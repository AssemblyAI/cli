from aai_cli.errors import APIError, CLIError, NotAuthenticated, UsageError, is_auth_failure


def test_not_authenticated_defaults():
    err = NotAuthenticated()
    assert err.exit_code == 2
    assert err.error_type == "not_authenticated"
    assert err.message == "Not authenticated."
    assert err.suggestion == "Run 'aai login'."


def test_api_error_carries_fields():
    err = APIError("boom", transcript_id="t_123")
    assert err.exit_code == 1
    assert err.error_type == "api_error"
    assert err.to_dict() == {
        "error": {"type": "api_error", "message": "boom", "transcript_id": "t_123"}
    }


def test_to_dict_includes_suggestion_when_present():
    err = CLIError("nope", error_type="generic", exit_code=1, suggestion="do this")
    assert err.to_dict() == {
        "error": {"type": "generic", "message": "nope", "suggestion": "do this"}
    }


def test_to_dict_omits_none_transcript_id_and_suggestion():
    err = CLIError("nope", error_type="generic", exit_code=1)
    assert err.to_dict() == {"error": {"type": "generic", "message": "nope"}}


def test_api_error_carries_suggestion():
    err = APIError("boom", suggestion="retry")
    assert err.to_dict() == {
        "error": {"type": "api_error", "message": "boom", "suggestion": "retry"}
    }


def test_is_auth_failure_matches_credential_signals():
    assert is_auth_failure(Exception("HTTP 401 Unauthorized"))
    assert is_auth_failure(Exception("Forbidden"))
    assert is_auth_failure(Exception("Authentication error, API token missing/invalid"))
    assert is_auth_failure(Exception("Invalid API key"))


def test_is_auth_failure_ignores_generic_errors():
    assert not is_auth_failure(RuntimeError("network unreachable"))
    assert not is_auth_failure(Exception("server exploded"))
    assert not is_auth_failure(ConnectionError("handshake refused"))


def test_is_auth_failure_ignores_bare_numeric_substrings():
    # Numbers like 401/403/1008 embedded in unrelated text must NOT be treated as auth.
    assert not is_auth_failure(Exception("decode failed at frame 1008"))
    assert not is_auth_failure(Exception("transcript abc401 not found"))
    assert not is_auth_failure(Exception("retry after 403 seconds"))
