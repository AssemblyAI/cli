from aai_cli.errors import APIError, CLIError, NotAuthenticated, is_auth_failure


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


def test_cli_error_defaults():
    # A bare CLIError must default to exit_code 1 and error_type "error" (nothing
    # else relies on these defaults, so pin them here).
    err = CLIError("nope")
    assert err.exit_code == 1
    assert err.error_type == "error"
    assert err.transcript_id is None
    assert err.suggestion is None


def test_to_dict_omits_none_transcript_id_and_suggestion():
    err = CLIError("nope", error_type="generic", exit_code=1)
    assert err.to_dict() == {"error": {"type": "generic", "message": "nope"}}


def test_api_error_carries_suggestion():
    err = APIError("boom", suggestion="retry")
    assert err.to_dict() == {
        "error": {"type": "api_error", "message": "boom", "suggestion": "retry"}
    }


def test_auth_failure_splits_message_and_suggestion():
    from aai_cli.errors import auth_failure

    err = auth_failure()
    assert err.error_type == "not_authenticated"
    assert err.message == "Your API key was rejected."
    assert "aai login" in (err.suggestion or "")
    assert "ASSEMBLYAI_API_KEY" in (err.suggestion or "")


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


def test_is_auth_failure_detects_structural_status_codes():
    # The structural signals folded in from the old agent-only _is_auth_rejection:
    # a WebSocket 1008 close (on .code or .rcvd.code) and an HTTP 401/403 response.
    class _Closed(Exception):
        code = 1008

    class _Frame:
        code = 1008

    class _ClosedRcvd(Exception):
        rcvd = _Frame()

    class _Resp403:
        status_code = 403

    class _Forbidden(Exception):
        response = _Resp403()

    class _Resp401:
        status_code = 401

    class _Unauthorized(Exception):
        response = _Resp401()

    assert is_auth_failure(_Closed("policy violation"))  # no auth word in the text
    assert is_auth_failure(_ClosedRcvd("connection closed"))
    # Both HTTP rejection codes must be matched structurally, with text that carries
    # no auth hint so only the status_code path can flag it (kills 401-vs-403 mutants).
    assert is_auth_failure(_Forbidden("nope"))
    assert is_auth_failure(_Unauthorized("nope"))


def test_is_auth_failure_ignores_unrelated_status_codes():
    class _Closed(Exception):
        code = 1000  # normal closure, not a policy rejection

    class _Resp:
        status_code = 500

    class _HTTPish(Exception):
        response = _Resp()

    assert not is_auth_failure(_Closed("bye"))
    assert not is_auth_failure(_HTTPish("server error"))
