from assemblyai_cli.errors import APIError, CLIError, NotAuthenticated, is_auth_failure


def test_not_authenticated_defaults():
    err = NotAuthenticated()
    assert err.exit_code == 2
    assert err.error_type == "not_authenticated"
    assert "aai login" in str(err)


def test_api_error_carries_fields():
    err = APIError("boom", transcript_id="t_123")
    assert err.exit_code == 1
    assert err.error_type == "api_error"
    assert err.to_dict() == {
        "error": {"type": "api_error", "message": "boom", "transcript_id": "t_123"}
    }


def test_to_dict_omits_none_transcript_id():
    err = CLIError("nope", error_type="generic", exit_code=1)
    assert err.to_dict() == {"error": {"type": "generic", "message": "nope"}}


def test_is_auth_failure_matches_credential_signals():
    assert is_auth_failure(RuntimeError("received 1008 (policy violation)"))
    assert is_auth_failure(Exception("HTTP 401 Unauthorized"))
    assert is_auth_failure(Exception("Authentication error, API token missing/invalid"))


def test_is_auth_failure_ignores_generic_errors():
    assert not is_auth_failure(RuntimeError("network unreachable"))
    assert not is_auth_failure(Exception("server exploded"))
    assert not is_auth_failure(ConnectionError("handshake refused"))
