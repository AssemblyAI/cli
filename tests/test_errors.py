from assemblyai_cli.errors import APIError, CLIError, NotAuthenticated


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
