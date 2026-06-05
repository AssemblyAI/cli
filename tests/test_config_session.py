from aai_cli import config


def test_set_and_get_session_roundtrips():
    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    assert config.get_session("default") == {"jwt": "jwt_1", "token": "tok_1"}
    assert config.get_account_id("default") == 42


def test_get_session_none_when_absent():
    assert config.get_session("default") is None
    assert config.get_account_id("default") is None


def test_clear_session_removes_jwt_and_account_id():
    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    config.clear_session("default")
    assert config.get_session("default") is None
    assert config.get_account_id("default") is None


def test_clear_session_is_safe_when_absent():
    config.clear_session("never-logged-in")  # must not raise


def test_get_session_returns_none_for_corrupt_blob():
    import keyring

    keyring.set_password(config.KEYRING_SERVICE, "session:default", "not-json")
    assert config.get_session("default") is None
