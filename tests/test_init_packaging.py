import importlib


def test_init_package_imports():
    mod = importlib.import_module("aai_cli.init")
    assert mod is not None


def test_questionary_is_available():
    # questionary is a runtime dependency of the CLI (the init picker).
    assert importlib.import_module("questionary") is not None
