import os
import time

import keyring
import pytest
from keyring.backend import KeyringBackend

# Captured at import, before `isolate_env` strips ASSEMBLYAI_API_KEY from the
# environment. The e2e suite uses this real key to drive the CLI as a subprocess;
# unit tests still run fully isolated.
REAL_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")

# Suites that legitimately reach the real network in-process (PyPI reachability
# probes before a real install, real-API e2e) are gated behind these markers. They
# opt out of the suite-wide --disable-socket; every other test stays blocked, so an
# unmocked call in the unit suite still fails loudly. Tests that only bind a loopback
# server use the tighter `@pytest.mark.allow_hosts(["127.0.0.1"])` instead.
_NETWORK_MARKERS = ("e2e", "install")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if any(item.get_closest_marker(name) for name in _NETWORK_MARKERS):
            item.add_marker(pytest.mark.enable_socket)


@pytest.fixture
def real_api_key():
    """The real API key from the environment, or skip if none is set."""
    if not REAL_API_KEY:
        pytest.skip("ASSEMBLYAI_API_KEY not set; skipping real-API e2e test.")
    return REAL_API_KEY


class MemoryKeyring(KeyringBackend):
    # A plain value is the documented way to set a backend's priority; keyring types
    # the base attribute as a classproperty, so pyright needs the override flagged.
    priority = 1  # pyright: ignore[reportAssignmentType]

    def __init__(self):
        self._store = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self._store:
            import keyring.errors

            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    for var in (
        "ASSEMBLYAI_API_KEY",
        "ASSEMBLYAI_BASE_URL",
        "AAI_ENV",
        "CI",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "NO_COLOR",
        # With these cleared (and SHIPPED_CLIENT_TOKEN empty in source), telemetry
        # is inert in every test unless one opts in explicitly.
        "AAI_TELEMETRY_CLIENT_TOKEN",
        "AAI_TELEMETRY_INTAKE_URL",
        "AAI_TELEMETRY_DISABLED",
        "DO_NOT_TRACK",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def pin_timezone(monkeypatch):
    # Pin the host timezone so any time rendering is deterministic across machines and
    # CI, regardless of the contributor's local zone. A fixed *non-UTC* zone is
    # deliberate: the CLI normalizes everything to UTC (timeparse.format_utc_*), so
    # correct code is unaffected, while a future naive/local clock would visibly shift
    # here instead of only on a laptop that happens to sit in that zone. Tests that
    # need a frozen "now" use time-machine on top of this.
    monkeypatch.setenv("TZ", "America/New_York")
    if hasattr(time, "tzset"):
        time.tzset()


@pytest.fixture
def fixed_render_size(monkeypatch):
    # Pin the render width/height so the CLI-surface goldens (the
    # test_snapshots_* modules) are byte-identical across machines and CI.
    # Named (not autouse): only the snapshot modules opt in via
    # `pytestmark = pytest.mark.usefixtures("fixed_render_size")`.
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "40")


@pytest.fixture
def preserve_logging_state():
    # Logging is process-global: root handlers and level, plus per-logger levels.
    # A test that enables verbose diagnostics (debuglog.enable) or trips the
    # realtime silencers would otherwise leak that state into unrelated tests —
    # an order dependence pytest-randomly only exposes on some seeds (it cost a
    # red CI round on PR #125). Named (not autouse): modules that touch global
    # logging opt in. The websockets wire loggers are reset to NOTSET up front
    # so a CRITICAL clamp left by an earlier test can't swallow records the
    # opting test asserts on.
    import logging

    from aai_cli import ws as wsutil

    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    wire_loggers = [logging.getLogger(name) for name in wsutil.WEBSOCKETS_LOGGERS]
    previous_wire_levels = [logger.level for logger in wire_loggers]
    for logger in wire_loggers:
        logger.setLevel(logging.NOTSET)
    yield
    root.handlers[:] = previous_handlers
    root.setLevel(previous_level)
    for logger, level in zip(wire_loggers, previous_wire_levels, strict=True):
        logger.setLevel(level)


@pytest.fixture(autouse=True)
def reset_active_environment():
    # The active environment is a process-global (set at CLI startup); pin it to
    # the default before each test so unit tests aren't affected by ordering.
    from aai_cli import environments

    environments.set_active(environments.get(environments.DEFAULT_ENV))


@pytest.fixture(autouse=True)
def memory_keyring():
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    return backend


@pytest.fixture(autouse=True)
def neutralize_shipped_token(monkeypatch):
    # The shipped Datadog client token makes telemetry live by default. In-process
    # patches (pytest-socket, mocked boundaries) never reach the detached flusher
    # *subprocess* telemetry spawns, so blank the token suite-wide: tests exercise
    # telemetry by opting in via AAI_TELEMETRY_CLIENT_TOKEN and patching dispatch.
    # Returns the real shipped value so its own tests can still assert its shape.
    from aai_cli import telemetry

    original = telemetry.SHIPPED_CLIENT_TOKEN
    monkeypatch.setattr(telemetry, "SHIPPED_CLIENT_TOKEN", "")
    return original


@pytest.fixture
def memory_fs():
    """fsspec's in-process memory filesystem, reset afterwards.

    Lets remote-source tests (memory:// URLs) exercise real fsspec glob/find/
    download code paths while pytest-socket stays armed. The reset matters:
    MemoryFileSystem state is process-global (class attributes), so leftover
    files would leak across randomly-ordered tests.
    """
    import fsspec
    from fsspec.implementations.memory import MemoryFileSystem

    yield fsspec.filesystem("memory")
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs[:] = [""]


@pytest.fixture(autouse=True)
def tmp_config(monkeypatch, tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr("aai_cli.config.config_dir", lambda: cfg_dir)
    return cfg_dir
