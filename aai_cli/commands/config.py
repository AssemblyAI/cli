"""`assembly config` — inspect and edit the persisted CLI settings.

The settings live in ``config.toml`` (``assembly config path`` prints where); the
API key itself lives only in the OS keyring and is deliberately not reachable
from here. Runtime precedence for everything this file stores: command flags
(``--profile``/``--env``) > environment variables (``AAI_ENV``,
``ASSEMBLYAI_API_KEY``) > these stored settings > built-in defaults.
"""

from __future__ import annotations

import typer
from rich.markup import escape

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.core import config, environments
from aai_cli.core.choices import ConfigKey
from aai_cli.core.errors import UsageError
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.SETUP,
    order=22,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("config",),
    group_name="config",
)

app = typer.Typer(
    help="Inspect and edit persisted CLI settings (profiles, env, telemetry)",
    no_args_is_help=True,
)

_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})


def _parse_bool(key: ConfigKey, raw: str) -> bool:
    word = raw.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise UsageError(
        f"{key} expects a boolean, got {raw!r}.",
        suggestion=f"Use one of: {', '.join(sorted(_TRUE_WORDS | _FALSE_WORDS))}.",
    )


def _validated_env(value: str) -> str:
    name = value.strip()
    if name not in environments.ENVIRONMENTS:
        raise UsageError(
            f"Unknown environment {value!r}.",
            suggestion=f"Use one of: {', '.join(environments.ENVIRONMENTS)}.",
        )
    return name


def _current_value(key: ConfigKey, state: AppState) -> object:
    if key is ConfigKey.active_profile:
        return config.get_active_profile()
    if key is ConfigKey.env:
        return config.get_profile_env(state.resolve_profile())
    return config.get_telemetry_enabled()


def _store_value(key: ConfigKey, raw: str, state: AppState) -> object:
    """Persist ``raw`` under ``key`` and return the typed value that was stored."""
    if key is ConfigKey.active_profile:
        config.set_active_profile(raw)
        return raw
    if key is ConfigKey.env:
        env = _validated_env(raw)
        config.set_profile_env(state.resolve_profile(), env)
        return env
    enabled = _parse_bool(key, raw)
    config.set_telemetry_enabled(enabled=enabled)
    return enabled


def _render_value(value: object) -> str:
    """One stable spelling per value for the pipe-friendly `get` output: booleans in
    TOML/JSON case (``true``/``false``), an unset value as ``unset``."""
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@app.command(
    epilog=examples_epilog(
        [
            ("Where settings are stored", "assembly config path"),
        ]
    )
)
def path(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Print where config.toml lives"""

    def body(_state: AppState, json_mode: bool) -> None:
        file = config.config_file_path()
        if json_mode:
            output.emit({"path": str(file)}, str, json_mode=True)
        else:
            # Raw print, not the Rich console: a long path must reach a pipe
            # unwrapped (`cd "$(assembly config path | xargs dirname)"`).
            output.emit_text(str(file))

    # The location is independent of the file's contents, so report it even when the
    # config is unreadable — this is the command you'd use to go fix the broken file.
    run_command(ctx, body, json=json_out, tolerate_unreadable_config=True)


@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("Show every persisted setting", "assembly config list"),
            ("As JSON for scripting", "assembly config list --json"),
        ]
    ),
)
def list_settings(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Show every persisted setting and the stored profiles"""

    def body(_state: AppState, json_mode: bool) -> None:
        data: dict[str, object] = {
            "path": str(config.config_file_path()),
            "active_profile": config.get_active_profile(),
            "profiles": config.list_profiles(),
            "telemetry_enabled": config.get_telemetry_enabled(),
        }

        def render(d: dict[str, object]) -> object:
            table = output.detail_table()
            table.add_row("Config file", escape(str(d["path"])))
            table.add_row("Active profile", escape(str(d["active_profile"])))
            profiles = config.list_profiles()
            listed = (
                ", ".join(
                    f"{name} ({env})" if env else name for name, env in sorted(profiles.items())
                )
                or "none yet"
            )
            table.add_row("Profiles", escape(listed))
            table.add_row("Telemetry", _render_value(d["telemetry_enabled"]))
            return output.stack(
                table,
                output.hint("Change a value with `assembly config set <key> <value>`."),
            )

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Read one setting (pipe-friendly)", "assembly config get env"),
            ("Read a named profile's env", "assembly -p staging config get env"),
        ]
    )
)
def get(
    ctx: typer.Context,
    key: ConfigKey = typer.Argument(..., help="Which setting to read"),
    json_out: bool = options.json_option(),
) -> None:
    """Print one setting's stored value (`env` reads the selected profile's)"""

    def body(state: AppState, json_mode: bool) -> None:
        value = _current_value(key, state)
        if json_mode:
            output.emit({"key": str(key), "value": value}, str, json_mode=True)
        else:
            # Raw print (see `path`): the bare value is the pipe contract here.
            output.emit_text(_render_value(value))

    run_command(ctx, body, json=json_out)


@app.command(
    name="set",
    epilog=examples_epilog(
        [
            ("Switch the default profile", "assembly config set active_profile staging"),
            ("Bind the active profile to the sandbox", "assembly config set env sandbox000"),
            ("Opt out of telemetry", "assembly config set telemetry_enabled false"),
        ]
    ),
)
def set_setting(
    ctx: typer.Context,
    key: ConfigKey = typer.Argument(..., help="Which setting to change"),
    value: str = typer.Argument(..., help="The new value"),
    json_out: bool = options.json_option(),
) -> None:
    """Change one setting (`env` writes to the selected profile)"""

    def body(state: AppState, json_mode: bool) -> None:
        stored = _store_value(key, value, state)
        output.emit(
            {"key": str(key), "value": stored},
            lambda d: output.success(f"{d['key']} = {escape(_render_value(d['value']))}"),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
