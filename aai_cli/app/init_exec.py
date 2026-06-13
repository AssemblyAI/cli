"""Run logic for `assembly init`: scaffold (and optionally launch) a starter app.

The command module (aai_cli/commands/init.py) only parses argv — it builds an
``InitOptions`` and hands it to ``run_init`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive scaffolding, install, and
launch by constructing options directly instead of round-tripping argv.

``run_init`` and ``launch_app`` are public because the onboarding wizard
(aai_cli/onboard/sections.py) scaffolds with ``launch=False`` and starts the
dev server itself once its remaining sections have run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import __version__
from aai_cli.app.context import AppState
from aai_cli.core import environments, stdio
from aai_cli.core.errors import CLIError, UsageError
from aai_cli.init import keys, runner, scaffold, templates
from aai_cli.ui import output, steps

DEFAULT_PORT = 3000


@dataclass(frozen=True)
class InitOptions:
    """Every `assembly init` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    template: str | None
    directory: str | None
    no_install: bool
    no_open: bool
    force: bool
    here: bool
    port: int


def _pick_template() -> str:
    """Interactive picker; raises a usage error when there's no TTY to prompt on."""
    if not stdio.interactive_stdio():
        raise CLIError(
            "No template given and not running interactively. "
            f"Pass one of: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="usage_error",
            exit_code=1,
        )
    try:
        import questionary
    except ImportError as exc:  # a broken/stale install missing the declared dep
        raise CLIError(
            "The interactive picker needs 'questionary'. Reinstall the CLI "
            "(e.g. `uv tool install --reinstall aai-cli`), or pass a template "
            f"directly: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="missing_dependency",
            exit_code=1,
        ) from exc

    choice = questionary.select(
        "Pick a template",
        choices=[
            questionary.Choice(title=templates.title_for(t), value=t)
            for t in templates.TEMPLATE_ORDER
        ],
    ).ask()
    if choice is None:  # user pressed Ctrl-C
        raise typer.Exit(code=130)
    return str(choice)


def _resolve_dir(directory: str | None, template: str, *, here: bool) -> Path:
    if here:
        return Path.cwd()
    if directory:
        return Path(directory)
    return Path.cwd() / template


def _resolve_template(template: str | None) -> str:
    """Resolve the template name: the picker when omitted, else validate the arg."""
    chosen = template if template is not None else _pick_template()
    if not templates.is_template(chosen):
        raise CLIError(
            f"Unknown template {chosen!r}. Choose one of: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="usage_error",
            exit_code=1,
        )
    return chosen


def _active_env_vars() -> dict[str, str]:
    """Pin the scaffolded app to the active environment's hosts.

    A sandbox key (minted by `assembly login` against a non-prod env) would otherwise be
    rejected by the production defaults baked into the template.
    """
    env = environments.active()
    return {
        "ASSEMBLYAI_BASE_URL": env.api_base,
        "ASSEMBLYAI_LLM_GATEWAY_URL": env.llm_gateway_base,
        "ASSEMBLYAI_STREAMING_HOST": env.streaming_host,
        # Voice Agent host mirrors the streaming host's naming across environments.
        "ASSEMBLYAI_AGENTS_HOST": env.streaming_host.replace("streaming", "agents", 1),
    }


def _install_step(
    target: Path, *, no_install: bool, api_key: str | None, use_uv: bool
) -> tuple[list[steps.Step], bool]:
    """Run (or skip) dependency install, returning the report rows and whether to launch.

    Launch only happens when deps are installed and there's a key; an install failure
    flips `will_launch` off so the caller exits non-zero instead of starting a server.
    """
    will_launch = not no_install and api_key is not None
    if no_install:
        return [{"name": "install", "status": "skipped", "detail": "--no-install"}], will_launch
    setup = runner.run_setup(target, use_uv=use_uv)
    if setup.returncode != 0:
        row: steps.Step = {
            "name": "install",
            "status": "failed",
            "detail": (setup.stderr or setup.stdout).strip()[:300],
        }
        # The False (don't-launch) is an equivalent mutant: run_init raises Exit(1) on
        # any failed step before it ever consults will_launch, so the value is unused
        # on this branch.
        return [row], False  # pragma: no mutate
    return [
        {
            "name": "install",
            "status": "installed",
            "detail": "uv" if use_uv else "venv + pip",
        }
    ], will_launch


def _resolve_target(
    directory: str | None, chosen: str, *, here: bool, force: bool
) -> tuple[Path, bool]:
    """Resolve the target directory, rejecting --here+DIRECTORY, an existing file, or
    a non-empty conflict. Returns the target and whether --force is overlaying it."""
    if here and directory:
        raise CLIError(
            "Pass either a DIRECTORY or --here, not both.",
            error_type="usage_error",
            exit_code=1,
        )
    target = _resolve_dir(directory, chosen, here=here)
    if target.exists() and not target.is_dir():
        raise UsageError(f"{target} exists and is not a directory.")
    conflict = scaffold.target_conflict(target)
    if conflict and not force:
        raise CLIError(
            f"{target} already exists and is not empty. "
            f"Use --force to overwrite or pick another directory.",
            error_type="usage_error",
            exit_code=1,
        )
    return target, conflict


def _key_row(api_key: str | None, key_source: str | None, preserved: str | None) -> steps.Step:
    """The report's `key` row — emitted symmetrically whether a key resolved or not."""
    if api_key is not None:
        # Literal branches rather than interpolating key_source: it rode in the same
        # return tuple as the API key, so CodeQL's coarse tuple taint marks it
        # sensitive and flags the report emit (py/clear-text-logging-sensitive-data).
        detail = "from environment" if key_source == "environment" else "from keyring"
        return {"name": "key", "status": "written", "detail": detail}
    if preserved is not None:
        return {"name": "key", "status": "kept", "detail": "existing .env key preserved"}
    return {
        "name": "key",
        "status": "skipped",
        "detail": "no API key found; wrote a placeholder to .env (run `assembly login`)",
    }


def _scaffold_report(
    chosen: str,
    target: Path,
    *,
    api_key: str | None,
    key_source: str | None,
    preserved: str | None,
) -> list[steps.Step]:
    """Write the template to `target` and return the opening report rows."""
    scaffold.scaffold(chosen, target, api_key=api_key or preserved, env_vars=_active_env_vars())
    return [
        {"name": "scaffold", "status": "created", "detail": str(target)},
        _key_row(api_key, key_source, preserved),
    ]


def _dev_hint(port: int) -> str:
    """The `assembly dev` invocation matching the chosen port (the default needs no flag)."""
    return "assembly dev" if port == DEFAULT_PORT else f"assembly dev --port {port}"


def launch_app(target: Path, *, port: int, use_uv: bool, no_open: bool, json_mode: bool) -> None:
    """Start the scaffolded app on a free port and open the browser, then block.

    Public (not underscore-private) because the onboarding wizard launches the
    scaffolded app as its final step, after the remaining wizard sections have run.
    """
    chosen_port = runner.find_free_port(port)
    url = f"http://localhost:{chosen_port}"
    if not json_mode:
        output.console.print(
            f"[aai.heading]Starting[/aai.heading] [aai.url]{escape(url)}[/aai.url]"
            "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
        )
    code = runner.launch_and_open(target, port=chosen_port, use_uv=use_uv, open_browser=not no_open)
    if code:
        raise typer.Exit(code=code)


def _build_report(
    state: AppState, chosen: str, target: Path, *, no_install: bool, use_uv: bool, port: int
) -> tuple[list[steps.Step], bool]:
    """Scaffold and assemble the report rows; returns them plus whether to launch."""
    api_key, key_source = keys.resolve_optional_api_key(profile=state.profile)
    # A configured (non-placeholder) .env key must survive a re-scaffold when no key
    # resolves — otherwise --force would silently reset it to the placeholder.
    preserved = scaffold.existing_env_key(target) if api_key is None else None
    effective_key = api_key or preserved
    report = _scaffold_report(
        chosen, target, api_key=api_key, key_source=key_source, preserved=preserved
    )

    install_rows, will_launch = _install_step(
        target, no_install=no_install, api_key=effective_key, use_uv=use_uv
    )
    report.extend(install_rows)

    # Deps are installed but there's no key, so the server can't start — say so
    # rather than exiting silently.
    if not no_install and effective_key is None:
        report.append(
            {
                "name": "launch",
                "status": "skipped",
                "detail": f"no API key; run `assembly login`, then: cd {target} && {_dev_hint(port)}",
            }
        )
    return report, will_launch


def run_init(
    opts: InitOptions,
    state: AppState,
    *,
    json_mode: bool,
    launch: bool = True,
) -> Path:
    """Scaffold (and optionally install/launch) a template; return the target dir.

    `launch=False` is for callers like the onboarding wizard that must not block on a
    running dev server mid-flow — it stops after install and leaves the run command as
    a hint (the wizard calls `launch_app` itself once its remaining sections are done).
    """
    chosen = _resolve_template(opts.template)
    target, overwriting = _resolve_target(opts.directory, chosen, here=opts.here, force=opts.force)
    if not json_mode:
        # Vercel-style banner, printed only once validation passes so pure error runs
        # (unknown template, conflicting target) stay undecorated like the sibling
        # commands. Decoration goes to stderr (data → stdout): it must never pollute
        # a piped stdout.
        output.error_console.print(
            f"[aai.heading]AssemblyAI CLI[/aai.heading] [aai.muted]{__version__}[/aai.muted]"
        )
    if overwriting:
        output.emit_warning(
            f"--force: overwriting existing files in {target} "
            "(the template is overlaid; files not in the template are kept).",
            json_mode=json_mode,
        )

    use_uv = runner.has_uv()
    report, will_launch = _build_report(
        state, chosen, target, no_install=opts.no_install, use_uv=use_uv, port=opts.port
    )

    output.emit(report, lambda d: steps.render_steps(d, heading="Setup"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    if launch and will_launch:
        launch_app(target, port=opts.port, use_uv=use_uv, no_open=opts.no_open, json_mode=json_mode)
    elif not json_mode:
        # Scaffolded but not launched (no key, or --no-install, or launch=False): leave the
        # user with the one command that starts their app, the way `vercel`/`supabase` sign off.
        output.console.print(
            output.hint(f"Run `cd {escape(str(target))} && {_dev_hint(opts.port)}`.")
        )
    return target
