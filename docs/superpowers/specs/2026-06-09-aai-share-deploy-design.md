# `aai share` and `aai deploy` — design

**Date:** 2026-06-09
**Status:** Approved (design)

Two follow-on commands to `aai dev`, both filed under the "Build an App" help panel.

## `aai share` — public tunnel over the local dev server

**Goal:** boot the app (the same way `aai dev` does) and expose it on a public
`https://*.trycloudflare.com` URL via a cloudflared quick tunnel, so the running
app can be shared instantly (demos, mobile testing, webhooks).

### Decisions

- **Quick tunnel**, not a named tunnel: `cloudflared tunnel --url http://localhost:PORT`.
  Zero-config, no Cloudflare account, prints an ephemeral `*.trycloudflare.com` URL.
- **cloudflared is required**; resolved via `shutil.which("cloudflared")`. Missing →
  `CLIError(error_type="missing_dependency")` with a `brew install cloudflared` hint.
  Added as `depends_on "cloudflared"` in `Formula/aai.rb` so Homebrew installs ship it.
- Reuses the `aai dev` boot path (Procfile `web:` + `--reload`), so "share" really is
  "dev, but also tunneled."
- Prints the public URL prominently; does **not** auto-open a browser (you're sharing
  the link with someone else). Same `--port` / `--no-install` flags as `dev`.

### Architecture

**New `aai_cli/init/devserver.py`** (shared by `dev` and `share`; lives in the `init`
layer so both command modules can import it without breaking the "commands are
independent" import contract):

- `install_step(target, *, no_install, use_uv) -> steps.Step` — moved from `dev.py`.
- `dev_command(target, web, *, use_uv) -> list[str]` — moved from `dev.py` (`uv run …`
  or `venv -m …`, with `--reload` appended).

`aai_cli/commands/dev.py` is refactored to import these (behavior unchanged).

**New `aai_cli/init/tunnel.py`:**

- `CLOUDFLARED = "cloudflared"`; `tunnel_command(port) -> list[str]` →
  `["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]`.
- `find_url(line: str) -> str | None` — regex `https://[a-z0-9-]+\.trycloudflare\.com`
  over a single output line (cloudflared logs the URL to stderr). Pure + unit-testable.

**`aai_cli/init/runner.py`** gains a small helper so `share` can run two processes
concurrently without duplicating Popen handling:

- `spawn(command, *, cwd, env=None, capture=False) -> subprocess.Popen` — thin Popen
  wrapper (`capture=True` sets `stdout=PIPE, stderr=STDOUT, text=True` for reading
  cloudflared's URL). `run_server` stays as-is for `dev`.

**New `aai_cli/commands/share.py`** — `run_share(*, port, no_install, json_mode)`:

1. `target = cwd`; `chosen_port = find_free_port(port)`; `env = {**os.environ, "PORT": str(chosen_port)}`.
2. `web = procfile.web_argv(target, env=env)` (validates project).
3. Require cloudflared: `if shutil.which(tunnel.CLOUDFLARED) is None: raise missing_dependency`.
4. Install (unless `--no-install`) via `devserver.install_step`; failed → exit 1.
5. `server = runner.spawn(devserver.dev_command(target, web, use_uv=use_uv), cwd=target, env=env)`;
   `runner.wait_for_port(chosen_port)`; if the server exited early → error + stop.
6. `proxy = runner.spawn(tunnel.tunnel_command(chosen_port), cwd=target, capture=True)`;
   read `proxy.stdout` line by line, `tunnel.find_url` until found or the stream ends
   (bounded loop). Print `Sharing <public-url>  ->  http://localhost:<port>`.
7. Block on `server.wait()`; on `KeyboardInterrupt` terminate both. `finally` always
   terminates both processes.

JSON mode: emit `{ "url": ..., "local": ..., "port": ... }` then still block (Ctrl-C
to stop), matching how `dev` treats `--json`.

**Registration:** `main.py` imports + `app.add_typer(share.app)`, `_COMMAND_ORDER`
gets `"share"` after `"dev"`, `rich_help_panel=help_panels.BUILD`. Add `dev` and
`share` to the `.importlinter` "Command modules are independent" contract. Add a
`cloudflared` row to `aai doctor`'s tool checks (optional/best-effort, like ffmpeg).

### Testing

- `tests/test_tunnel.py`: `find_url` matches a real cloudflared banner line / returns
  None otherwise; `tunnel_command` shape.
- `tests/test_devserver.py`: `install_step` (skipped/failed/installed), `dev_command`
  (uv vs venv, `--reload` appended).
- `tests/test_share.py` (mock `runner.spawn`, `runner.wait_for_port`, `runner.find_free_port`,
  `shutil.which`, and a fake proxy process whose stdout yields a trycloudflare line):
  prints the public URL; missing cloudflared → exit 1 with brew hint; missing Procfile →
  exit 1; install failure → exit 1, no spawn; both processes terminated on exit; `--json`
  emits the url payload.
- Snapshot regen for `aai --help` + `aai share --help`.

## `aai deploy` — confirm, then `vercel deploy`

**Goal:** deploy the current project to Vercel from the CLI, guarded by a yes/no
confirmation.

### Decisions

- **Confirm first:** prompt `Deploy this project to Vercel? [y/N]` and abort on anything
  but yes (exit 0, no error). `--yes`/`-y` skips the prompt (for automation); when output
  is non-interactive/agentic and `--yes` wasn't passed, abort with a clear message rather
  than hang.
- **Require the Vercel CLI:** `shutil.which("vercel")` → missing → `CLIError(missing_dependency)`
  with an `npm i -g vercel` hint. (Vercel CLI is npm-distributed; not added to the brew
  formula.)
- Run `vercel deploy` in the current directory, streaming its output; `--prod` passes
  through to `vercel deploy --prod`. Propagate vercel's exit code.

### Architecture

**New `aai_cli/commands/deploy.py`** — `run_deploy(*, prod, assume_yes, json_mode)`:

1. Require `vercel` via `shutil.which`; missing → `missing_dependency` error.
2. Confirm: unless `assume_yes`, prompt y/N (using `typer.confirm`); on no → print
   "Aborted." and return (exit 0). If non-interactive and not `assume_yes` → usage error.
3. `cmd = ["vercel", "deploy"] + (["--prod"] if prod else [])`; run via
   `subprocess.run(cmd, cwd=Path.cwd())` (inherit stdio so vercel's own progress shows).
   `raise typer.Exit(code=result.returncode)` when non-zero.

Flags: `--prod`, `--yes/-y`, `--json`. Registered under "Build an App" after `share`;
added to the import-linter independence contract.

### Testing

- `tests/test_deploy.py` (mock `shutil.which`, `subprocess.run`, and `typer.confirm`):
  confirm-no aborts without running vercel; `--yes` skips the prompt and runs;
  missing vercel → exit 1 with npm hint; `--prod` adds the flag; non-zero vercel exit
  propagates; non-interactive without `--yes` → usage error.
- Snapshot regen for `aai --help` + `aai deploy --help`.

## Out of scope

- Named/persistent cloudflared tunnels and custom domains.
- Vercel project linking/env setup (delegated to `vercel` itself).
- Non-Vercel deploy targets (the Procfile already covers Render/Railway/Heroku/Cloud Run).
