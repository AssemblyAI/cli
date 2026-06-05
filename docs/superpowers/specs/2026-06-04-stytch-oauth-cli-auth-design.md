# Stytch OAuth login for the AssemblyAI CLI — design

**Date:** 2026-06-04
**Status:** Draft for review
**Author:** Alex Kroman (with Claude)

## Goal

Replace the interactive "paste an API key" login with a browser-based Stytch
OAuth login (`aai auth login`). After login, the user holds a dedicated,
revocable AssemblyAI API key — exactly the credential the CLI already uses — so
nothing downstream changes.

## Non-goals

- No change to how the CLI calls AssemblyAI APIs. Every command keeps using an
  API key via `aai.settings.api_key` (`client.py`).
- No persisted OAuth tokens. OAuth is a *login mechanism to obtain the API key*,
  nothing more (see "Credential model").
- No headless/device flow. The CLI assumes a browser is available (per product
  decision); we fall back to printing the URL if it can't auto-open.

## Decisions (settled during brainstorming)

1. **Browser flow = client-side, loopback redirect.** The CLI runs the browser
   login itself (vs. the Stripe-style backend-brokered poll flow). Concretely this
   is **Stytch B2B OAuth discovery** (public_token + loopback; no PKCE/code
   exchange) — *not* Connected Apps — because that's what AMS consumes (see O1).
2. **Credential model = store-key-only.** Login ends by storing an AssemblyAI
   API key in the OS keychain, exactly as today. OAuth access/refresh tokens are
   used transiently and discarded. This is the Stripe-CLI pattern and keeps the
   entire existing call path untouched.
3. **Dedicated, find-or-create CLI key.** Login provisions/reuses an API key
   named `AssemblyAI CLI` (ideally per-device, e.g. `AssemblyAI CLI — <hostname>`)
   rather than reusing the account's primary key. Independent revocation,
   auditability, smaller blast radius.
4. **Keep escape hatches.** `ASSEMBLYAI_API_KEY` env var and a non-interactive
   `--api-key` on `login` remain, for CI/automation. Only the *interactive
   default* changes to OAuth.
5. **Profiles unchanged.** Login writes the key into the active profile's
   keychain slot via the existing `config.set_api_key`.

## Confirmed external facts

### Stytch project (test)

| Thing | Value |
|---|---|
| Project ID | `project-test-55761beb-2738-4258-825d-be699c3b5336` |
| Project domain | `https://psychedelic-journey-5884.customers.stytch.dev` |
| Connected App client ID (test) | `connected-app-test-4a88e99a-ac79-4c2a-82b5-027e2231a307` |
| Workspace / org | `organization-prod-f12ebd5e-17c6-415e-be27-d375635f0b39` |
| Token endpoint | `https://{project-domain}/v1/oauth2/token` |
| Loopback redirect (registered) | `http://127.0.0.1/callback` (no port → any port allowed) |
| Client type | First Party Public (PKCE, no client secret) |
| `offline_access` consent | bypassed (toggle on) |
| `full_access` / Access Token Exchange | enabled (toggle on) |

Discovery (`/.well-known/openid-configuration`) returns HTTP 400
`authorization_endpoint_not_configured_for_project` (Connected Apps OIDC). This
is **moot** for the chosen design: the CLI uses Stytch **B2B OAuth discovery**,
not Connected Apps (see O1). The Connected App rows above are retained for record
but are **unused**. What the design needs instead: the project's **public_token**
and a B2B OAuth provider + loopback redirect allowlist (P1).

### Accounts Management Service (AMS) — the "token API"

Source: `AssemblyAI/DeepLearning` → `assemblyai/experimental/fcastillo/bruno`
("Accounts Management Service" collection). All user endpoints authenticate with
a **`Cookie: stytch_session_jwt=<JWT>`** (a Stytch *member session*, not the
OAuth access token).

| Purpose | Method + path | Notes |
|---|---|---|
| Who am I | `GET {ams}/v1/auth` | Returns `id` (account_id), `email`, `api_token`. |
| List projects + their keys | `GET {ams}/v1/users/accounts/{account_id}/projects` | Each project has `tokens[]` with `api_key`, `name`. Enables **find**. |
| Create API key | `POST {ams}/v1/users/accounts/{account_id}/tokens` body `{project_id, token_name}` | Returns `{id, project_id, api_key, name, ...}`. Enables **create**. |
| Rename key | `PUT {ams}/v1/users/accounts/{account_id}/tokens/{token_id}` `{token_name}` | |
| Delete key | `DELETE {ams}/v1/users/accounts/{account_id}/tokens/{token_id}` | Used by `logout` revoke (optional). |

Base URLs: prod in the collection is `https://ams.internal.assemblyai-labs.com`
(internal); a **publicly reachable sandbox exists** at
`https://ams.sandbox000.assemblyai-labs.com` (OpenAPI confirmed live, see below).

### AMS OpenAPI (confirmed live at sandbox `/openapi.json`, 2026-06-04)

- **Auth: global `BearerAuth` (HTTP bearer, JWT)** applied to all endpoints
  (`security: [{BearerAuth: []}]`). `/v1/auth` additionally accepts a
  `stytch_session_jwt` **cookie** (optional param).
- **AMS brokers the Stytch session exchange itself** — no project secret needed
  in the CLI:
  - `POST {ams}/v2/auth/discover` body `{token, token_type}` where
    `token_type ∈ {"discovery", "discovery_oauth"}` → returns
    `{organizations[] (id+name), email, intermediate_session_token}`.
  - `POST {ams}/v2/auth/exchange` body `{intermediate_session_token,
    organization_id}` → `SignedInResponse {account, session_jwt, session_token}`.
    (0 orgs → `POST /v2/auth/organization` to create one; MFA path possible.)
- **`TokenSchema` has no expiry field** → AssemblyAI keys are long-lived
  → store-key-only needs no refresh (resolves O2).
- `CreateTokenRequest = {project_id:int, token_name:str}` →
  `TokenSchema {id, project_id, api_key, name, is_disabled, created, updated}`.
- `GET …/projects` → `ProjectDetailResponse[] = {project, tokens[]}` (each token
  has `api_key`+`name`) → supports find-or-create directly.

### The access_token → session_jwt step

AMS owns this server-side via `/v2/auth/discover` + `/v2/auth/exchange` (it holds
the Stytch project secret). The CLI never needs the secret and never calls
Stytch's backend "Exchange Access Token" endpoint directly. The one thing to
confirm with the AMS owner is **which credential `/v2/auth/discover` expects**
(see O1).

## Architecture — Stytch B2B OAuth discovery via loopback (chosen)

The CLI drives Stytch's **B2B OAuth discovery** (public_token + loopback redirect,
**no PKCE/code exchange** — the discovery token arrives directly in the redirect),
then AMS brokers session exchange and key provisioning through its **existing**
endpoints. No backend changes, Connected App not used.

```
aai auth login
  1. resolve profile; generate random `state` (CSRF)
  2. bind loopback HTTP server on 127.0.0.1:8585  (fixed; exact-match redirect)
  3. open browser → Stytch B2B OAuth discovery start:        [CLI → Stytch, client-side]
        https://{project-domain}/v1/b2b/public/oauth/{provider}/discovery/start
          ?public_token={public_token}
          &discovery_redirect_url=http://127.0.0.1:<port>/callback
  4. user authenticates with provider → Stytch redirects to the loopback:
        http://127.0.0.1:<port>/callback?stytch_token_type=discovery_oauth&token=<T>
  5. callback handler captures `token` (verify stytch_token_type=discovery_oauth)
  6. POST {ams}/v2/auth/discover {token, token_type:"discovery_oauth"}   [CLI → AMS]
        → {organizations[], email, intermediate_session_token}
  7. choose organization_id (1 org → use it; >1 → pick/prompt; 0 → /v2/auth/organization)
  8. POST {ams}/v2/auth/exchange {intermediate_session_token, organization_id}
        → SignedInResponse {account, session_jwt, session_token}
  9. GET {ams}/v1/auth   (Cookie: stytch_session_jwt=session_jwt)  → account.id
 10. GET {ams}/v1/users/accounts/{account_id}/projects
        → find token named "AssemblyAI CLI"; else
     POST {ams}/v1/users/accounts/{account_id}/tokens {project_id, token_name}
        → {api_key}
 11. config.set_api_key(profile, api_key)            [unchanged storage path]
✓ all other commands work exactly as before
```

Notes: the start endpoint is **client-side**, authenticated by the project's
**public_token** (not the secret) — safe to ship in the CLI. Steps 9–10 may
present `session_jwt` as `Authorization: Bearer` (global AMS scheme) instead of
the cookie (O5). Handle MFA/verify-email responses from exchange (rare for the
CLI; surface a clear message and stop).

## Components / files

- **`assemblyai_cli/auth/` (new package)**
  - `loopback.py` — bind `127.0.0.1:0`, serve `/callback`, capture `token` +
    `stytch_token_type`, return a "you can close this tab" page, enforce a timeout.
  - `browser.py` — open system browser to the B2B discovery start URL; on failure,
    print the URL.
  - `discovery.py` — build the B2B OAuth discovery start URL (provider +
    public_token + loopback redirect); generate/verify `state`.
  - `provision.py` — AMS chain: `discover` → `exchange` → `/v1/auth` →
    find-or-create token; returns the `api_key`.
  - `endpoints.py` — config constants (below), env-overridable.
- **`assemblyai_cli/commands/login.py` (modify)**
  - `login` → orchestrates the flow above; keeps `--api-key` non-interactive path
    and `--json`.
  - `logout` → `config.clear_api_key` (existing) + best-effort AMS `DELETE` of the
    CLI token (revoke). Keep local delete authoritative; revoke is best-effort.
  - `whoami` → unchanged (operates on the stored key).
- **`assemblyai_cli/config.py`** — no schema change; reuse `set_api_key` /
  `get_api_key` / `clear_api_key`. Optionally store `account_id`/`token_id`
  alongside (in `config.toml` profile) to make `logout` revoke precise.

### Config constants (`endpoints.py`), env-overridable for sandbox→prod

- `STYTCH_PROJECT_DOMAIN` (`https://psychedelic-journey-5884.customers.stytch.dev`)
- `STYTCH_PUBLIC_TOKEN` (**needed** — from Stytch dashboard; public, safe to ship)
- `STYTCH_OAUTH_PROVIDER = "google"` (enabled + verified on the project)
- `LOOPBACK_REDIRECT = "http://127.0.0.1:8585/callback"` (fixed port; exact-match
  validation — Stytch rejects unregistered ports/paths. Single registered URL.)
- `AMS_BASE_URL` (`https://ams.sandbox000.assemblyai-labs.com`; prod URL per P2)
- `CLI_TOKEN_NAME = "AssemblyAI CLI"`

Hardcode sandbox defaults; allow override via env (e.g. `AAI_AUTH_*`) so swapping
to production (AMS prod URL, prod project) is config, not code. The Connected App
client ID is **not used** in this design.

## Error handling

- **Browser won't open** → print the authorize URL for manual paste.
- **Callback timeout** (e.g. 120s) → abort with a clear message; tear down the
  loopback server.
- **`state` mismatch** → abort (possible CSRF); do not exchange.
- **`stytch_token_type` not `discovery_oauth`** on callback → abort with a clear
  message (misconfigured provider/redirect).
- **AMS discover/exchange failure** → AMS already maps Stytch errors to clean
  HTTP (401 invalid/expired credentials or session token, etc.); surface its
  `detail`. Map 401/403 to a `NotAuthenticated`-style message, else `APIError`.
- **Expired discovery token** (slow user) → AMS returns 401
  (`oauth_token_not_found`/expired); instruct retry of `aai auth login`.
- **MFA / verify-email required** from exchange → surface the required action and
  stop (out of scope for v1 CLI; revisit if accounts enforce MFA).
- All errors flow through the existing `run_command` → `CLIError` → exit-code
  machinery.

## Testing

Mirror existing test patterns in `tests/` (see `test_login*`-style if present).

- `discovery.py`: start-URL builder includes provider, public_token, redirect;
  `state` generated and verified; rejects mismatched `state`.
- `loopback.py`: binds an ephemeral port; captures `token`/`stytch_token_type`;
  rejects wrong token_type; honors timeout. Drive with a synthetic GET to the port.
- `provision.py`: AMS `discover`→`exchange`→`/v1/auth`→projects/tokens mocked
  (httpx/requests mock) for success + 401; find-vs-create logic; multi-org and
  MFA/verify-email branches.
- `login` command: end-to-end with browser + servers mocked; asserts
  `config.set_api_key` called with the returned key; `--api-key` non-interactive
  path still works; `--json` output shape.
- `logout`: clears key; best-effort revoke tolerates AMS being unreachable.

## Prerequisites (backend / outside the CLI)

- **P1 — DONE & verified (2026-06-04).** `http://127.0.0.1:8585/callback` is
  registered and returns a live 307 redirect to Google. Validation is
  exact-match, so the CLI binds the fixed port 8585. Google OAuth is enabled using
  Stytch's shared **test** credentials (no own Google app needed in test). Public
  token `public-token-test-79ad7d8d…` verified against project
  `project-test-55761beb`.
- **P-NEW — Confirm AMS↔project binding (only unverified link).** Confirm
  `ams.sandbox000.assemblyai-labs.com` uses Stytch project `project-test-55761beb`
  and that `alex@assemblyai.com` has an account/org there. AMS is alive (`/v1/auth`
  → 401 unauth; `/v2/auth/discover` processes via Stytch). Settle via fcastillo or
  the first real end-to-end login. Note: a first-time user with **0 orgs** hits the
  `/v2/auth/organization` create branch — the CLI must handle it.
- **P2 — Blessed public, stable AMS URL.** A public sandbox is already reachable
  (`ams.sandbox000.assemblyai-labs.com`); a production public URL (out of
  `experimental/`) is needed for release. *(Downgraded: the "internal-only,
  unreachable" risk is resolved — a public ingress demonstrably exists.)*

*(Former P3 removed: AMS already brokers the Stytch exchange via
`/v2/auth/discover` + `/v2/auth/exchange`; no new endpoint is required.)*

## Open questions

- **O1 — RESOLVED by source** (`integration/auth_gateway.py`). AMS uses
  `stytch.B2BClient` and `discovery_oauth` maps to
  `oauth.discovery.authenticate(discovery_oauth_token=token)` — Stytch **B2B OAuth
  discovery**, the same primitive the AssemblyAI dashboard uses. AMS validates
  sessions via B2B member-session JWTs (`sessions.authenticate_jwt`).
  **The Connected App (`connected-app-test-…`) is NOT consumed by AMS** — it's a
  different Stytch primitive. The CLI must obtain a Stytch B2B login credential
  (`discovery_oauth_token`, magic-link, or password), not a Connected Apps token.
  **Remaining decision (D1, for the user):** use the existing B2B discovery flow
  (recommended; pick sub-method below), or have the backend add Connected Apps
  support to AMS (extra backend work, not recommended).
- **D1 sub-method** (if using B2B discovery): OAuth provider via loopback
  (e.g. Google/Microsoft — most "OAuth-like"), magic link (email round-trip), or
  email+password (no browser). Provider + loopback redirect URL must be
  allowlisted in Stytch.
- **O2 — resolved.** `TokenSchema` has no expiry → keys are long-lived →
  store-key-only, no refresh needed.
- **O3.** Project selection for multi-project accounts: default to the first/only
  project, or prompt?
- **O4.** Production Connected App client ID (current one is `…-test-…`).
- **O5.** Auth presentation on AMS user endpoints: `stytch_session_jwt` cookie
  (per `/v1/auth`) vs. `Authorization: Bearer session_jwt` (global scheme).
```
