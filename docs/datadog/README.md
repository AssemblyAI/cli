# Datadog dashboard — CLI usage & reliability

`cli-usage-dashboard.json` is an importable Datadog dashboard for the `aai` CLI's
anonymous usage telemetry (`aai_cli/telemetry.py`). The CLI ships one log per
command run to the Datadog **Logs** intake (`source:aai-cli`) — it is Logs, **not**
RUM. Each log line = one command run.

## Prerequisite: create the facets

Log-analytics widgets only work on attributes that are **facets**. Before the
dashboard shows data, add these in **Logs → Configuration → Facets** (or expand any
`source:aai-cli` log and click each attribute → *Create facet*):

- Facets: `@command`, `@outcome`, `@cli_version`, `@os`, `@python_version`,
  `@exit_code`, `@ci`, `@device_id`
- Measure (numeric, not a plain facet): `@duration_ms`

`@device_id` (unique-device counts) and `@duration_ms` (latency percentiles) are the
two that matter most.

## Error Tracking

Failures ship with `status:error` and the reserved `@error.kind` attribute (set to the
anonymous error type — the same value as `@outcome`, e.g. `api_error`, `not_authenticated`),
so they feed Datadog **Error Tracking** (issue grouping, first-/last-seen, regression
detection, alerting), not just log search. `error.kind` is enough to enable tracking on
its own. The error message and stack trace are **deliberately omitted** — no free text or
PII ever leaves the machine — so issues group by error *type × command*, not by stack.
The error tiles on the dashboard filter on `status:error`.

## Import

UI: **Dashboards → New Dashboard → ⚙️ → Import dashboard JSON**, paste the file.

API:

```sh
curl -X POST "https://api.datadoghq.com/api/v1/dashboard" \
  -H "DD-API-KEY: $DD_API_KEY" -H "DD-APPLICATION-KEY: $DD_APP_KEY" \
  -H "Content-Type: application/json" \
  -d @docs/datadog/cli-usage-dashboard.json
```

## Notes

- "Error rate" counts any `status:error` (every non-success outcome) — that includes
  expected user errors (e.g. `not_authenticated`), not just crashes. Filter to
  `@error.kind:internal_error` for a crash-only view.
- `@device_id` is a per-machine random UUID, so it approximates installs, not people.
- Events take ~1–2 minutes to index; a fresh `source:aai-cli` search can look empty
  for a minute even when delivery succeeded.
