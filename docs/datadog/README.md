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

- "Error rate" counts any `@outcome != success` — that includes expected user errors
  (e.g. `auth_failure`), not just crashes. Filter to `@outcome:internal_error` for a
  crash-only view.
- `@device_id` is a per-machine random UUID, so it approximates installs, not people.
- Events take ~1–2 minutes to index; a fresh `source:aai-cli` search can look empty
  for a minute even when delivery succeeded.
