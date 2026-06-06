`POST /api/datasets` body (every field optional; defaults shown):

| field | default | meaning |
|---|---|---|
| `pca_dims` | 32 | PCA dims of the embedding block (1..=1536) |
| `test_ratio` | 0.2 | test split fraction (0.05..=0.5) |
| `seed` | 42 | split shuffle seed |
| `log_target` | true | train on log1p({{target_field}}) |
| `fields` | `{}` | per-field enables, keyed by field or group name (below); when non-empty, authoritative |
| `vocab_top_n` | `{}` | per-categorical vocabulary-size overrides (field name → top-N) |
| `area_content_backfill` | false | backfill missing areas from "… m²" mentions in the document text |
| `impute_numerics` | false | fill missing reconciled numerics with train-split group medians (drops indicator columns) |
| `numerics_collection` | `"{{reconcile_collection}}"` | companion collection for the reconciled-numerics join |
| `collection` | server's | source Qdrant collection |
| `quality` | see below | quality filter config |
| `currency` | see below | currency handling |

The feature fields come from the domain config (`GET /api/domain`, or read
`domain.toml`); pass them in `fields`, e.g.
`{"fields": {"raw_numerics": true, "coordinates": true, "city": false}}`.
For this domain:

| field | default | what it is |
|---|---|---|
{{dataset_feature_rows}}

(Pre-domain clients may still send the legacy named flags — `bedrooms`,
`property_type`, `neighborhood_top_n`, `city`, `province`, `cluster`,
`raw_numerics`, `coordinates` — which apply only when `fields` is empty.)

`quality` (rule toggles + thresholds; default excludes only nonpositive
target values — rule keys are stable identifiers bound to this domain's
fields by domain.toml): `nonpositive_price` (true), `price_outlier` (false,
MAD z on the log target per outlier group, threshold `price_outlier_mad_z`
3.5), `price_range` (false, hard caps `price_min` / `price_max`),
`bedrooms_outlier` (false, cap `bedrooms_max` on the domain's capped
numeric), `duplicate_content` (false), `short_content` (false, floor
`short_content_min_chars` 80), `missing_fields` (false, the domain's
critical fields).

`currency` (domains with a `[currency]` section in domain.toml; single-
currency domains ignore it): `mode` `"filter"` (default; also `"off"` |
`"convert"`), `keep` `"{{currency_keep}}"`, `reconcile_collection`
`"{{reconcile_collection}}"`, `rate_source` (the rate series for `convert`).
