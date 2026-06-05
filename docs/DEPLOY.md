# Deployment topologies

One binary, three roles. Postgres is the coordination point; binary
artifacts travel through it (runs, model snapshots) or over the hub's HTTP
API (dataset archives), so no role needs a shared filesystem.

```
            ┌──────────────────────────── Postgres ───────────────────────────┐
            │  definitions · runs (queue) · run_events · run_artifacts        │
            │  datasets index · models · model_artifacts (snapshots)          │
            └──────▲──────────────────▲──────────────────────▲────────────────┘
                   │                  │                       │
   ┌───────────────┴───┐   ┌──────────┴─────────┐   ┌─────────┴──────────┐
   │ HUB               │   │ TRAINING WORKER(S) │   │ INFERENCE NODE(S)  │
   │ lensing-server         │   │ lensing-server worker   │   │ lensing-server infer    │
   │ UI + API + builds │◄──┤ claims queued runs │   │ materializes       │
   │ local runs too    │   │ GET …/archive      │   │ promoted models    │
   │ owns data/ + the  │   │ trains, uploads    │   │ from the DB,       │
   │ Qdrant corpus     │   │ artifacts          │   │ serves /predict    │
   └───────────────────┘   └────────────────────┘   └────────────────────┘
```

## Roles

### Hub (default)

```sh
lensing-server --port 8080 --database-url $DATABASE_URL    # or: zig build serve
```

Everything as before: UI, API, dataset builds, local training. Runs posted
with `"queue": true` are enqueued for workers instead of training locally:

```sh
curl -s localhost:8080/api/runs -H content-type:application/json \
  -d '{"dataset_id":"ds-…","predictor":"xgboost","queue":true}'
```

### Training worker

```sh
lensing-server worker --hub-url http://hub:8080 --database-url $DATABASE_URL
lensing-server worker --hub-url … --once      # process one run and exit (batch/lambda)
```

Claims the oldest queued run (`FOR UPDATE SKIP LOCKED` — any number of
workers race safely), downloads the dataset from the hub's archive endpoint
(cached locally), trains, streams progress into `run_events`, writes the
terminal metrics into `runs`, and uploads the predictor-written outputs
(checkpoint, metrics, predictions, viz) into `run_artifacts`. The hub then
lists, inspects and **promotes** the run with no access to the worker's disk.

Host requirements: this binary, `registry.toml`, and the predictor
toolchains it should run (predictor binaries under `target/release/`,
`predictors/.venv`, julia envs) — i.e. an image/checkout of the repo.
Limitations: graceful STOP is not supported on remote runs yet; a worker
that dies mid-run leaves the run `running` (clear it by deleting the run).

### Inference node

```sh
lensing-server infer --infer-port 8090 --database-url $DATABASE_URL
```

Materializes every promoted model from `model_artifacts` into its local
cache at startup and serves ONLY: `/api/health`, `/api/domain`,
`/api/models`, `/api/models/:name` (+ `/contract`, `/viz`),
`POST /api/models/:name/predict`. No UI, no orchestration, no writes.

Host requirements: this binary + `registry.toml` + `domain.toml` + the
*predict* toolchains of the families it serves. Qdrant access is needed
only for `point_ids` predicts; inline-`items` predicts are self-contained.
Restart the node to pick up newly promoted models.

## Sizing notes

- Model snapshots are small (this corpus: 23 models ≈ 85 MB total in
  Postgres); dataset archives are tens of MB and cached on each worker.
- The queue claim is atomic per run; scale workers horizontally by just
  starting more of them (or invoking `--once` per queued run from a
  scheduler/lambda).
