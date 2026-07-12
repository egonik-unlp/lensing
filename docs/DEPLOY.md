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

## Deploy with Docker

The same three roles ship as containers. The build happens in disposable
builder stages of the `Dockerfile`; the runtime images carry only the binary +
the toolchains it needs to *run* (no Zig, no cargo, no Node). Two targets:

| Target | Image | Carries | For |
|---|---|---|---|
| `runtime` | `lensing-runtime` | server + pipeline + predictor zoo + `predictors/.venv` + `ui/dist` | hub, worker |
| `infer` | `lensing-infer` | server + Rust/burn predictor binaries only (no UI, no venv) | inference of Rust/burn families |

To serve Python-family models from an inference node, run it from the
`runtime` image instead (it carries the venv): set `LENSING_INFER_IMAGE` to a
`runtime`-target image.

Config comes from `.env` (copy `docker/.env.example` → `.env`, then
`python3 scripts/render-deploy-env.py --write` to seed the per-instance
identity/ports from `domain.toml`). `COMPOSE_PROJECT_NAME` prefixes container/
network/volume names so multiple instances coexist on one host (D1). Host ports
are parametrizable; container-internal ports are fixed (hub 8080, infer 8090,
Postgres 5432). State persists in named volumes: `pgdata`, `qdrant_storage`,
`lensing_data` (the hub's `data/`).

### Single host

Everything on one machine via `docker-compose.yml` profiles:

```sh
cp docker/.env.example .env && python3 scripts/render-deploy-env.py --write
docker compose --profile qdrant up -d           # optional local corpus
docker compose --profile hub up -d --build --wait        # DB + hub (== zig build docker-up)
docker compose --profile hub --profile worker up -d --scale worker=3
docker compose --profile infer up -d --build     # a predict-only node
```

The hub runs the idempotent `migrate-data` backfill before serving on a clean
DB (`LENSING_MIGRATE_ON_START=true`, hub-only). `zig build docker-up
-Ddocker-profile=hub|worker|infer` is the wrapper.

### Split across machines

One self-contained compose file per machine (no shared filesystem: datasets
travel over the hub's HTTP archive API, artifacts/models through Postgres — the
same contract as the host-run roles above). On each machine, `.env` points the
remote endpoints at the hub host.

```sh
# Machine A — server/hub (exposes :8080 and Postgres :5433 to the others)
docker compose -f docker/compose.hub.yml --env-file .env up -d --build --wait

# Machine B — training workers
#   .env: LENSING_HUB_URL=http://A:8080   DATABASE_URL=postgres://pg:pg@A:5433/lensing
docker compose -f docker/compose.worker.yml --env-file .env up -d --build --scale worker=3

# Machine C — inference
#   .env: DATABASE_URL=postgres://pg:pg@A:5433/lensing
docker compose -f docker/compose.infer.yml --env-file .env up -d --build
```

Open the firewall on the hub host for the hub port and the Postgres host port
so the remote roles can reach the queue, dataset API and metadata DB.
