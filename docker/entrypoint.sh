#!/usr/bin/env sh
# Container entrypoint: resolve the container's ROLE and exec the lensing-server
# binary in the matching mode. One image, three roles (see docs/DEPLOY.md).
#
# Role comes from $LENSING_ROLE (hub|worker|infer). DATABASE_URL is read by the
# binary directly from the environment (all roles). Role-specific knobs:
#   hub    : LENSING_HUB_PORT (8080), LENSING_QDRANT_URL, LENSING_COLLECTION?,
#            LENSING_MIGRATE_ON_START (true) — runs the idempotent migrate-data
#            backfill before serving on a clean DB. Only the hub does this.
#   worker : LENSING_HUB_URL (http://hub:8080), LENSING_WORKER_ONCE?, LENSING_POLL_SECS?
#   infer  : LENSING_INFER_PORT (8090), LENSING_QDRANT_URL
set -eu

ROLE="${LENSING_ROLE:-hub}"
ROOT="${LENSING_ROOT:-/app}"
BIN="$ROOT/target/release/lensing-server"
cd "$ROOT"

case "$ROLE" in
  hub)
    HUB_PORT="${LENSING_HUB_PORT:-8080}"
    QURL="${LENSING_QDRANT_URL:-http://qdrant:6333}"
    if [ "${LENSING_MIGRATE_ON_START:-true}" = "true" ]; then
      echo "[entrypoint] hub: migrate-data (idempotent backfill)…"
      "$BIN" --root "$ROOT" migrate-data
    fi
    set -- --root "$ROOT" --port "$HUB_PORT" --qdrant-url "$QURL"
    [ -n "${LENSING_COLLECTION:-}" ] && set -- "$@" --collection "$LENSING_COLLECTION"
    echo "[entrypoint] hub: serving on :$HUB_PORT (qdrant $QURL)"
    exec "$BIN" "$@"
    ;;

  worker)
    HUB_URL="${LENSING_HUB_URL:-http://hub:8080}"
    set -- worker --hub-url "$HUB_URL"
    [ "${LENSING_WORKER_ONCE:-false}" = "true" ] && set -- "$@" --once
    [ -n "${LENSING_POLL_SECS:-}" ] && set -- "$@" --poll-secs "$LENSING_POLL_SECS"
    echo "[entrypoint] worker: hub $HUB_URL"
    exec "$BIN" "$@"
    ;;

  infer)
    INFER_PORT="${LENSING_INFER_PORT:-8090}"
    QURL="${LENSING_QDRANT_URL:-http://qdrant:6333}"
    echo "[entrypoint] infer: serving predict on :$INFER_PORT"
    exec "$BIN" --root "$ROOT" --qdrant-url "$QURL" infer --infer-port "$INFER_PORT"
    ;;

  *)
    echo "[entrypoint] FATAL: unknown LENSING_ROLE='$ROLE' (expected: hub | worker | infer)" >&2
    exit 64
    ;;
esac
