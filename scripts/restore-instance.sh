#!/usr/bin/env bash
# restore-instance.sh — rebuild this instance from a bundle on a fresh machine.
#
# Consumes a bundle produced by scripts/export-instance.sh (Postgres dump +
# Qdrant snapshots + data/ essentials + manifest.json) and restores all three
# stores so that `zig build serve` comes up with the same models, runs,
# promotions and corpus the bundle was captured from.
#
# Source the bundle from a local directory, a single .tar(.zst/.gz) file, or HTTP:
#   scripts/restore-instance.sh /path/to/instance-bundle
#   scripts/restore-instance.sh /path/to/instance-bundle.tar     # e.g. from Google Drive
#   DUMP_BASE_URL=https://<r2-base>/ scripts/restore-instance.sh
#
# Safety: refuses to clobber a populated instance unless FORCE=1, and aborts if
# a server with live training runs is reachable (those die on disruption).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- config ------------------------------------------------------------------
SRC_DIR="${1:-}"                                # local bundle dir (optional)
DUMP_BASE_URL="${DUMP_BASE_URL:-}"              # http base (optional)
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
PG_SERVICE="${PG_SERVICE:-postgres}"
PG_USER="${LENSING_DB_USER:-pg}"
PG_DB="${LENSING_DB_NAME:-lensing}"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"
FORCE="${FORCE:-0}"

log()  { printf '\033[1;36m[restore]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[restore]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[restore] %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker  >/dev/null || die "docker not found"
command -v curl    >/dev/null || die "curl not found"
command -v python3 >/dev/null || die "python3 not found"

[ -n "$SRC_DIR" ] || [ -n "$DUMP_BASE_URL" ] \
  || die "no source: pass a bundle dir as \$1 or set DUMP_BASE_URL"

# ---- 0. guard against clobbering a live/populated instance --------------------
if curl -fsS --max-time 3 "$SERVER_URL/api/runs" >/dev/null 2>&1; then
  live=$(curl -fsS "$SERVER_URL/api/runs" 2>/dev/null \
         | python3 -c 'import sys,json;d=json.load(sys.stdin);print(sum(1 for r in (d if isinstance(d,list) else d.get("runs",[])) if str(r.get("status","")).lower() in ("running","queued","training")))' 2>/dev/null || echo 0)
  [ "$live" = "0" ] || die "server has $live live run(s) at $SERVER_URL — stop them before restoring"
  warn "a server is already running at $SERVER_URL; restoring underneath it is unsafe — stop it first"
  [ "$FORCE" = "1" ] || die "refusing to restore while the server is up (set FORCE=1 to override)"
fi
if [ -e data/models ] && [ "$FORCE" != "1" ]; then
  die "data/models already exists — this instance looks populated (set FORCE=1 to overwrite)"
fi

# ---- 1. obtain the bundle locally --------------------------------------------
WORK=""
cleanup() { [ -n "$WORK" ] && [ -d "$WORK" ] && rm -rf "$WORK"; }
trap cleanup EXIT
if [ -n "$SRC_DIR" ] && [ -d "$SRC_DIR" ]; then
  BUNDLE="$SRC_DIR"
  log "using local bundle dir: $BUNDLE"
elif [ -n "$SRC_DIR" ] && [ -f "$SRC_DIR" ]; then
  WORK="$(mktemp -d)"; BUNDLE="$WORK"
  log "unpacking single-file bundle: $SRC_DIR"
  case "$SRC_DIR" in
    *.tar.zst|*.tzst) command -v zstd >/dev/null || die "need zstd to unpack $SRC_DIR"
                      tar --use-compress-program=zstd -C "$BUNDLE" -xf "$SRC_DIR" ;;
    *.tar.gz|*.tgz)   tar --gzip -C "$BUNDLE" -xf "$SRC_DIR" ;;
    *.tar)            tar -C "$BUNDLE" -xf "$SRC_DIR" ;;
    *) die "unrecognized bundle file: $SRC_DIR (expected a dir or .tar/.tar.zst/.tar.gz)" ;;
  esac
elif [ -n "$SRC_DIR" ]; then
  die "bundle source not found: $SRC_DIR"
else
  WORK="$(mktemp -d)"
  BUNDLE="$WORK"
  base="${DUMP_BASE_URL%/}"
  log "fetching manifest from $base/manifest.json…"
  curl -fsS "$base/manifest.json" -o "$BUNDLE/manifest.json" || die "manifest fetch failed"
  for f in $(python3 -c 'import json;print(" ".join(x["name"] for x in json.load(open("'"$BUNDLE"'/manifest.json"))["files"]))'); do
    log "downloading $f…"
    curl -fS "$base/$f" -o "$BUNDLE/$f" || die "download failed: $f"
  done
fi

MANIFEST="$BUNDLE/manifest.json"
[ -f "$MANIFEST" ] || die "manifest.json missing from bundle"
read_manifest() { python3 -c 'import json,sys;print(eval(sys.argv[2],{"m":json.load(open(sys.argv[1]))}))' "$MANIFEST" "$1"; }
DATA_ARCHIVE=$(read_manifest 'm["data_archive"]')
PG_DUMP=$(read_manifest 'm["postgres"]["dump"]')
COLLECTIONS=$(read_manifest '" ".join(m["qdrant"]["collections"])')
log "manifest: scope=$(read_manifest 'm.get("bundle_scope","?")') git=$(read_manifest 'm.get("git_rev","?")[:10]') collections=[$COLLECTIONS]"

# ---- 1b. verify checksums ----------------------------------------------------
if command -v sha256sum >/dev/null; then
  log "verifying checksums…"
  python3 -c 'import json;[print(f["sha256"],f["name"]) for f in json.load(open("'"$MANIFEST"'"))["files"]]' \
    | while read -r sha name; do echo "$sha  $BUNDLE/$name"; done \
    | sha256sum -c - >/dev/null || die "checksum mismatch — bundle is corrupt or incomplete"
fi

# ---- 2. bring up infra (Postgres + a local Qdrant) ---------------------------
log "starting Postgres + Qdrant via docker compose…"
docker compose --profile qdrant up -d --wait postgres qdrant 2>/dev/null \
  || docker compose --profile qdrant up -d postgres qdrant   # qdrant has no healthcheck
log "waiting for Qdrant at $QDRANT_URL…"
for _ in $(seq 1 60); do
  curl -fsS "$QDRANT_URL/readyz" >/dev/null 2>&1 && break
  curl -fsS "$QDRANT_URL/collections" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "$QDRANT_URL/collections" >/dev/null 2>&1 || die "Qdrant not reachable at $QDRANT_URL"

# ---- 3. restore Postgres -----------------------------------------------------
log "restoring Postgres into '$PG_DB'…"
docker compose exec -T "$PG_SERVICE" \
  pg_restore -U "$PG_USER" -d "$PG_DB" --clean --if-exists --no-owner --no-privileges \
  < "$BUNDLE/$PG_DUMP" \
  || warn "pg_restore reported errors (often harmless --clean drops on a fresh DB)"

# ---- 4. restore Qdrant snapshots ---------------------------------------------
for col in $COLLECTIONS; do
  snap="$BUNDLE/qdrant-$col.snapshot"
  [ -f "$snap" ] || die "missing snapshot for '$col': $snap"
  log "uploading snapshot for '$col'…"
  curl -fsS -X POST \
    "$QDRANT_URL/collections/$col/snapshots/upload?priority=snapshot" \
    -H 'Content-Type:multipart/form-data' \
    -F "snapshot=@$snap" >/dev/null \
    || die "snapshot upload failed for '$col'"
  cnt=$(curl -fsS "$QDRANT_URL/collections/$col" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"].get("points_count"))' 2>/dev/null || echo "?")
  log "  '$col' restored ($cnt points)"
done

# ---- 5. restore data/ artifacts ----------------------------------------------
log "extracting $DATA_ARCHIVE into data/…"
if [ "$FORCE" = "1" ] && [ -e data/models ]; then warn "overwriting existing data/ artifacts"; fi
case "$DATA_ARCHIVE" in
  *.zst) command -v zstd >/dev/null || die "need zstd to extract $DATA_ARCHIVE"
         tar --use-compress-program=zstd -xf "$BUNDLE/$DATA_ARCHIVE" ;;
  *.gz)  tar --gzip -xf "$BUNDLE/$DATA_ARCHIVE" ;;
  *)     die "unknown archive type: $DATA_ARCHIVE" ;;
esac

# ---- 6. .env reminder --------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null || true
  warn "created .env from .env.example — set OPENAI_API_KEY before adding manual entries"
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Instance restored. Final steps:

  1. Set OPENAI_API_KEY in .env (needed only for POST /api/listings).
  2. Start the server, pointing Qdrant at the restored local instance:

       zig build serve -Dqdrant-url=$QDRANT_URL

  3. (optional) Reconcile Postgres against the restored files:

       zig build migrate-data        # idempotent, prints a consistency report

Open http://localhost:8080 — your models, runs and best-models group are back.
────────────────────────────────────────────────────────────────────────────
EOF
