#!/usr/bin/env bash
# ensure-qdrant.sh — make sure the Qdrant corpus this instance points at is up
# before `zig build serve` / `dataset`. The corpus normally lives in an external
# docker container (its name in $LENSING_QDRANT_CONTAINER, read from .env). If the
# URL is already healthy we do nothing; if it's down and we have a container we
# can manage, we start it and wait; otherwise we only WARN (never block the
# server — it degrades to corpus-less serving on its own).
#
#   scripts/ensure-qdrant.sh [QDRANT_URL]
#
# Config (process env wins, else .env): LENSING_QDRANT_CONTAINER, QDRANT_URL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Read KEY from process env, else from .env (first match), stripping quotes.
read_cfg() {
  local key="$1" val=""
  if [ -n "${!key:-}" ]; then printf '%s' "${!key}"; return; fi
  [ -f .env ] || return 0
  val="$(sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?${key}=//p" .env | head -n1)"
  val="${val%$'\r'}"
  val="${val#\"}"; val="${val%\"}"
  val="${val#\'}"; val="${val%\'}"
  printf '%s' "$val"
}

QURL="${1:-$(read_cfg QDRANT_URL)}"
QURL="${QURL:-http://localhost:6333}"
CONTAINER="$(read_cfg LENSING_QDRANT_CONTAINER)"

healthy() {
  curl -fsS --max-time 3 "$QURL/healthz" >/dev/null 2>&1 \
    || curl -fsS --max-time 3 "$QURL/" >/dev/null 2>&1
}

if healthy; then
  exit 0   # corpus already up — stay out of the way
fi

if [ -n "$CONTAINER" ] && command -v docker >/dev/null 2>&1 \
   && docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ensure-qdrant: $QURL down — starting container '$CONTAINER'…" >&2
  docker start "$CONTAINER" >/dev/null
  for _ in $(seq 1 30); do
    if healthy; then
      echo "ensure-qdrant: corpus healthy at $QURL." >&2
      exit 0
    fi
    sleep 1
  done
  echo "ensure-qdrant: WARNING — '$CONTAINER' started but $QURL still not answering; continuing." >&2
  exit 0
fi

printf '\033[1;33mensure-qdrant: WARNING — %s unreachable and no manageable container.\n' "$QURL" >&2
if [ -z "$CONTAINER" ]; then
  printf '  Set LENSING_QDRANT_CONTAINER=<name> in .env to auto-start the corpus container,\n' >&2
  printf '  or run a local one: docker compose --profile qdrant up -d (empty volume).\n' >&2
fi
printf '  Serving without the corpus (predictions on existing datasets still work).\033[0m\n' >&2
exit 0
