#!/usr/bin/env python3
"""Derive the Docker deploy identity/ports from domain.toml and emit them as
.env lines — so container/network/volume names and host ports are per-instance
and never hardcoded into the compose files (they'd otherwise ship into every
bootstrap via upstream-sync and collide on a shared host; D1).

  python3 scripts/render-deploy-env.py            # print the deploy block
  python3 scripts/render-deploy-env.py --write     # idempotently upsert into .env

Only the keys this script owns are touched on --write; everything else in .env
is preserved. The Rust server never reads [deploy]; this is host-side wiring.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAIN = ROOT / "domain.toml"
ENV = ROOT / ".env"

# Keys this script owns (upserted on --write, in this order).
OWNED = ["COMPOSE_PROJECT_NAME", "LENSING_HUB_PORT", "LENSING_INFER_PORT", "LENSING_DB_PORT"]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
    return slug or "lensing"


def derive() -> dict[str, str]:
    with DOMAIN.open("rb") as fh:
        domain = tomllib.load(fh)
    deploy = domain.get("deploy", {})
    project = domain.get("project", {})

    project_name = str(project.get("name", "") or "")
    compose_project = str(deploy.get("compose_project", "") or "")
    if not compose_project:
        # Fall back to the project name, unless it's the unconfigured placeholder.
        compose_project = project_name if project_name and project_name != "unconfigured" else "lensing"

    return {
        "COMPOSE_PROJECT_NAME": slugify(compose_project),
        "LENSING_HUB_PORT": str(deploy.get("hub_port", 8080)),
        "LENSING_INFER_PORT": str(deploy.get("infer_port", 8090)),
        "LENSING_DB_PORT": str(deploy.get("db_port", 5433)),
    }


def upsert_env(values: dict[str, str]) -> None:
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = m.group(1) if m else None
        if key in values:
            out.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            out.append(line)
    appended = [k for k in OWNED if k not in seen]
    if appended:
        if out and out[-1].strip():
            out.append("")
        out.append("# Deploy identity/ports (from domain.toml [deploy]; render-deploy-env.py)")
        out.extend(f"{k}={values[k]}" for k in appended)
    ENV.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="upsert the keys into .env instead of printing")
    args = ap.parse_args()

    values = derive()
    if args.write:
        upsert_env(values)
        print(f"[render-deploy-env] wrote {', '.join(OWNED)} to {ENV.relative_to(ROOT)}", file=sys.stderr)
        for k in OWNED:
            print(f"  {k}={values[k]}", file=sys.stderr)
    else:
        for k in OWNED:
            print(f"{k}={values[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
