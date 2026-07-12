# syntax=docker/dockerfile:1
#
# Lensing framework — containerized runtime for the "one binary, three roles"
# topology (hub / worker / infer; see docs/DEPLOY.md).
#
# The BUILD happens in disposable `*-builder` stages; the runtime images carry
# ONLY the compiled binary + the toolchains it needs to *run* — no cargo, no
# Zig, no Node. This mirrors `zig build backend` / `ui` / `py-setup` without
# making Zig a runtime dependency.
#
# Two runtime targets:
#   --target runtime   hub + worker: server + pipeline + predictor zoo + venv + ui/dist
#   --target infer     lean predict-only node (Rust/burn families; no ui, no venv
#                       by default — see docs/DEPLOY.md §Inference)
#
# Base images are ARGs so an instance can pin/override them.

ARG RUST_IMAGE=rust:1-bookworm
ARG NODE_IMAGE=node:22-bookworm-slim
# Runtime + venv share ONE python base so the venv is ABI-compatible with the
# interpreter that runs it in the final image.
ARG PY_IMAGE=python:3.12-slim-bookworm

# --------------------------------------------------------------------------
# rust-builder — cargo build --release --workspace (== `zig build backend`)
# --------------------------------------------------------------------------
FROM ${RUST_IMAGE} AS rust-builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# lensing-core embeds the repo-root domain.toml at compile time
# (include_str!("$CARGO_MANIFEST_DIR/../../domain.toml")).
COPY domain.toml ./domain.toml
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --workspace \
 && mkdir -p /out \
 && cp target/release/lensing-server \
       target/release/lensing-pipeline \
       target/release/predictor-burn-mlp \
       target/release/predictor-burn-cnn \
       target/release/predictor-blend \
       /out/
# (the cache mount is dropped at layer end, so the binaries are copied to /out
#  which is a real layer.)

# --------------------------------------------------------------------------
# ui-builder — npm install + npm run build (== `zig build ui`) -> ui/dist
# --------------------------------------------------------------------------
FROM ${NODE_IMAGE} AS ui-builder
WORKDIR /src/ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm install --no-fund --no-audit
COPY ui/ ./
RUN npm run build

# --------------------------------------------------------------------------
# venv-builder — predictors/.venv with the CPU torch wheel (== `zig build py-setup`)
# --------------------------------------------------------------------------
FROM ${PY_IMAGE} AS venv-builder
WORKDIR /src
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*
COPY predictors/requirements.txt ./predictors/requirements.txt
RUN python -m venv predictors/.venv \
 && predictors/.venv/bin/pip install --no-cache-dir --upgrade pip \
 && predictors/.venv/bin/pip install --no-cache-dir \
      -r predictors/requirements.txt \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple

# --------------------------------------------------------------------------
# runtime — hub + worker (full toolchain)
# --------------------------------------------------------------------------
FROM ${PY_IMAGE} AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Config + Python predictor sources (train.py/predict.py). registry.toml points
# at target/release/* and predictors/.venv/bin/python, resolved from --root.
COPY registry.toml domain.toml models.toml VERSION ./
COPY predictors ./predictors

# Built artifacts from the builder stages.
COPY --from=rust-builder /out/ ./target/release/
COPY --from=venv-builder /src/predictors/.venv ./predictors/.venv
COPY --from=ui-builder /src/ui/dist ./ui/dist

# Role entrypoint.
COPY docker/entrypoint.sh /usr/local/bin/lensing-entrypoint
RUN chmod +x /usr/local/bin/lensing-entrypoint \
 && mkdir -p /app/data

# data/ is a mount point for the artifact volume (features.f32, weights, run
# outputs, dataset cache) — declared so it persists even without an explicit
# compose volume.
VOLUME ["/app/data"]

ENV LENSING_ROLE=hub \
    LENSING_ROOT=/app
EXPOSE 8080 8090
ENTRYPOINT ["lensing-entrypoint"]

# --------------------------------------------------------------------------
# infer — lean predict-only node. Rust/burn families out of the box; to serve
# Python-family models, build with --target runtime instead (it carries the
# venv). No UI, no pipeline, no build toolchain -> smaller than `runtime`.
# --------------------------------------------------------------------------
FROM ${PY_IMAGE} AS infer
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY registry.toml domain.toml ./
COPY --from=rust-builder /out/lensing-server ./target/release/lensing-server
COPY --from=rust-builder /out/predictor-burn-mlp /out/predictor-burn-cnn /out/predictor-blend ./target/release/

COPY docker/entrypoint.sh /usr/local/bin/lensing-entrypoint
RUN chmod +x /usr/local/bin/lensing-entrypoint \
 && mkdir -p /app/data

VOLUME ["/app/data"]

ENV LENSING_ROLE=infer \
    LENSING_ROOT=/app
EXPOSE 8090
ENTRYPOINT ["lensing-entrypoint"]
