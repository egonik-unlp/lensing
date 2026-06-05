//! Project runner for price-guesser-models, written against Zig 0.15.
//!
//! This repo is polyglot: a Rust workspace (pipeline, server, burn predictor),
//! a Vite + React UI, a Python predictor and a Julia predictor. `zig build`
//! is the one entry point that knows how to drive all of them; each tool
//! still owns its own incremental caching (cargo, npm, vite), so steps are
//! cheap to re-run.
//!
//! List every step:           zig build -l
//! Build everything:          zig build
//! Build + run the server:    zig build serve
//! UI dev loop (hot reload):  zig build dev          (server must be up)
//! All tests + lints:         zig build check
//!
//! Options (pass as -Dname=value):
//!   -Dport=8080                      lensing-server port for `serve`
//!   -Dqdrant-url=http://host:6333    Qdrant base URL for `serve` / `dataset`
//!   -Dcollection=properties-tagged   Qdrant collection for `serve` / `dataset`
//!   -Ddatabase-url=postgres://...    metadata database for `serve` / `migrate-data`
//!
//! Step graph (A → B means A runs B first):
//!   (default)     → backend, ui
//!   serve         → backend, ui, db-up, then runs target/release/lensing-server
//!   dataset       → backend, then runs target/release/lensing-pipeline build
//!   db-up         → docker compose up -d postgres --wait
//!   db-down       → docker compose down (the pgdata volume survives)
//!   migrate-data  → backend, db-up, then lensing-server migrate-data (idempotent
//!                   file→postgres backfill + consistency report)
//!   check         → test, lint, py-check
//!   dev, julia-setup, py-setup, test, lint, py-check are independent leaves.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const port = b.option(u16, "port", "lensing-server port (default 8080)") orelse 8080;
    const qdrant_url = b.option(
        []const u8,
        "qdrant-url",
        "Qdrant base URL (default http://localhost:6333)",
    ) orelse "http://localhost:6333";
    const collection = b.option(
        []const u8,
        "collection",
        "Qdrant collection (default properties-tagged)",
    ) orelse "properties-tagged";
    const database_url = b.option(
        []const u8,
        "database-url",
        "Postgres metadata database (default postgres://pg:pg@localhost:5433/lensing)",
    ) orelse "postgres://pg:pg@localhost:5433/lensing";

    // ---------------- backend: the Rust workspace ----------------
    // Release profile on purpose: training and PCA are CPU-bound, and
    // registry.toml points at target/release/ binaries.
    const cargo_build = b.addSystemCommand(&.{ "cargo", "build", "--release", "--workspace" });
    cargo_build.setCwd(b.path("."));
    cargo_build.has_side_effects = true; // cargo does its own caching

    const backend = b.step("backend", "Build the Rust workspace (release: server, pipeline, burn predictor)");
    backend.dependOn(&cargo_build.step);

    // ---------------- ui: install deps, build ui/dist ----------------
    const npm_install = b.addSystemCommand(&.{ "npm", "install", "--no-fund", "--no-audit" });
    npm_install.setCwd(b.path("ui"));
    npm_install.has_side_effects = true;

    const npm_build = b.addSystemCommand(&.{ "npm", "run", "build" });
    npm_build.setCwd(b.path("ui"));
    npm_build.has_side_effects = true;
    npm_build.step.dependOn(&npm_install.step);

    const ui = b.step("ui", "Install UI dependencies and build ui/dist (what lensing-server serves)");
    ui.dependOn(&npm_build.step);

    // ---------------- default: `zig build` builds everything ----------------
    b.getInstallStep().dependOn(backend);
    b.getInstallStep().dependOn(ui);

    // ---------------- db-up / db-down: the metadata database ----------------
    // docker compose owns the Postgres lifecycle; --wait blocks on the
    // healthcheck so dependents always see an accepting socket.
    const compose_up = b.addSystemCommand(&.{
        "docker", "compose", "up", "-d", "postgres", "--wait",
    });
    compose_up.setCwd(b.path("."));
    compose_up.stdio = .inherit;
    compose_up.has_side_effects = true;

    const db_up = b.step("db-up", "Start the Postgres metadata database (docker compose, waits for healthy)");
    db_up.dependOn(&compose_up.step);

    const compose_down = b.addSystemCommand(&.{ "docker", "compose", "down" });
    compose_down.setCwd(b.path("."));
    compose_down.stdio = .inherit;
    compose_down.has_side_effects = true;

    const db_down = b.step("db-down", "Stop the docker compose services (the pgdata volume survives)");
    db_down.dependOn(&compose_down.step);

    // ---------------- migrate-data: file -> postgres backfill ----------------
    // Idempotent: seeds/heals the database from models.toml + data/ and
    // prints a consistency report. Never touches the files.
    const run_migrate = b.addSystemCommand(&.{"target/release/lensing-server"});
    run_migrate.addArgs(&.{ "--database-url", database_url });
    run_migrate.addArg("migrate-data");
    run_migrate.setCwd(b.path("."));
    run_migrate.stdio = .inherit;
    run_migrate.has_side_effects = true;
    run_migrate.step.dependOn(backend);
    run_migrate.step.dependOn(db_up);

    const migrate_data = b.step("migrate-data", "Backfill file-based state into Postgres and print a consistency report");
    migrate_data.dependOn(&run_migrate.step);

    // ---------------- serve: build everything, run lensing-server ----------------
    const run_server = b.addSystemCommand(&.{"target/release/lensing-server"});
    run_server.addArgs(&.{ "--port", b.fmt("{d}", .{port}) });
    run_server.addArgs(&.{ "--qdrant-url", qdrant_url });
    run_server.addArgs(&.{ "--collection", collection });
    run_server.addArgs(&.{ "--database-url", database_url });
    run_server.setCwd(b.path(".")); // repo root: registry.toml, data/, ui/dist
    run_server.stdio = .inherit; // long-running; stream logs, allow ctrl-c
    run_server.step.dependOn(backend);
    run_server.step.dependOn(ui);
    run_server.step.dependOn(db_up);

    const serve = b.step("serve", "Build backend + UI, start the database, then run lensing-server (http://localhost:<port>)");
    serve.dependOn(&run_server.step);

    // ---------------- worker / infer: distributed roles (docs/DEPLOY.md) ----------------
    const hub_url = b.option(
        []const u8,
        "hub-url",
        "Hub base URL for `worker` dataset downloads (default http://localhost:8080)",
    ) orelse "http://localhost:8080";

    const run_worker = b.addSystemCommand(&.{ "target/release/lensing-server", "worker" });
    run_worker.addArgs(&.{ "--hub-url", hub_url });
    run_worker.addArgs(&.{ "--database-url", database_url });
    run_worker.setCwd(b.path("."));
    run_worker.stdio = .inherit;
    run_worker.step.dependOn(backend);
    run_worker.step.dependOn(db_up);

    const worker = b.step("worker", "Run a training worker (claims queued runs; needs the hub for datasets)");
    worker.dependOn(&run_worker.step);

    const run_infer = b.addSystemCommand(&.{ "target/release/lensing-server", "infer" });
    run_infer.addArgs(&.{ "--database-url", database_url });
    run_infer.setCwd(b.path("."));
    run_infer.stdio = .inherit;
    run_infer.step.dependOn(backend);
    run_infer.step.dependOn(db_up);

    const infer = b.step("infer", "Run a predict-only inference node (models materialized from Postgres, port 8090)");
    infer.dependOn(&run_infer.step);

    // ---------------- dev: Vite dev server with hot reload ----------------
    // Vite proxies /api to :8080, so run `zig build serve` in another
    // terminal first (or any lensing-server instance).
    const vite_dev = b.addSystemCommand(&.{ "npm", "run", "dev" });
    vite_dev.setCwd(b.path("ui"));
    vite_dev.stdio = .inherit;
    vite_dev.step.dependOn(&npm_install.step);

    const dev = b.step("dev", "Run the Vite dev server (hot reload; needs a running lensing-server for /api)");
    dev.dependOn(&vite_dev.step);

    // ---------------- dataset: build an artifact from Qdrant ----------------
    // Defaults mirror the pipeline CLI; for non-default flags (PCA dims,
    // quality filters, …) call target/release/lensing-pipeline build directly.
    const run_pipeline = b.addSystemCommand(&.{ "target/release/lensing-pipeline", "build" });
    run_pipeline.addArgs(&.{ "--qdrant-url", qdrant_url });
    run_pipeline.addArgs(&.{ "--collection", collection });
    run_pipeline.setCwd(b.path("."));
    run_pipeline.stdio = .inherit; // progress lines on stderr
    run_pipeline.step.dependOn(backend);

    const dataset = b.step("dataset", "Build a dataset artifact from Qdrant with default flags");
    dataset.dependOn(&run_pipeline.step);

    // ---------------- test / lint / check ----------------
    const cargo_test = b.addSystemCommand(&.{ "cargo", "test", "--workspace" });
    cargo_test.setCwd(b.path("."));
    cargo_test.has_side_effects = true;

    const tst = b.step("test", "Run the Rust test suite (featurizer round-trip, quality rules, meta compat)");
    tst.dependOn(&cargo_test.step);

    const tsc = b.addSystemCommand(&.{ "npx", "tsc", "-b" });
    tsc.setCwd(b.path("ui"));
    tsc.has_side_effects = true;
    tsc.step.dependOn(&npm_install.step);

    const eslint = b.addSystemCommand(&.{ "npm", "run", "lint" });
    eslint.setCwd(b.path("ui"));
    eslint.has_side_effects = true;
    eslint.step.dependOn(&npm_install.step);

    const lint = b.step("lint", "Typecheck (tsc -b) and lint (eslint) the UI");
    lint.dependOn(&tsc.step);
    lint.dependOn(&eslint.step);

    const py_compile = b.addSystemCommand(&.{
        "python3",                                "-m",
        "py_compile",                             "predictors/baseline-median/predict.py",
        "predictors/ridge/ridge.py",              "predictors/torch-cnn/train.py",
        "predictors/xgboost/train.py",            "predictors/lightgbm/train.py",
        "predictors/random-forest/train.py",
    });
    py_compile.setCwd(b.path("."));
    py_compile.has_side_effects = true;

    const py_check = b.step("py-check", "Syntax-check all Python predictors");
    py_check.dependOn(&py_compile.step);

    // ---------------- render-agents: skills/agents from domain.toml ----------------
    // agents-src/ is the single source for the agent/skill files under
    // .claude/, .agents/ and .gemini/ (previously hand-triplicated). The
    // rendered outputs carry a GENERATED marker — edit the templates.
    const render_agents_cmd = b.addSystemCommand(&.{ "python3", "agents-src/render.py" });
    render_agents_cmd.setCwd(b.path("."));
    render_agents_cmd.stdio = .inherit;
    render_agents_cmd.has_side_effects = true;

    const render_agents = b.step("render-agents", "Render .claude/.agents/.gemini skills+agents from agents-src/ templates + domain.toml");
    render_agents.dependOn(&render_agents_cmd.step);

    const render_check = b.addSystemCommand(&.{ "python3", "agents-src/render.py", "--check" });
    render_check.setCwd(b.path("."));
    render_check.has_side_effects = true;

    const check = b.step("check", "Everything CI would run: test + lint + py-check + agent-template drift");
    check.dependOn(tst);
    check.dependOn(lint);
    check.dependOn(py_check);
    check.dependOn(&render_check.step);

    // ---------------- package: distributable template ----------------
    // Copies the framework into dist/<name>-template (+ tarball) with the
    // instance files seeded fresh (empty models.toml, PROJECT-FACTS
    // skeleton, docs skeletons). data/ and the experiment history stay here.
    const package_cmd = b.addSystemCommand(&.{ "python3", "tools/package.py" });
    package_cmd.setCwd(b.path("."));
    package_cmd.stdio = .inherit;
    package_cmd.has_side_effects = true;
    package_cmd.step.dependOn(render_agents);

    const package = b.step("package", "Package the framework as a template under dist/ (the distributable 'library')");
    package.dependOn(&package_cmd.step);

    // ---------------- julia-setup: one-time Flux.jl environments ----------------
    const julia_inst = b.addSystemCommand(&.{
        "julia", "--project=predictors/flux-mlp", "-e", "using Pkg; Pkg.instantiate()",
    });
    julia_inst.setCwd(b.path("."));
    julia_inst.stdio = .inherit; // package install can be slow; show progress
    julia_inst.has_side_effects = true;

    const julia_inst_cnn = b.addSystemCommand(&.{
        "julia", "--project=predictors/flux-cnn", "-e", "using Pkg; Pkg.instantiate()",
    });
    julia_inst_cnn.setCwd(b.path("."));
    julia_inst_cnn.stdio = .inherit;
    julia_inst_cnn.has_side_effects = true;

    const julia_setup = b.step("julia-setup", "Install the flux-mlp + flux-cnn predictors' Julia packages (one-time)");
    julia_setup.dependOn(&julia_inst.step);
    julia_setup.dependOn(&julia_inst_cnn.step);

    // ---------------- py-setup: one-time Python venv for the Python predictors ----------------
    // baseline-median stays on system python3 (stdlib only); this venv serves
    // the predictors with real dependencies. The PyTorch CPU index keeps the
    // install slim and deterministic on GPU boxes too.
    const venv_create = b.addSystemCommand(&.{ "python3", "-m", "venv", "predictors/.venv" });
    venv_create.setCwd(b.path("."));
    venv_create.stdio = .inherit;
    venv_create.has_side_effects = true;

    const pip_install = b.addSystemCommand(&.{
        "predictors/.venv/bin/pip",                "install",
        "-r",                                      "predictors/requirements.txt",
        "--index-url",                             "https://download.pytorch.org/whl/cpu",
        "--extra-index-url",                       "https://pypi.org/simple",
    });
    pip_install.setCwd(b.path("."));
    pip_install.stdio = .inherit; // wheel download can be slow; show progress
    pip_install.has_side_effects = true;
    pip_install.step.dependOn(&venv_create.step);

    const py_setup = b.step("py-setup", "Create predictors/.venv and install the Python predictor deps (one-time)");
    py_setup.dependOn(&pip_install.step);
}
