#!/usr/bin/env julia
# MLP target-value predictor on the language-neutral dataset artifact. Flux.jl, CPU.
# Mirrors crates/predictor-burn-mlp: same architecture (Dense+relu+Dropout per
# hidden layer, linear output), same hyperparams and defaults, same per-epoch
# SplitMix64 Fisher-Yates reshuffle, same scaler.json / metrics.json /
# predictions.json outputs. Implements contract v2: `train` and `predict`.

using Flux
using JLD2
using JSON3
using Random

# ---------------------------------------------------------------- contract IO

"Progress as JSON-lines on stdout (server streams line-by-line)."
function emit(obj)
    println(JSON3.write(obj))
    flush(stdout)
end

read_f32(path) = collect(reinterpret(Float32, read(path)))
read_u32(path) = collect(reinterpret(UInt32, read(path)))
read_u64(path) = collect(reinterpret(UInt64, read(path)))

struct Hyperparams
    epochs::Int
    lr::Float64
    batch_size::Int
    hidden::Vector{Int}
    dropout::Float64
    checkpoint_every::Int  # save a loadable checkpoint every N epochs (0 = end only)
end

"Flat JSON object; missing keys fall back to the registry defaults."
function load_hyperparams(path)
    hp = JSON3.read(read(path, String))
    Hyperparams(
        get(hp, :epochs, 50),
        get(hp, :lr, 1e-3),
        get(hp, :batch_size, 256),
        collect(Int, get(hp, :hidden, [256, 128])),
        get(hp, :dropout, 0.1),
        get(hp, :checkpoint_every, 0),
    )
end

"Invert manifest.target.transform back to target space."
invert(transform::AbstractString, y::Float64) = transform == "log1p" ? expm1(y) : y

# --------------------------------------------------------------------- scaler

"Per-column mean/std fit on the train split only (population std, f64 accum)."
function fit_scaler(X::Matrix{Float32})  # n_cols × n
    n_cols, n = size(X)
    mean = zeros(Float64, n_cols)
    for j in 1:n, i in 1:n_cols
        mean[i] += X[i, j]
    end
    mean ./= n
    var = zeros(Float64, n_cols)
    for j in 1:n, i in 1:n_cols
        d = X[i, j] - mean[i]
        var[i] += d * d
    end
    std = Float32[(s = sqrt(v / n); s < 1e-12 ? 1.0f0 : Float32(s)) for v in var]
    (mean = Float32.(mean), std = std)
end

standardize(scaler, X) = (X .- scaler.mean) ./ scaler.std

function load_scaler(path)
    s = JSON3.read(read(path, String))
    (mean = collect(Float32, s.mean), std = collect(Float32, s.std))
end

# ---------------------------------------------------------------------- model

function build_mlp(n_in::Int, hidden::Vector{Int}, dropout::Float64)
    layers = Any[]
    prev = n_in
    for h in hidden
        push!(layers, Dense(prev => h, relu))  # dropout after relu, as in burn
        push!(layers, Dropout(dropout))
        prev = h
    end
    push!(layers, Dense(prev => 1))  # output: no activation
    Chain(layers...)
end

"Gather 0-based row indices from the flat row-major matrix into n_cols × n."
function gather(features::Vector{Float32}, n_cols::Int, idx)
    X = Matrix{Float32}(undef, n_cols, length(idx))
    for (j, r) in enumerate(idx)
        base = Int(r) * n_cols
        @views X[:, j] .= features[base+1:base+n_cols]
    end
    X
end

"Same SplitMix64 stream as lensing-core, so the epoch shuffles match burn-mlp."
mutable struct SplitMix64
    s::UInt64
end

function next_u64!(r::SplitMix64)
    r.s += 0x9E3779B97F4A7C15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

"Predict transformed-space targets for all columns of X (caller sets mode)."
function predict_batched(model, X::Matrix{Float32}, batch_size::Int)
    n = size(X, 2)
    out = Vector{Float32}(undef, n)
    for batch in Iterators.partition(1:n, max(batch_size, 1))
        out[batch] = vec(model(X[:, batch]))
    end
    out
end

"Atomic checkpoint write: a kill mid-write can never corrupt model.jld2."
function save_checkpoint_atomic(run_dir, model)
    tmp = joinpath(run_dir, "model.jld2.tmp")
    jldsave(tmp; state = Flux.state(model))
    mv(tmp, joinpath(run_dir, "model.jld2"); force = true)
end

"Hand-written loop; losses are MSE in transformed (log) target space."
function run_training!(model, train_X, train_y, test_X, test_y, hp::Hyperparams, run_dir)
    n_train = length(train_y)
    opt = Flux.setup(Adam(hp.lr), model)
    order = collect(1:n_train)
    rng = SplitMix64(UInt64(1337) ⊻ 0xA5A5A5A5)

    for epoch in 1:hp.epochs
        # Graceful stop (contract v2): the server drops a STOP file into the
        # run dir; finish early and let the caller run the normal
        # end-of-training path (eval + metrics + final checkpoint).
        if isfile(joinpath(run_dir, "STOP"))
            emit((event = "stopping",))
            emit((event = "log", msg =
                "stop requested; evaluating with params as of epoch $(epoch - 1)"))
            break
        end

        # Fisher-Yates reshuffle each epoch.
        for i in n_train:-1:2
            j = Int(next_u64!(rng) % UInt64(i)) + 1
            order[i], order[j] = order[j], order[i]
        end

        trainmode!(model)
        loss_sum = 0.0
        for batch in Iterators.partition(order, hp.batch_size)
            x = train_X[:, batch]
            y = reshape(train_y[batch], 1, :)
            loss, grads = Flux.withgradient(m -> Flux.mse(m(x), y), model)
            Flux.update!(opt, model, grads[1])
            loss_sum += Float64(loss) * length(batch)
        end
        train_loss = loss_sum / n_train

        # Validation with dropout inactive.
        testmode!(model)
        val_pred = predict_batched(model, test_X, hp.batch_size)
        val_loss = sum(abs2, Float64.(val_pred) .- Float64.(test_y)) / length(test_y)

        emit((event = "epoch", epoch = epoch, total_epochs = hp.epochs,
              train_loss = train_loss, val_loss = val_loss))

        # Periodic checkpoint: after this event the run is promotable even if
        # the process is later killed. The final epoch is skipped — the
        # caller saves right after training anyway.
        if hp.checkpoint_every > 0 && epoch % hp.checkpoint_every == 0 && epoch < hp.epochs
            testmode!(model)
            save_checkpoint_atomic(run_dir, model)
            emit((event = "checkpoint", epoch = epoch))
        end
    end
    testmode!(model)
    model
end

# ------------------------------------------------------------------------ viz

"""
Architecture diagram as a self-contained SVG (`viz.svg` in the run dir),
mirroring crates/predictor-burn-mlp/src/viz.rs: input → hidden layers
(Dense + ReLU + dropout) → linear output, hand-built by string formatting.
"""
function write_viz(path, n_in::Int, hidden::Vector{Int}, dropout::Float64)
    box_w, box_h, gap, pad, box_y, height = 150, 86, 44, 24, 46, 168
    boxes = Tuple{String,String,String,String}[("input", string(n_in), "features", "")]
    prev = n_in
    for (i, h) in enumerate(hidden)
        push!(boxes, ("hidden $i", string(h), "Dense $prev→$h", "ReLU · dropout $dropout"))
        prev = h
    end
    push!(boxes, ("output", "1", "Dense $prev→1", "linear"))

    n = length(boxes)
    w = 2pad + n * box_w + (n - 1) * gap
    arch = join(hidden, " → ")

    io = IOBuffer()
    print(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $w $height" font-family="ui-monospace, SFMono-Regular, Menlo, monospace"><rect x="0" y="0" width="$w" height="$height" rx="6" fill="#fcfcfb"/><defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L8,4L0,8z" fill="#8a8a8a"/></marker></defs><text x="$pad" y="28" font-size="13" fill="#5a5a5a">MLP · $n_in → $arch → 1</text>""")
    for (i, (title, units, sub1, sub2)) in enumerate(boxes)
        x = pad + (i - 1) * (box_w + gap)
        cx = x + box_w ÷ 2
        print(io, """<rect x="$x" y="$box_y" width="$box_w" height="$box_h" rx="6" fill="#ffffff" stroke="#b9b6ae"/><text x="$cx" y="$(box_y + 18)" font-size="11" fill="#7a766c" text-anchor="middle">$title</text><text x="$cx" y="$(box_y + 44)" font-size="22" fill="#2b2a26" text-anchor="middle">$units</text>""")
        isempty(sub1) || print(io, """<text x="$cx" y="$(box_y + 62)" font-size="10" fill="#7a766c" text-anchor="middle">$sub1</text>""")
        isempty(sub2) || print(io, """<text x="$cx" y="$(box_y + 76)" font-size="10" fill="#7a766c" text-anchor="middle">$sub2</text>""")
        if i < n
            ym = box_y + box_h ÷ 2
            print(io, """<line x1="$(x + box_w + 4)" y1="$ym" x2="$(x + box_w + gap - 6)" y2="$ym" stroke="#8a8a8a" stroke-width="1.25" marker-end="url(#arr)"/>""")
        end
    end
    print(io, "</svg>")
    write(path, take!(io))
end

# -------------------------------------------------------------------- metrics

"Exact lensing_core::compute_metrics formulas, target space, mape/medape fractions."
function compute_metrics(pairs::Vector{Tuple{Float64,Float64}})
    n = length(pairs)
    abs_err = 0.0
    sq_err = 0.0
    ss_tot = 0.0
    apes = Float64[]
    mean_actual = sum(first, pairs) / n
    for (actual, predicted) in pairs
        e = predicted - actual
        abs_err += abs(e)
        sq_err += e * e
        ss_tot += (actual - mean_actual)^2
        push!(apes, abs(e / actual))  # actual never 0 (dataset filter)
    end
    sort!(apes)
    medape = isodd(n) ? apes[n ÷ 2 + 1] : (apes[n ÷ 2] + apes[n ÷ 2 + 1]) / 2
    (mae = abs_err / n, rmse = sqrt(sq_err / n), r2 = 1.0 - sq_err / ss_tot,
     mape = sum(apes) / n, medape = medape, n_test = n)
end

# ------------------------------------------------------------------- commands

function cmd_train(dataset_dir, run_dir, hp_path)
    hp = load_hyperparams(hp_path)
    hp.epochs >= 1 || error("epochs must be >= 1")
    hp.batch_size >= 1 || error("batch_size must be >= 1")
    !isempty(hp.hidden) || error("hidden must have at least one layer")
    mkpath(run_dir)

    manifest = JSON3.read(read(joinpath(dataset_dir, "manifest.json"), String))
    n_cols = Int(manifest.n_cols)
    features = read_f32(joinpath(dataset_dir, "features.f32"))
    target = read_f32(joinpath(dataset_dir, "target.f32"))
    row_ids = read_u64(joinpath(dataset_dir, "row_ids.u64"))
    train_idx = read_u32(joinpath(dataset_dir, "train_idx.u32"))
    test_idx = read_u32(joinpath(dataset_dir, "test_idx.u32"))
    emit((event = "log", msg = "dataset $(manifest.dataset_id): " *
        "$(length(train_idx)) train / $(length(test_idx)) test rows, $n_cols features"))
    emit((event = "log", msg = "mlp $(hp.hidden), dropout $(hp.dropout), " *
        "lr $(hp.lr), batch $(hp.batch_size), $(hp.epochs) epochs"))

    # Architecture diagram (registry capability `visualization`): written
    # before the first epoch so the live run view can show it immediately.
    write_viz(joinpath(run_dir, "viz.svg"), n_cols, hp.hidden, hp.dropout)

    train_X_raw = gather(features, n_cols, train_idx)
    test_X_raw = gather(features, n_cols, test_idx)
    train_y = Float32[target[Int(r)+1] for r in train_idx]
    test_y = Float32[target[Int(r)+1] for r in test_idx]

    # Standardize features with train-split statistics only.
    scaler = fit_scaler(train_X_raw)
    write(joinpath(run_dir, "scaler.json"), JSON3.write(scaler))
    train_X = standardize(scaler, train_X_raw)
    test_X = standardize(scaler, test_X_raw)

    Random.seed!(1337)
    model = build_mlp(n_cols, hp.hidden, hp.dropout)
    run_training!(model, train_X, train_y, test_X, test_y, hp, run_dir)

    # Predictions in target space.
    transform = String(manifest.target.transform)
    predicted_t = predict_batched(model, test_X, hp.batch_size)
    pairs = Tuple{Float64,Float64}[]
    predictions = []
    for (j, r) in enumerate(test_idx)
        actual = invert(transform, Float64(test_y[j]))
        predicted = max(invert(transform, Float64(predicted_t[j])), 0.0)
        push!(pairs, (actual, predicted))
        push!(predictions, (row_id = row_ids[Int(r)+1], actual = actual, predicted = predicted))
    end
    metrics = compute_metrics(pairs)
    write(joinpath(run_dir, "metrics.json"), JSON3.write(metrics))
    write(joinpath(run_dir, "predictions.json"), JSON3.write(predictions))
    save_checkpoint_atomic(run_dir, model)

    emit((event = "log", msg =
        "MAE $(round(Int, metrics.mae))  RMSE $(round(Int, metrics.rmse))  " *
        "medAPE $(round(metrics.medape * 100; digits=1))%  R² $(round(metrics.r2; digits=3))"))
    emit((event = "done",))
end

function cmd_predict(model_dir, input_dir, output_path)
    hp = load_hyperparams(joinpath(model_dir, "hyperparams.json"))

    manifest = JSON3.read(read(joinpath(input_dir, "manifest.json"), String))
    n_cols = Int(manifest.n_cols)
    features = read_f32(joinpath(input_dir, "features.f32"))
    row_ids = read_u64(joinpath(input_dir, "row_ids.u64"))
    emit((event = "log", msg =
        "predicting $(manifest.n_rows) rows, $n_cols features, mlp $(hp.hidden)"))

    scaler = load_scaler(joinpath(model_dir, "scaler.json"))
    length(scaler.mean) == n_cols ||
        error("scaler was fit on $(length(scaler.mean)) columns, input has $n_cols")
    X = standardize(scaler, gather(features, n_cols, 0:length(row_ids)-1))

    model = build_mlp(n_cols, hp.hidden, hp.dropout)
    Flux.loadmodel!(model, JLD2.load(joinpath(model_dir, "model.jld2"), "state"))
    testmode!(model)
    predicted_t = predict_batched(model, X, hp.batch_size)

    transform = String(manifest.target.transform)
    predictions = [(row_id = rid, predicted = max(invert(transform, Float64(p)), 0.0))
                   for (rid, p) in zip(row_ids, predicted_t)]
    write(output_path, JSON3.write(predictions))
    emit((event = "done",))
end

# ------------------------------------------------------------------------ cli

function parse_flags(args)
    flags = Dict{String,String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected argument $(args[i])")
        i < length(args) || error("missing value for $(args[i])")
        flags[args[i][3:end]] = args[i+1]
        i += 2
    end
    flags
end

function main()
    isempty(ARGS) && error("usage: train.jl <train|predict> --<flag> <value> ...")
    cmd = ARGS[1]
    flags = parse_flags(ARGS[2:end])
    if cmd == "train"
        cmd_train(flags["dataset"], flags["output"], flags["hyperparams"])
    elseif cmd == "predict"
        cmd_predict(flags["model"], flags["input"], flags["output"])
    else
        error("unknown command: $cmd")
    end
end

main()
