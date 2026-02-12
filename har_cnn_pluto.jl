### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 818526bf-69f6-4a89-976f-331c853e3f31
begin
    import Pkg
    Pkg.activate(@__DIR__)
    # Pkg.instantiate()  # uncomment on a new machine
end

# ╔═╡ 2a907f20-9239-4fed-b7f1-985c7ed92ad6
begin
    using CSV
    using DataFrames
    using Flux
    using Random
    using Statistics
    using StatsPlots
end

# ╔═╡ 3b0b8b8e-7f52-44cb-ad25-511e82d6e08f
DATA_DIR = "HARDataset"

# ╔═╡ 45c73ab0-1c88-4d78-ab4e-82b5011ccd55
begin
    function load_file(path::AbstractString)
        df = CSV.read(path, DataFrame; delim=' ', ignorerepeated=true, header=false)
        return Matrix{Float32}(df)
    end

    function load_group(filenames::Vector{String}, dir::AbstractString)
        first = load_file(joinpath(dir, filenames[1]))
        n_samples, n_steps = size(first)
        n_features = length(filenames)

        X = Array{Float32}(undef, n_samples, n_steps, n_features)
        X[:, :, 1] = first

        for (j, name) in enumerate(filenames[2:end])
            X[:, :, j + 1] = load_file(joinpath(dir, name))
        end
        return X
    end

    function load_dataset_group(group::AbstractString, prefix::AbstractString)
        inertial_dir = joinpath(prefix, group, "Inertial Signals")

        filenames = String[]
        append!(filenames, [
            "total_acc_x_$(group).txt", "total_acc_y_$(group).txt", "total_acc_z_$(group).txt",
            "body_acc_x_$(group).txt",  "body_acc_y_$(group).txt",  "body_acc_z_$(group).txt",
            "body_gyro_x_$(group).txt", "body_gyro_y_$(group).txt", "body_gyro_z_$(group).txt"
        ])

        X = load_group(filenames, inertial_dir)

        y_mat = load_file(joinpath(prefix, group, "y_$(group).txt"))
        y = vec(Int.(y_mat[:, 1]))
        return X, y
    end

    function onehot_labels(y::Vector{Int}, K::Int)
        Y = zeros(Float32, length(y), K)
        for i in eachindex(y)
            Y[i, y[i] + 1] = 1.0f0
        end
        return Y
    end

    function load_dataset(base_dir::AbstractString)
        trainX, trainy = load_dataset_group("train", base_dir)
        println("trainX: ", size(trainX), "  trainy: ", size(trainy))

        testX, testy = load_dataset_group("test", base_dir)
        println("testX:  ", size(testX),  "  testy:  ", size(testy))

        trainy0 = trainy .- 1
        testy0  = testy  .- 1

        K = maximum(trainy)
        trainY = onehot_labels(trainy0, K)
        testY  = onehot_labels(testy0,  K)

        println("final shapes:")
        println("trainX: ", size(trainX), " trainY: ", size(trainY))
        println("testX:  ", size(testX),  " testY:  ", size(testY))

        return trainX, trainY, testX, testY
    end
end

# ╔═╡ 6408e488-785a-4bc2-a9fb-118ec47fe479
trainX, trainY, testX, testY = load_dataset(DATA_DIR)

# ╔═╡ fa670c76-59e3-4d25-9373-aa670dfbd998
begin
    n_samples, n_timesteps, n_features = size(trainX)
    n_outputs = size(trainY, 2)
    (n_timesteps, n_features, n_outputs)
end

# ╔═╡ eb2006ca-8cab-488e-ae76-8b1d8fb6060d
to_flux(X) = permutedims(X, (2, 3, 1))  # (N,T,C) -> (T,C,N)

# ╔═╡ c11ab399-eaec-42b0-97f9-47b37b919227
begin
    function build_model(n_timesteps, n_features, n_outputs)
        fe = Chain(
            Conv((3,), n_features => 64, relu),
            Conv((3,), 64 => 64, relu),
            Dropout(0.5),
            MaxPool((2,)),
            Flux.flatten
        )
        flat_size = Flux.outputsize(fe, (n_timesteps, n_features, 1))[1]
        return Chain(
            fe,
            Dense(flat_size, 100, relu),
            Dense(100, n_outputs),
            softmax
        )
    end
end

# ╔═╡ 6eea02e7-29a3-4dea-848d-ff47bc725e9b
begin
    function evaluate_model(trainX, trainY, testX, testY; epochs=10, batch_size=32, seed=0)
        Random.seed!(seed)

        Xtr = to_flux(trainX)
        Xte = to_flux(testX)

        Ytr = permutedims(trainY, (2, 1))
        Yte = permutedims(testY,  (2, 1))

        _, n_timesteps, n_features = size(trainX)
        n_outputs = size(trainY, 2)

        model = build_model(n_timesteps, n_features, n_outputs)

        opt = Flux.Adam()
        opt_state = Flux.setup(opt, model)

        accuracy(X, Y) = mean(Flux.onecold(model(X)) .== Flux.onecold(Y))

        for _ in 1:epochs
            idx = Random.randperm(size(Xtr, 3))
            for start in 1:batch_size:length(idx)
                batch_idx = idx[start:min(start + batch_size - 1, end)]
                x = Xtr[:, :, batch_idx]
                y = Ytr[:, batch_idx]

                gs = Flux.gradient(model) do m
                    Flux.crossentropy(m(x), y)
                end
                Flux.update!(opt_state, model, gs[1])
            end
        end

        return accuracy(Xte, Yte)
    end
end

# ╔═╡ f337bdf6-301f-4346-9881-e373823156b7
begin
    function summarize_results(scores)
        println(scores)
        m, s = mean(scores), std(scores)
        println("Accuracy: $(round(m, digits=3))% (+/-$(round(s, digits=3)))")
        return (mean=m, std=s)
    end
end

# ╔═╡ 193f08e0-b8e9-4387-a24f-263692dc79b5
begin
    function run_experiment(; repeats=10, epochs=10, batch_size=32)
        trainX, trainY, testX, testY = load_dataset(DATA_DIR)

        scores = Float64[]
        for r in 1:repeats
            score = evaluate_model(trainX, trainY, testX, testY; epochs=epochs, batch_size=batch_size, seed=r) * 100
            println(">#$(r): $(round(score, digits=3))")
            push!(scores, score)
        end

        return summarize_results(scores)
    end
end

# ╔═╡ 85e43805-1289-4f17-9b77-8db46cfc40f0


# ╔═╡ 6a6b43f6-2e8a-4a8b-9d7a-b0c6c9c0a6c1
begin
    # remove 50% overlap (keep last half of each window), then flatten across time & windows
    function long_format(X::Array{Float32,3})
        N, T, C = size(X)
        cut = Int(T ÷ 2)
        X2 = X[:, (T - cut + 1):T, :]          # (N, cut, C)
        return reshape(X2, N * cut, C)         # (N*cut, C)
    end

    # flatten full windows (all timesteps) into (N*T, C)
    function flat_format(X::Array{Float32,3})
        N, T, C = size(X)
        return reshape(X, N * T, C)
    end
end

# ╔═╡ 9c3e2b0c-9a2b-4a67-8f8c-2d6d8a9a8d5c
begin
    function plot_variable_distributions(trainX::Array{Float32,3}; bins=100)
        X = long_format(trainX)  # (N*cut, 9)

        names = [
            "total_acc_x","total_acc_y","total_acc_z",
            "body_acc_x","body_acc_y","body_acc_z",
            "body_gyro_x","body_gyro_y","body_gyro_z"
        ]

        p = plot(layout=(size(X,2), 1), legend=false)
        for i in 1:size(X,2)
            histogram!(p[i], X[:, i]; bins=bins, xlims=(-1, 1), title=names[i])
        end
        p
    end
end

# ╔═╡ 2d8c6f83-1ef0-41fb-9b55-7d4d8d9a2b11
begin
    function scale_data(trainX::Array{Float32,3}, testX::Array{Float32,3}, standardize::Bool)
        if !standardize
            return trainX, testX
        end

        # fit scaler on de-overlapped long training data
        longX = long_format(trainX)    # (N*cut, C)
        μ = vec(mean(longX, dims=1))   # (C,)
        σ = vec(std(longX, dims=1)) .+ 1f-6

        # apply to flattened full train/test, then reshape back
        flatTrain = flat_format(trainX)
        flatTest  = flat_format(testX)

        flatTrain = (flatTrain .- μ') ./ σ'
        flatTest  = (flatTest  .- μ') ./ σ'

        Ntr, Ttr, Ctr = size(trainX)
        Nte, Tte, Cte = size(testX)

        trainX2 = reshape(flatTrain, Ntr, Ttr, Ctr)
        testX2  = reshape(flatTest,  Nte, Tte, Cte)

        return trainX2, testX2
    end
end

# ╔═╡ 6e1f3c3e-3f2e-4f1c-9e6a-3e9b1b8b6c4a
begin
    function evaluate_model_param(trainX, trainY, testX, testY, standardize::Bool; epochs=10, batch_size=32, seed=0)
        Random.seed!(seed)

        # optional standardization
        trainX2, testX2 = scale_data(trainX, testX, standardize)

        Xtr = to_flux(trainX2)
        Xte = to_flux(testX2)

        Ytr = permutedims(trainY, (2, 1))
        Yte = permutedims(testY,  (2, 1))

        _, n_timesteps, n_features = size(trainX2)
        n_outputs = size(trainY, 2)

        model = build_model(n_timesteps, n_features, n_outputs)

        opt = Flux.Adam()
        opt_state = Flux.setup(opt, model)

        accuracy(X, Y) = mean(Flux.onecold(model(X)) .== Flux.onecold(Y))

        for _ in 1:epochs
            idx = Random.randperm(size(Xtr, 3))
            for start in 1:batch_size:length(idx)
                batch_idx = idx[start:min(start + batch_size - 1, end)]
                x = Xtr[:, :, batch_idx]
                y = Ytr[:, batch_idx]

                gs = Flux.gradient(model) do m
                    Flux.crossentropy(m(x), y)
                end
                Flux.update!(opt_state, model, gs[1])
            end
        end

        return accuracy(Xte, Yte)
    end
end

# ╔═╡ 0c7f7c1a-2b9b-4c9b-9e8d-1d2e3f4a5b6c
begin
    function summarize_results_params(all_scores, params; outfile="exp_cnn_standardize.png")
        # print mean/std per param
        for (i, p) in enumerate(params)
            m, s = mean(all_scores[i]), std(all_scores[i])
            println("Param=$(p): $(round(m, digits=3))% (+/-$(round(s, digits=3)))")
        end

        # boxplot
        labels = string.(params)
        bp = StatsPlots.boxplot(labels, all_scores; legend=false)
        savefig(bp, outfile)
        return bp
    end
end

# ╔═╡ 4b1a2c3d-5e6f-4a7b-8c9d-0e1f2a3b4c5d
begin
    function run_experiment_params(params; repeats=10, epochs=10, batch_size=32)
        trainX, trainY, testX, testY = load_dataset(DATA_DIR)

        all_scores = Vector{Vector{Float64}}()
        for p in params
            scores = Float64[]
            for r in 1:repeats
                score = evaluate_model_param(trainX, trainY, testX, testY, p; epochs=epochs, batch_size=batch_size, seed=r) * 100
                println(">p=$(p) #$(r): $(round(score, digits=3))")
                push!(scores, score)
            end
            push!(all_scores, scores)
        end

        summarize_results_params(all_scores, params)
    end
end

# ╔═╡ a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5e
run_experiment_params([false, true]; repeats=10, epochs=10, batch_size=32)

# ╔═╡ Cell order:
# ╠═818526bf-69f6-4a89-976f-331c853e3f31
# ╠═2a907f20-9239-4fed-b7f1-985c7ed92ad6
# ╠═3b0b8b8e-7f52-44cb-ad25-511e82d6e08f
# ╠═45c73ab0-1c88-4d78-ab4e-82b5011ccd55
# ╠═6408e488-785a-4bc2-a9fb-118ec47fe479
# ╠═fa670c76-59e3-4d25-9373-aa670dfbd998
# ╠═eb2006ca-8cab-488e-ae76-8b1d8fb6060d
# ╠═c11ab399-eaec-42b0-97f9-47b37b919227
# ╠═6eea02e7-29a3-4dea-848d-ff47bc725e9b
# ╠═f337bdf6-301f-4346-9881-e373823156b7
# ╠═193f08e0-b8e9-4387-a24f-263692dc79b5
# ╠═85e43805-1289-4f17-9b77-8db46cfc40f0
# ╠═6a6b43f6-2e8a-4a8b-9d7a-b0c6c9c0a6c1
# ╠═9c3e2b0c-9a2b-4a67-8f8c-2d6d8a9a8d5c
# ╠═2d8c6f83-1ef0-41fb-9b55-7d4d8d9a2b11
# ╠═6e1f3c3e-3f2e-4f1c-9e6a-3e9b1b8b6c4a
# ╠═0c7f7c1a-2b9b-4c9b-9e8d-1d2e3f4a5b6c
# ╠═4b1a2c3d-5e6f-4a7b-8c9d-0e1f2a3b4c5d
# ╠═a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5e
