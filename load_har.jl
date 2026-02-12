import Pkg
Pkg.activate(@__DIR__)

using CSV, DataFrames

# -----------------------------
# load a single whitespace-delimited file -> Matrix{Float32}
# shape: (samples, timesteps)
# -----------------------------
function load_file(path::String)
    df = CSV.read(path, DataFrame; delim=' ', ignorerepeated=true, header=false)
    return Matrix{Float32}(df)
end

# -----------------------------
# load list of files into 3D array
# shape: (samples, timesteps, features)
# -----------------------------
function load_group(filenames::Vector{String}, dir::String)
    # Load first file to get sizes
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

# -----------------------------
# load one dataset group (train or test)
# returns:
#   X :: (samples, timesteps, features)
#   y :: Vector{Int} (class labels, 1..6)
# -----------------------------
function load_dataset_group(group::String, prefix::String)
    inertial_dir = joinpath(prefix, group, "Inertial Signals")

    filenames = String[]
    # total acceleration
    append!(filenames, [
        "total_acc_x_$(group).txt",
        "total_acc_y_$(group).txt",
        "total_acc_z_$(group).txt"
    ])
    # body acceleration
    append!(filenames, [
        "body_acc_x_$(group).txt",
        "body_acc_y_$(group).txt",
        "body_acc_z_$(group).txt"
    ])
    # body gyroscope
    append!(filenames, [
        "body_gyro_x_$(group).txt",
        "body_gyro_y_$(group).txt",
        "body_gyro_z_$(group).txt"
    ])

    X = load_group(filenames, inertial_dir)

    y_mat = load_file(joinpath(prefix, group, "y_$(group).txt"))  # (samples, 1)
    y = vec(Int.(y_mat[:, 1]))                                   # Vector{Int}

    return X, y
end

# -----------------------------
# one-hot encode labels
# input y: Vector{Int} with values 0..(K-1) OR 1..K (you choose)
# output: Matrix{Float32} shape (samples, K)
# -----------------------------
function onehot_labels(y::Vector{Int}, K::Int)
    n = length(y)
    Y = zeros(Float32, n, K)
    for i in 1:n
        Y[i, y[i] + 1] = 1.0f0   # +1 because Julia is 1-indexed
    end
    return Y
end

# -----------------------------
# load full dataset (like the article)
# expects folder: HARDataset/train and HARDataset/test
# -----------------------------
function load_dataset(base_dir::String = "HARDataset")
    trainX, trainy = load_dataset_group("train", base_dir)
    println("trainX: ", size(trainX), "  trainy: ", size(trainy))

    testX, testy = load_dataset_group("test", base_dir)
    println("testX: ", size(testX), "  testy: ", size(testy))

    # zero-offset class labels (1..6 -> 0..5)
    trainy0 = trainy .- 1
    testy0  = testy  .- 1

    K = maximum(trainy)  # should be 6 for UCI HAR
    trainY = onehot_labels(trainy0, K)
    testY  = onehot_labels(testy0, K)

    println("final shapes:")
    println("trainX: ", size(trainX), " trainY: ", size(trainY))
    println("testX:  ", size(testX),  " testY:  ", size(testY))

    return trainX, trainY, testX, testY
end

# run if this file is executed directly
trainX, trainY, testX, testY = load_dataset("HARDataset")