const PMM = MixtureModel{Univariate, Discrete, Poisson}

struct BayesianPMM
    α
    prior::Vector{Gamma}
    BayesianPMM(α, prior) = new(α, prior)
end

function rand(model::BayesianPMM)
    K = length(model.prior)
    a = rand(Dirichlet(K, model.α))
    λ = rand.(model.prior)
    MixtureModel(Poisson, λ, a)
end

function one_hot(model::PMM, N::Int)
    K = length(model.components)
    a = probs(model) # mixture weight

    # one hot vector of (K, N)
    onehotbatch(rand(Categorical(a), N), collect(1:K))
end

function generate_toy_data(model::PMM, N::Int)
    D = length(model)

    data = Array{eltype(model.components[1])}(undef, N, D)
    # convert to vector if D == 1
    D == 1 ? data = dropdims(data, dims = 2) : data
    S = one_hot(model, N)
    @inbounds for n = 1:N
        s = argmax(S[:, n])
        data[n, :] .= rand(model.components[s])
    end
    data, S
end

function init_S(data, prior_model::BayesianPMM)
    N = size(data, 1)
    K = length(prior_model.prior)
    onehotbatch(rand(Categorical(ones(K)./K), N), collect(1:K))
end

function add_stats(prior_model::BayesianPMM, data, S)
    @argcheck length(prior_model.prior) == size(S, 1)
    D = length(prior_model.prior[1])
    K = size(S, 1)
    sum_S = sum(S, dims=2)
    α = [prior_model.α + sum_S[k] for k = 1:K]
    XS = data'*S'

    a = Vector{Float64}(undef, K)
    b = Vector{Float64}(undef, K)
    for k = 1:K
        a[k] = prior_model.prior[k].α + XS[d, k] for d = 1:D
        b[k] = prior_model.prior[k].θ + sum_S[k]
    end
    BayesianPMM(α, Gamma.(a, b))
end