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