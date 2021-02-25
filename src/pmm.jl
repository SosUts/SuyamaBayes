mutable struct PriorParameters
    a::AbstractVector
    b::AbstractVector
    α::AbstractVector
    PriorParameters() = new()
    PriorParameters(a, b, α) = new(a, b, α)
end

mutable struct PoissionMixtureModel
    model::MixtureModel{Univariate,Discrete,Poisson,Float64}
    λs::AbstractMatrix
    ps::AbstractMatrix
    loglikelihoods::AbstractVector
    PoissionMixtureModel() = new()
end

function init_model(;K::Int = 3, a = 1.0, b = 0.01)
    α = init_α(K)
    λ = init_λ(a, b, K)
    p = init_p(α)
    prior = PriorParameters(ones(K) .* a, ones(K) .*b, α)
    prior, MixtureModel(Poisson, λ, p)
end

function generate_toy_data(
    N::Int = 10^4;
    true_λ::AbstractVector = [100.0, 50.0, 15.0],
    true_p::AbstractVector = [0.5, 0.2, 0.3],
)
    true_data = MixtureModel(Poisson, true_λ, true_p)
    toy_data = rand(true_data, N)
    true_data, toy_data
end

init_α(K) = rand(Uniform(0, 100), K)
init_λ(a, b, K) = rand(Gamma(a, 1 / b), K)
init_p(α) = rand(Dirichlet(α))

function sample_s(model, data)
    N = length(data)
    K = length(model.components)

    sₙ = Matrix{Int}(undef, N, K)
    η = Matrix{Float64}(undef, N, K)
    λ = [Distributions.params(comp)[1] for comp in model.components]
    p = probs(model)
    @inbounds for n = 1:N
        @simd for k = 1:K
            η[n, k] = data[n] * log(λ[k]) - λ[k] + log(p[k])
        end
        logsum_η = -logsumexp(η[n, :])
        @simd for k = 1:K
            η[n, k] = exp(η[n, k] + logsum_η)
        end
        sₙ[n, :] .= rand(Multinomial(1, η[n, :]))
    end
    sₙ
end

function sample_λ(model, prior, sₙ, toy_data)
    K = size(sₙ, 2)
    λ = Vector{Float64}(undef, K)
    @inbounds for k = 1:K
        â = sum(sₙ[:, k] .* toy_data) + prior.a[k]
        b̂ = sum(sₙ[:, k]) + prior.b[k]
        λ[k] = rand(Gamma(â, 1/b̂))
    end
    λ
end

function sample_p(model, prior, sₙ)
    N, K = size(sₙ)
    α̂ = Vector{Float64}(undef, K)
    @inbounds for k = 1:K
        α̂[k] = sum(sₙ[:, k]) + prior.α[k]
    end
    rand(Dirichlet(α̂))
end

function gibbs!(
    pmm::PoissionMixtureModel,
    prior::PriorParameters,
    toy_data::AbstractVector;
    maxiter::Int = 2000,
)
    model = pmm.model
    N = length(toy_data)
    K = length(model.components)

    pmm.λs = Matrix{Float64}(undef, maxiter, K)
    pmm.ps = Matrix{Float64}(undef, maxiter, K)
    pmm.loglikelihoods = Vector{Float64}(undef, maxiter)

    for k = 1:K
        pmm.λs[1, k] = model.components[k].λ
        pmm.ps[1, k] = probs(model)[k]
    end
    pmm.loglikelihoods[1] = sum(logpdf.(model, toy_data))

    @inbounds @simd for i = 1:maxiter
        sₙ = sample_s(model, toy_data)
        λ = sample_λ(model, prior, sₙ, toy_data)
        p = sample_p(model, prior, sₙ)

        pmm.λs[i, :] .= λ
        pmm.ps[i, :] .= p
        model = MixtureModel(Poisson, λ, p)
        pmm.loglikelihoods[i] = sum(logpdf.(model, toy_data))
        pmm.model = model
        fill!(sₙ, 0)
    end
    pmm
end

function update_s(prior, data)
    N = length(data)
    K = length(prior.α)

    η = Matrix{Float64}(undef, N, K)

    a = prior.a
    b = prior.b
    α = prior.α
    λ = a ./ b
    lnλ = digamma.(a) .- digamma.(b)
    lnp = [digamma(α[k]) + digamma(sum(α)) for k = 1:K]

    @inbounds for n = 1:N
        @simd for k = 1:K
            η[n, k] = data[n] * lnλ[k] - λ[k] + lnp[k]
        end
        logsum_η = -logsumexp(η[n, :])
        @simd for k = 1:K
            η[n, k] = exp(η[n, k] + logsum_η)
        end
    end
    η
end

function update_λ(prior, η, toy_data)
    K = size(η, 2)
    â = Vector{Float64}(undef, K)
    b̂ = Vector{Float64}(undef, K)
    a = prior.a
    b = prior.b
    @inbounds for k = 1:K
        â[k] = sum(η[:, k] .* toy_data) + a[k]
        b̂[k] = sum(η[:, k]) + b[k]
    end
    â, b̂
end

function update_p(prior, η)
    N, K = size(η)
    α̂ = Vector{Float64}(undef, K)
    α = prior.α
    @inbounds for k = 1:K
        α̂[k] = sum(η[:, k]) + α[k]
    end
    α̂
end

function vi!(prior::PriorParameters, toy_data::AbstractVector; maxiter::Int = 2000)
    N = length(toy_data)
    K = length(prior.α)

    #     elbo = Vector{Float64}(undef, N)
    #     elbo[1] = 
    @inbounds for i = 1:maxiter
        η = update_s(prior, toy_data)
        â, b̂ = update_λ(prior, η, toy_data)
        α̂ = update_p(prior, η)

        prior.a = â
        prior.b = b̂
        prior.α = α̂
    end
end