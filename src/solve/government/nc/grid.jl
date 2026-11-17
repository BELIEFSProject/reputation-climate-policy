## Normalised state space
struct NonCommittedMarkovGrid{T}
    h::T
    φ::AffineGridMap{T}
    A::AffineGridMap{T}
end

function NonCommittedMarkovGrid(model::AbstractModel, h; alower = model.firm.a₀)
    N = round(Int, inv(h))
    T = promote_type(typeof(float(h)), typeof(alower), typeof(model.firm.η))
    return NonCommittedMarkovGrid(one(T) / N, AffineGridMap(zero(T), one(T)), AffineGridMap(T(alower), T(model.firm.η)))
end

Base.size(grid::NonCommittedMarkovGrid) = (round(Int, inv(grid.h)) + 1, round(Int, inv(grid.h)) + 1)
Base.size(grid::NonCommittedMarkovGrid, d::Integer) = size(grid)[d]
function Base.CartesianIndices(grid::NonCommittedMarkovGrid)
    idx = 1:size(grid, 1)
    jdx = 1:size(grid, 2)

    return CartesianIndices((idx, jdx))
end

physicalstate(grid::NonCommittedMarkovGrid, i::Integer, j::Integer) = (i == size(grid, 1) ? grid.φ.upper : grid.φ((i - 1) * grid.h), j == size(grid, 2) ? grid.A.upper : grid.A((j - 1) * grid.h))

struct NCMarkovValue{T, G, M <: AbstractMatrix}
    t::T
    grid::G
    q::M
    uᵍ::M
    u::M
    τ::M
end

struct NonCommittedMarkovDiagnostics{T}
    iterations::Int
    valuegap::T
    bellmangap::T
    policygap::T
    policyimprovement::T
end

struct NonCommittedMarkovSolution{T, V, P, D}
    t::Vector{T}
    u::Vector{V}
    parameters::P
    diagnostics::Vector{D}
end

function (solution::NonCommittedMarkovSolution)(t)
    k = min(searchsortedlast(solution.t, t), length(solution.t) - 1)
    weight = (t - solution.t[k]) / (solution.t[k + 1] - solution.t[k])
    left, right = solution.u[k], solution.u[k + 1]
    q = (1 - weight) .* left.q .+ weight .* right.q
    uᵍ = (1 - weight) .* left.uᵍ .+ weight .* right.uᵍ
    u = (1 - weight) .* left.u .+ weight .* right.u
    path = solution.parameters.τᶜ
    # The last pre-jump slab uses its frozen left policy, without mixing in the new regime.
    τ = t < right.t && path(right.t; side = :left) != path(right.t) ? copy(left.τ) : (1 - weight) .* left.τ .+ weight .* right.τ
    τ = clamp.(τ, zero(t), path(t))
    return NCMarkovValue(t, left.grid, q, uᵍ, u, τ)
end

## Beliefs and initial values
ncexpectedtax(φ, τ, τᶜ) = φ * τᶜ + (1 - φ) * τ

function ncbeliefmoments(τ, τᶜ, φ, signal::Signal)
    χ = (signal.ϵ / signal.σ) * (τᶜ - τ)
    return -φ^2 * (1 - φ) * χ^2, φ * (1 - φ) * χ
end

ncstatictax(φ, a, τᶜ, parameters) = clamp(parameters.government.τ̄, 0, τᶜ)

function ncinitial(t, τᶜ, parameters)
    model, grid = parameters.model, parameters.grid
    household, firm = model.household, model.firm
    q, uᵍ, u, τ = (zeros(typeof(grid.h), size(grid)) for _ in 1:4)
    @inbounds for j in axes(q, 2), i in axes(q, 1)
        φ, a = physicalstate(grid, i, j)
        τ[i, j] = ncstatictax(φ, a, τᶜ, parameters)
        τᵉ = ncexpectedtax(φ, τ[i, j], τᶜ)
        q[i, j] = y(n(τᵉ, a, household, firm), firm) * τᵉ / household.r
        a ≥ firm.η && (q[i, j] = min(c(firm), q[i, j]))
        investment = a ≥ firm.η ? zero(t) : α(q[i, j], firm)
        uᵍ[i, j] = simplifiedcost(τᵉ, a, investment, household, firm, model.climate) / household.r
        u[i, j] = simplifiedcost(τᵉ, a, investment, household, firm, model.climate) / household.r
    end
    return NCMarkovValue(t, grid, q, uᵍ, u, τ)
end
