## Normalised state space
struct NonCommittedMarkovGrid{T}
    h::T
    φ::AffineGridMap{T}
    A::AffineGridMap{T}
end

function NonCommittedMarkovGrid(model::AbstractModel, h; alower = model.firm.a₀)
    isfinite(h) && h > 0 || throw(ArgumentError("The grid step must be finite and positive."))
    N = round(Int, inv(h))
    N ≥ 1 && isapprox(N * h, one(h)) || throw(ArgumentError("The grid step must divide [0, 1]."))
    isfinite(alower) && isfinite(model.firm.η) && 0 ≤ alower < model.firm.η || throw(ArgumentError("The abatement grid must lie in [0,η] and have finite positive width."))
    T = promote_type(typeof(float(h)), typeof(alower), typeof(model.firm.η))
    return NonCommittedMarkovGrid(one(T) / N, AffineGridMap(zero(T), one(T)), AffineGridMap(T(alower), T(model.firm.η)))
end

Base.size(grid::NonCommittedMarkovGrid) = (round(Int, inv(grid.h)) + 1, round(Int, inv(grid.h)) + 1)
Base.size(grid::NonCommittedMarkovGrid, d::Integer) = size(grid)[d]
physicalstate(grid::NonCommittedMarkovGrid, i::Integer, j::Integer) = (i == size(grid, 1) ? grid.φ.upper : grid.φ((i - 1) * grid.h), j == size(grid, 2) ? grid.A.upper : grid.A((j - 1) * grid.h))

# Retained to identify and read saved results from the superseded temporal scheme.
struct NonCommittedMarkovParameters{M, G, S, C, D, T}
    model::M
    government::G
    signal::S
    τᶜ::C
    grid::D
    Δtmax::T
    tolerance::T
    policytolerance::T
    relaxation::T
    maxiters::Int
end

struct NonCommittedMarkovValue{T, G, M <: AbstractMatrix}
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
    first(solution.t) ≤ t ≤ last(solution.t) || throw(DomainError(t, "Calendar time is outside the solution."))
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
    return NonCommittedMarkovValue(t, left.grid, q, uᵍ, u, τ)
end

## Beliefs and initial values
noncommittedexpectedtax(φ, τ, τᶜ) = φ * τᶜ + (1 - φ) * τ

function noncommittedbeliefmoments(τ, τᶜ, φ, signal::Signal)
    χ = (signal.ϵ / signal.σ) * (τᶜ - τ)
    return -φ^2 * (1 - φ) * χ^2, φ * (1 - φ) * χ
end

function noncommittedstatictax(φ, a, τᶜ, parameters)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    if household isa Household
        return clamp(statictax(a, household, firm, climate), 0, τᶜ)
    end
    curvature = κ₂(a, household, firm, climate)
    iszero(curvature) && return zero(τᶜ)
    return clamp(-κ₁(a, household, firm, climate) / curvature, zero(τᶜ), τᶜ)
end

function noncommittedinitial(t, τᶜ, parameters)
    model, grid = parameters.model, parameters.grid
    household, firm = model.household, model.firm
    q, uᵍ, u, τ = (zeros(typeof(grid.h), size(grid)) for _ in 1:4)
    @inbounds for j in axes(q, 2), i in axes(q, 1)
        φ, a = physicalstate(grid, i, j)
        τ[i, j] = noncommittedstatictax(φ, a, τᶜ, parameters)
        τᵉ = noncommittedexpectedtax(φ, τ[i, j], τᶜ)
        q[i, j] = y(n(τᵉ, a, household, firm), firm) * τᵉ / household.r
        a ≥ firm.η && (q[i, j] = min(c(firm), q[i, j]))
        investment = a ≥ firm.η ? zero(t) : α(q[i, j], firm)
        uᵍ[i, j] = simplifiedcost(τ[i, j], a, investment, household, firm, model.climate) / household.r
        u[i, j] = simplifiedcost(τᵉ, a, investment, household, firm, model.climate) / household.r
    end
    return NonCommittedMarkovValue(t, grid, q, uᵍ, u, τ)
end
