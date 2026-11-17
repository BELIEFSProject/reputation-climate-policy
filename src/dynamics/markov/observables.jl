"Physical observables (φ, τ, τᵉ, a, q, investment, labour, output, emissions) and signal separation χ."
function observables(u, t, p; side = t == p.stop ? :left : :right)
    solution = p.solution
    parameters = solution.parameters
    household, firm, signal, grid = parameters.model.household, parameters.model.firm, parameters.signal, parameters.grid
    φ = LE.logistic(u.z)
    a = clamp(u.a, firm.a₀, firm.η)
    τᶜ = parameters.τᶜ(t; side)

    k = min(searchsortedlast(solution.t, t), length(solution.t) - 1)
    if side == :left && k > 1 && t == solution.t[k] && parameters.τᶜ(t; side = :left) != parameters.τᶜ(t)
        k -= 1
    end

    left, right = solution.u[k], solution.u[k + 1]
    w = clamp((t - left.t) / (right.t - left.t), 0, 1)
    jump = parameters.τᶜ(right.t; side = :left) != parameters.τᶜ(right.t)
    wτ = jump && (t < right.t || side == :left) ? 0 : w
    zφ, za = normalise(φ, grid.φ) / grid.h, normalise(a, grid.A) / grid.h
    i, j = min(floor(Int, zφ) + 1, size(grid, 1) - 1), min(floor(Int, za) + 1, size(grid, 2) - 1)
    wφ, wa = zφ - (i - 1), za - (j - 1)
    q, fraction = zero(a), zero(a)

    @inbounds for dj in 0:1, di in 0:1
        weight = (di == 0 ? 1 - wφ : wφ) * (dj == 0 ? 1 - wa : wa)
        q += weight * ((1 - w) * left.q[i + di, j + dj] + w * right.q[i + di, j + dj])
        fraction += weight * ((1 - wτ) * p.fractions[k][i + di, j + dj] + wτ * p.fractions[k + 1][i + di, j + dj])
    end

    τⁿᶜ = a ≥ firm.η ? clamp(parameters.government.τ̄, 0, τᶜ) : τᶜ * clamp(fraction, 0, 1)
    τᵉ = ncexpectedtax(φ, τⁿᶜ, τᶜ)
    τ = p.committed ? τᶜ : τⁿᶜ
    investment = a ≥ firm.η ? 0 : α(q, firm)
    labour = n(τᵉ, a, household, firm)
    χ = signal.ϵ * (τᶜ - τⁿᶜ) / signal.σ

    return MarkovObservables(promote(φ, τ, τᵉ, a, q, investment, labour, y(labour, firm), e(labour, a, firm), χ)...)
end

function μ!(du, u, p, t)
    values = observables(u, t, p)
    du[1] = (p.committed ? 1 : -1) * values.χ^2 / 2
    du[2] = values.α
    du[3] = values.e
    return nothing
end

function σ!(du, u, p, t)
    du[1] = observables(u, t, p).χ
    du[2] = zero(eltype(du))
    du[3] = zero(eltype(du))
    return nothing
end

