"Derivatives with respect to inherited clean capacity, dirty value and the current tax."
struct CommittedDerivative{T} <: StaticArrays.FieldVector{3, T}
    xᶜ::T
    vᵈ::T
    τ::T
end

struct CommittedStaticDerivatives{D}
    n::D
    p::D
    xd::D
    y::D
    welfare::D
    profit::D
end

struct CommittedGeometry{T}
    n::T
    p::T
    x::T
    y::T
    nz::T
    nv::T
    pz::T
    pv::T
    xz::T
    xv::T
    yz::T
    yv::T
    creditgap::T
end

struct CommittedPolicy{A, D, T}
    allocation::A
    tax::T
    derivatives::D
    hamiltonian::T
    foc::T
    gap::T
    creditkink::Bool
    dirtykink::Bool
    singular::Bool
end

"Intratemporal welfare excludes installation costs, which are fixed when choosing the tax."
committedflow(model, allocation) = v(allocation.n, model.household) + model.υ * allocation.xᵈ^2 / 2 - allocation.y * (1 - model.climate.γ * model.dirty.η * allocation.xᵈ)

"Analytic partial derivatives before imposing energy-market clearing."
function committedgeometry(model, z, vd; creditbranch=:auto)
    @unpack household, firm, bank = model

    ϕ = inversefrisch(household)
    barL = lendinglimit(vd, bank)
    fz = f(z, firm)
    constrained = ((1 - firm.α) * barL / (bank.δᴸ * household.ν * fz))^inv(1 + ϕ)
    unconstrained = unconstrainedlabour(z, household, firm, household.r, bank.δᴸ)
    labourlevel = min(constrained, unconstrained)

    share = firm.α * z^firm.ρ / fz
    binding = creditbranch === :binding || (creditbranch === :auto && constrained <= unconstrained)
    nz = binding ? -firm.ρ * share * labourlevel / ((1 + ϕ) * z) : (1 - firm.ρ) * share * labourlevel / (ϕ * z)
    nv = binding ? labourlevel * (1 - bank.κᵈ) * bank.θ / ((1 + ϕ) * bank.κₗ * barL) : zero(labourlevel)

    price = energyprice(z, labourlevel, household, firm)
    output = y(labourlevel, z, firm)

    pz = price * (ϕ * nz / labourlevel + (firm.ρ - 1) / z)
    pv = price * ϕ * nv / labourlevel

    xz = labourlevel + z * nz
    xv = z * nv

    yz = output * (nz / labourlevel + share / z)
    yv = output * nv / labourlevel

    CommittedGeometry(promote(labourlevel, price, z * labourlevel, output, nz, nv, pz, pv, xz, xv, yz, yv, log(unconstrained / constrained))...)
end

"Differentiate market clearing on a specified credit and dirty-utilisation branch."
function committedstaticderivatives(model, allocation, τ; creditbranch=:auto, dirtybranch=:auto)
    @unpack household, dirty = model

    geometry = committedgeometry(model, allocation.z, allocation.vᵈ; creditbranch)
    margin = allocation.p - dirty.η * τ
    interior = dirtybranch === :interior || (dirtybranch === :auto && 0 < margin < dirty.γᵈ * dirty.kᵈ)
    χ = interior ? inv(dirty.γᵈ) : zero(margin)
    denominator = geometry.xz - χ * geometry.pz

    ∂z = CommittedDerivative(inv(denominator), -(geometry.xv - χ * geometry.pv) / denominator, -χ * dirty.η / denominator)

    ∂n = geometry.nz * ∂z + CommittedDerivative(zero(margin), geometry.nv, zero(margin))
    ∂p = geometry.pz * ∂z + CommittedDerivative(zero(margin), geometry.pv, zero(margin))
    ∂xd = χ * (∂p - CommittedDerivative(zero(margin), zero(margin), dirty.η))
    ∂y = geometry.yz * ∂z + CommittedDerivative(zero(margin), geometry.yv, zero(margin))

    ∂w = household.ν * allocation.n^inversefrisch(household) * ∂n - (1 - model.climate.γ * dirty.η * allocation.xᵈ) * ∂y + (model.υ * allocation.xᵈ + model.climate.γ * dirty.η * allocation.y) * ∂xd
    # The producer's envelope condition holds in the interior and at both utilisation bounds.
    ∂profit = allocation.xᵈ * (∂p - CommittedDerivative(zero(margin), zero(margin), dirty.η))

    CommittedStaticDerivatives(∂n, ∂p, ∂xd, ∂y, ∂w, ∂profit)
end

committedblend(left, right, weight) = CommittedStaticDerivatives(weight * left.n + (1 - weight) * right.n, weight * left.p + (1 - weight) * right.p, weight * left.xd + (1 - weight) * right.xd, weight * left.y + (1 - weight) * right.y, weight * left.welfare + (1 - weight) * right.welfare, weight * left.profit + (1 - weight) * right.profit)

committedtaxderivative(derivatives, λq, λv) = derivatives.welfare.τ - λq * derivatives.p.τ - λv * derivatives.profit.τ

"Use a Clarke derivative at a minimising kink, without changing the underlying market regimes."
function committedpolicyderivatives(model, allocation, τ, λq, λv)
    geometry = committedgeometry(model, allocation.z, allocation.vᵈ)
    margin = allocation.p - model.dirty.η * τ

    roundoff = 64 * eps(one(float(margin)))
    creditkink = abs(geometry.creditgap) <= roundoff
    shutdown = abs(margin) <= roundoff * max(one(margin), abs(allocation.p))
    capacitykink = abs(margin - model.dirty.γᵈ * model.dirty.kᵈ) <= roundoff * max(one(margin), abs(margin))
    lowerbound = τ <= roundoff * max(one(τ), abs(τ))

    creditbranches = creditkink ? (lowerbound ? (:slack,) : (:slack, :binding)) : (:auto,)
    dirtybranches = shutdown ? (:shutdown,) : capacitykink ? (lowerbound ? (:interior,) : (:interior, :capacity)) : (:auto,)
    variants = [committedstaticderivatives(model, allocation, τ; creditbranch, dirtybranch) for creditbranch in creditbranches for dirtybranch in dirtybranches]

    slopes = map(derivatives -> committedtaxderivative(derivatives, λq, λv), variants)
    selected = variants[argmin(abs.(slopes))]

    if !lowerbound && !shutdown && minimum(slopes) < 0 < maximum(slopes)
        left = argmin(slopes)
        right = argmax(slopes)
        weight = slopes[right] / (slopes[right] - slopes[left])
        selected = committedblend(variants[left], variants[right], weight)
    end

    slope = committedtaxderivative(selected, λq, λv)
    foc = lowerbound ? max(-slope, zero(slope)) : shutdown ? zero(slope) : abs(slope)

    # Simultaneous switches can admit several valid convex combinations; retain that diagnostic.
    singular = length(variants) > 2 || (length(variants) > 1 && all(iszero, slopes))

    selected, foc, creditkink, shutdown || capacitykink, singular
end

"Derivative of the reduced Hamiltonian with respect to the implemented input ratio."
function committedpolicyslope(model, xc, vd, z, λq, λv; creditbranch=:auto)
    geometry = committedgeometry(model, z, vd; creditbranch)
    xd = geometry.x - xc

    model.household.ν * geometry.n^inversefrisch(model.household) * geometry.nz - (1 - model.climate.γ * model.dirty.η * xd) * geometry.yz + (model.υ * xd + model.climate.γ * model.dirty.η * geometry.y - λv * model.dirty.γᵈ * xd) * geometry.xz - λq * geometry.pz
end

"Minimise the current Hamiltonian over utilisation and capacity rents, rather than tax-path coefficients."
function committedpolicy(model, xc, vd, q, λq, λv)
    @unpack dirty = model

    untaxed = committedmarketallocation(model, xc, vd, q, zero(xc))
    upper = untaxed.z
    allcleanresidual = @closure z -> committedgeometry(model, z, vd).x - xc
    lower = Roots.find_zero(allcleanresidual, (committedratiolowerbound(model, xc, vd), upper))
    segments = [lower, upper]
    creditgap = @closure z -> committedgeometry(model, z, vd).creditgap

    if signbit(creditgap(lower)) != signbit(creditgap(upper))
        switch = Roots.find_zero(creditgap, (lower, upper), Roots.Brent())
        lower < switch < upper && push!(segments, switch)
    end

    sort!(segments)
    unique!(segments)
    candidates = copy(segments)

    for i in 1:(length(segments) - 1)
        left, right = segments[i], segments[i + 1]

        branch = creditgap((left + right) / 2) > 0 ? :binding : :slack
        slope = @closure z -> committedpolicyslope(model, xc, vd, z, λq, λv; creditbranch=branch)

        # Roots.find_zeros is a heuristic enumeration; the reported candidate margin is not a global certificate.
        append!(candidates, Roots.find_zeros(slope, left, right))
    end

    sort!(candidates)
    unique!(candidates)
    allocations = [untaxed]

    for z in candidates
        geometry = committedgeometry(model, z, vd)

        # Clipping only removes roundoff at the two endpoints of the feasible utilisation interval.
        xd = clamp(geometry.x - xc, zero(xc), dirty.kᵈ)
        τ = max((geometry.p - dirty.γᵈ * xd) / dirty.η, zero(xc))

        push!(allocations, committedallocationat(model, xc, vd, q, z, τ))
    end

    taxes = [zero(xc); [max((allocation.p - dirty.γᵈ * allocation.xᵈ) / dirty.η, zero(xc)) for allocation in allocations[2:end]]]
    objectives = map(allocation -> committedflow(model, allocation) - λq * allocation.p - λv * allocation.profit, allocations)

    best = argmin(objectives)
    allocation = allocations[best]
    τ = taxes[best]

    derivatives, foc, creditkink, dirtykink, singular = committedpolicyderivatives(model, allocation, τ, λq, λv)
    alternatives = [objectives[i] for i in eachindex(objectives) if i != best && taxes[i] != τ]
    gap = isempty(alternatives) ? oftype(τ, Inf) : max(minimum(alternatives) - objectives[best], zero(τ))
    singular |= gap <= 64 * eps(one(float(τ))) * max(one(τ), abs(objectives[best]))

    CommittedPolicy(allocation, τ, derivatives, objectives[best], foc, gap, creditkink, dirtykink, singular)
end
