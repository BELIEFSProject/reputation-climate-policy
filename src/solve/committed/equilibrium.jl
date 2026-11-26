## Static equilibrium

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
    nvᶜ::T
    pvᶜ::T
    xvᶜ::T
    yvᶜ::T
    creditgap::T
end

struct CommittedAllocation{T}
    z::T
    n::T
    ω::T
    p::T
    x::T
    xᵈ::T
    xᶜ::T
    y::T
    L::T
    L̄::T
    μ::T
    vᵈ::T
    vᶜ::T
    q::T
    τ::T
    h::T
    e::T
    investment::T
    welfare::T
    profit::T
end

"Bracket a positive equilibrium root by expanding both endpoints."
function committedroot(residual, context; upper=1.0)
    lower = min(oftype(upper, 1e-8), upper / 2)
    for expansion in 0:64
        left, right = residual(lower), residual(upper)
        if !(isfinite(left) && isfinite(right))
            throw(DomainError((context, lower, upper, left, right), "Non-finite static-equilibrium residual."))
        end
        if iszero(left)
            return lower
        end
        if iszero(right)
            return upper
        end
        if signbit(left) != signbit(right)
            return oftype(upper, Roots.find_zero(residual, (lower, upper), Roots.Brent()))
        end
        if expansion == 64
            break
        end
        lower /= 2
        upper *= 2
    end
    throw(DomainError((context, lower, upper, residual(lower), residual(upper)), "No positive equilibrium root was bracketed after 64 expansions."))
end

function committedgeometry(model, z, vᵈ, vᶜ; creditbranch=:auto)
    @unpack household, firm, bank = model
    ϕ = inversefrisch(household)
    L̄ = lendinglimit(vᵈ, vᶜ, bank)
    fz = f(z, firm)
    constrained = ((1 - firm.α) * L̄ / (bank.δᴸ * household.ν * fz))^inv(1 + ϕ)
    unconstrained = unconstrainedlabour(z, household, firm, household.r, bank.δᴸ)
    n = min(constrained, unconstrained)
    share = firm.α * z^firm.ρ / fz
    binding = creditbranch === :binding || (creditbranch === :auto && constrained <= unconstrained)
    nz = binding ? -firm.ρ * share * n / ((1 + ϕ) * z) : (1 - firm.ρ) * share * n / (ϕ * z)
    nv = binding ? n * (1 - bank.κᵈ) * bank.θ / ((1 + ϕ) * bank.κₗ * L̄) : zero(n)
    nvᶜ = binding ? n * (1 - bank.κᶜ) * bank.θᶜ / ((1 + ϕ) * bank.κₗ * L̄) : zero(n)
    p = energyprice(z, n, household, firm)
    output = y(n, z, firm)
    pz, pv = p * (ϕ * nz / n + (firm.ρ - 1) / z), p * ϕ * nv / n
    xz, xv = n + z * nz, z * nv
    yz, yv = output * (nz / n + share / z), output * nv / n
    pvᶜ, xvᶜ, yvᶜ = p * ϕ * nvᶜ / n, z * nvᶜ, output * nvᶜ / n
    T = promote_type(typeof(n), typeof(p), typeof(output), typeof(vᵈ), typeof(vᶜ))

    CommittedGeometry{T}(n, p, z * n, output, nz, nv, pz, pv, xz, xv, yz, yv, nvᶜ, pvᶜ, xvᶜ, yvᶜ, log(unconstrained / constrained))
end

function committedallocationat(model, xᶜ, vᵈ, vᶜ, q, z, τ; geometry=committedgeometry(model, z, vᵈ, vᶜ), xᵈ=dirtyoutput(geometry.p, τ, model.dirty), h=dirtyabatement(τ, xᵈ, model.dirty))
    @unpack household, firm, bank, dirty, clean = model
    investment = cleaninvestment(q, clean)
    welfare = welfarecost(geometry.n, xᵈ, h, investment, xᶜ, geometry.y, household, dirty, clean, model.government, model.climate.γ)
    profit = (geometry.p - dirty.η * τ) * xᵈ - dirty.γᵈ * xᵈ^2 / 2 + τ * h - dirty.ζ * h^2 / 2
    ω = wage(z, geometry.n, household)
    L = liquidity(z, geometry.n, household, firm, bank.δᴸ)
    μ = max(shadowcost(z, geometry.n, household, firm, bank.δᴸ, household.r), zero(q))
    T = promote_type(typeof(geometry.p), typeof(welfare), typeof(xᶜ), typeof(vᵈ), typeof(vᶜ), typeof(q), typeof(τ))

    CommittedAllocation{T}(z, geometry.n, ω, geometry.p, geometry.x, xᵈ, xᶜ, geometry.y, L, lendinglimit(vᵈ, vᶜ, bank), μ, vᵈ, vᶜ, q, τ, h, dirty.η * xᵈ - h, investment, welfare, profit)
end

"Clear the intratemporal markets for a given expected levy and private valuations."
function committedallocation(model, xᶜ, vᵈ, vᶜ, q, τ)
    residual = @closure z -> begin
        local g = committedgeometry(model, z, vᵈ, vᶜ)
        g.x - xᶜ - dirtyoutput(g.p, τ, model.dirty)
    end
    z = committedroot(residual, (:allocation, xᶜ, vᵈ, vᶜ, τ))
    committedallocationat(model, xᶜ, vᵈ, vᶜ, q, z, τ)
end

"Stationary reference for the initial guess, at p=rc."
function committedstationary(model, τ)
    r = model.household.r
    q = model.clean.c
    p = r * q
    xᵈ = dirtyoutput(p, τ, model.dirty)
    vᵈ = dirtyprofit(p, τ, xᵈ, model.dirty) / r
    residual = @closure z -> begin
        local n = ((1 - model.firm.α) * p * z^(1 - model.firm.ρ) / (model.firm.α * model.household.ν))^inv(inversefrisch(model.household))
        local vᶜ = q * max(z * n - xᵈ, zero(q))
        committedgeometry(model, z, vᵈ, vᶜ).p - p
    end
    z = committedroot(residual, (:stationary, τ))
    n = ((1 - model.firm.α) * p * z^(1 - model.firm.ρ) / (model.firm.α * model.household.ν))^inv(inversefrisch(model.household))
    vᶜ = q * (z * n - xᵈ)
    if !(vᶜ > 0)
        error("The stationary reference requires positive clean capacity.")
    end
    geometry = committedgeometry(model, z, vᵈ, vᶜ)

    committedallocationat(model, geometry.x - xᵈ, vᵈ, vᶜ, q, z, τ; geometry)
end

## Policy and fixed-tax derivatives

"Return welfare, price and profit gradients, ordered as (xᶜ, vᵈ, vᶜ, τ)."
function committedderivatives(model, allocation, λq, λvᵈ)
    @unpack household, dirty = model
    margin = allocation.p - dirty.η * allocation.τ
    roundoff = 64 * eps(one(float(margin)))
    geometry = committedgeometry(model, allocation.z, allocation.vᵈ, allocation.vᶜ)
    creditkink = abs(geometry.creditgap) <= roundoff
    abatement = allocation.τ >= dirty.ζ * dirty.η * allocation.xᵈ - roundoff * max(one(margin), abs(allocation.τ))
    capacitykink = abs(margin - dirty.γᵈ * dirty.kᵈ) <= roundoff * max(one(margin), abs(margin))
    lowerbound = allocation.τ <= roundoff * max(one(margin), abs(allocation.τ))

    function derivatives(creditbranch, dirtybranch)
        g = committedgeometry(model, allocation.z, allocation.vᵈ, allocation.vᶜ; creditbranch)
        fullabatement = dirtybranch === :abatement || (dirtybranch === :auto && abatement)
        capacity = dirtybranch === :capacity || (dirtybranch === :auto && allocation.xᵈ >= dirty.kᵈ) || (fullabatement && allocation.p >= (dirty.γᵈ + dirty.ζ * dirty.η^2) * dirty.kᵈ)
        χ = capacity ? zero(margin) : inv(dirty.γᵈ + (fullabatement ? dirty.ζ * dirty.η^2 : zero(margin)))
        taxsupply = fullabatement ? zero(margin) : -dirty.η * χ
        denominator = g.xz - χ * g.pz
        ∂z = SA.SVector(inv(denominator), -(g.xv - χ * g.pv) / denominator, -(g.xvᶜ - χ * g.pvᶜ) / denominator, taxsupply / denominator)
        ∂n = g.nz * ∂z + SA.SVector(zero(margin), g.nv, g.nvᶜ, zero(margin))
        ∂p = g.pz * ∂z + SA.SVector(zero(margin), g.pv, g.pvᶜ, zero(margin))
        direction = SA.SVector(zero(margin), zero(margin), zero(margin), one(margin))
        ∂xᵈ = χ * ∂p + taxsupply * direction
        ∂h = fullabatement ? dirty.η * ∂xᵈ : direction / dirty.ζ
        ∂e = dirty.η * ∂xᵈ - ∂h
        ∂y = g.yz * ∂z + SA.SVector(zero(margin), g.yv, g.yvᶜ, zero(margin))
        ∂w = household.ν * allocation.n^inversefrisch(household) * ∂n - (1 - model.climate.γ * allocation.e) * ∂y + model.government.υ * allocation.xᵈ * ∂xᵈ + dirty.ζ * allocation.h * ∂h + model.climate.γ * allocation.y * ∂e
        ∂π = allocation.xᵈ * ∂p - allocation.e * direction
        ∂w, ∂p, ∂π
    end

    selected = derivatives(:auto, abatement ? :abatement : :auto)
    if !creditkink && !capacitykink
        return selected
    end

    # At a minimising kink, combine limiting gradients to make the tax derivative zero.
    slope(d) = d[1][4] - λq * d[2][4] - λvᵈ * d[3][4]
    creditbranches = creditkink ? (lowerbound ? (:slack,) : (:slack, :binding)) : (:auto,)
    dirtybranches = abatement ? (:abatement,) : capacitykink ? (lowerbound ? (:interior,) : (:interior, :capacity)) : (:auto,)
    selected = derivatives(first(creditbranches), first(dirtybranches))
    left, right = selected, selected
    for creditbranch in creditbranches, dirtybranch in dirtybranches
        candidate = derivatives(creditbranch, dirtybranch)
        if abs(slope(candidate)) < abs(slope(selected))
            selected = candidate
        end
        if slope(candidate) < slope(left)
            left = candidate
        end
        if slope(candidate) > slope(right)
            right = candidate
        end
    end
    if !lowerbound && !abatement && slope(left) < 0 < slope(right)
        weight = slope(right) / (slope(right) - slope(left))
        selected = map((l, r) -> weight * l + (1 - weight) * r, left, right)
    end
    selected
end

"Find a stationary policy branch and compare it with bounds and the credit corner."
function committedpolicy(model, xᶜ, vᵈ, vᶜ, q, λq, λvᵈ, λvᶜ; zguess=one(xᶜ))
    xᶜ = max(xᶜ, eps(one(float(xᶜ))))
    vᵈ = max(vᵈ, zero(vᵈ))
    vᶜ = max(vᶜ, zero(vᶜ))
    pricecostate = λq + xᶜ * λvᶜ
    @unpack household, dirty = model
    untaxed = @closure z -> begin
        local g = committedgeometry(model, z, vᵈ, vᶜ)
        g.x - xᶜ - dirtyoutput(g.p, zero(q), dirty)
    end
    upper = committedroot(untaxed, (:untaxed, xᶜ, vᵈ))
    abated = @closure z -> begin
        local g = committedgeometry(model, z, vᵈ, vᶜ)
        g.x - xᶜ - min(g.p / (dirty.γᵈ + dirty.ζ * dirty.η^2), dirty.kᵈ)
    end
    lower = min(committedroot(abated, (:abatement, xᶜ, vᵈ); upper), upper)

    residual = @closure z -> begin
        local ratio = clamp(z, lower, upper)
        local g = committedgeometry(model, ratio, vᵈ, vᶜ)
        local xᵈ = g.x - xᶜ
        local τ = (g.p - dirty.γᵈ * xᵈ) / dirty.η
        local h = τ / dirty.ζ
        local e = dirty.η * xᵈ - h
        local τz = (g.pz - dirty.γᵈ * g.xz) / dirty.η
        slope = household.ν * g.n^inversefrisch(household) * g.nz - (1 - model.climate.γ * e) * g.yz + (model.government.υ * xᵈ + model.climate.γ * dirty.η * g.y - λvᵈ * dirty.γᵈ * xᵈ) * g.xz + ((1 - λvᵈ) * τ - model.climate.γ * g.y) * τz / dirty.ζ - pricecostate * g.pz
        z - clamp(ratio - slope, lower, upper)
    end
    z = if lower == upper
        lower
    else
        try
            Roots.find_zero(residual, clamp(zguess, lower, upper), Roots.Order0())
        catch exception
            if !(exception isa Roots.ConvergenceFailed)
                rethrow()
            end
            Roots.find_zero(residual, (lower, upper), Roots.Brent())
        end
    end
    z = clamp(oftype(upper, z), lower, upper)
    candidate = @closure ratio -> begin
        local g = committedgeometry(model, ratio, vᵈ, vᶜ)
        local xᵈ = clamp(g.x - xᶜ, zero(q), dirty.kᵈ)
        local τ = clamp((g.p - dirty.γᵈ * xᵈ) / dirty.η, zero(q), dirty.ζ * dirty.η * xᵈ)
        committedallocationat(model, xᶜ, vᵈ, vᶜ, q, ratio, τ; geometry=g)
    end
    value = @closure a -> a.welfare - pricecostate * a.p - λvᵈ * a.profit
    allocation = candidate(z)
    for ratio in (lower, upper)
        boundary = candidate(ratio)
        if value(boundary) < value(allocation)
            allocation = boundary
        end
    end
    leftgap = committedgeometry(model, lower, vᵈ, vᶜ).creditgap
    rightgap = committedgeometry(model, upper, vᵈ, vᶜ).creditgap
    if signbit(leftgap) != signbit(rightgap)
        creditratio = oftype(upper, Roots.find_zero(z -> committedgeometry(model, z, vᵈ, vᶜ).creditgap, (lower, upper), Roots.Brent()))
        corner = candidate(creditratio)
        if value(corner) < value(allocation)
            allocation = corner
        end
    end
    g = committedgeometry(model, upper, vᵈ, vᶜ)
    if g.x - xᶜ >= dirty.kᵈ - 64eps(one(q)) * max(one(q), dirty.kᵈ)
        taxtop = clamp((g.p - dirty.γᵈ * dirty.kᵈ) / dirty.η, zero(q), dirty.ζ * dirty.η * dirty.kᵈ)
        vertex = λvᵈ < 1 ? clamp((model.climate.γ * g.y - λvᵈ * dirty.ζ * dirty.η * dirty.kᵈ) / (1 - λvᵈ), zero(q), taxtop) : zero(q)
        for τ in (zero(q), vertex, taxtop)
            capacity = committedallocationat(model, xᶜ, vᵈ, vᶜ, q, upper, τ; geometry=g)
            if value(capacity) < value(allocation)
                allocation = capacity
            end
        end
    end
    ∂w, ∂p, ∂π = committedderivatives(model, allocation, pricecostate, λvᵈ)

    allocation, ∂w, ∂p, ∂π
end

## Stationary tail with interior dirty utilisation

"Stationary tail at p=rq; return gradients of f=vᵈ-πd/r, g=w/r and τ in (xᶜ,vᵈ,q)."
function committedtail(model, xᶜ, vᵈ, q)
    r = model.household.r
    xᶜ = max(xᶜ, eps(one(float(xᶜ))))
    vᵈ = max(vᵈ, zero(vᵈ))
    q = max(q, eps(one(float(q))))
    vᶜ = q * xᶜ
    residual = @closure z -> committedgeometry(model, z, vᵈ, vᶜ).p - r * q
    z = committedroot(residual, (:tail, xᶜ, vᵈ, q))
    geometry = committedgeometry(model, z, vᵈ, vᶜ)
    xᵈ = geometry.x - xᶜ
    τ = (geometry.p - model.dirty.γᵈ * xᵈ) / model.dirty.η
    # Extend the interior tail smoothly to infeasible trial points; validate after the solve.
    h = τ / model.dirty.ζ
    allocation = committedallocationat(model, xᶜ, vᵈ, vᶜ, min(q, model.clean.c), z, τ; geometry, xᵈ, h)
    ∂vᶜ = SA.SVector(q, zero(q), xᶜ)
    ∂z = (SA.SVector(zero(q), -geometry.pv, r) - geometry.pvᶜ * ∂vᶜ) / geometry.pz
    ∂n = geometry.nz * ∂z + SA.SVector(zero(q), geometry.nv, zero(q)) + geometry.nvᶜ * ∂vᶜ
    ∂p = SA.SVector(zero(q), zero(q), r)
    ∂xᵈ = geometry.xz * ∂z + SA.SVector(-one(q), geometry.xv, zero(q)) + geometry.xvᶜ * ∂vᶜ
    ∂τ = (∂p - model.dirty.γᵈ * ∂xᵈ) / model.dirty.η
    ∂h = ∂τ / model.dirty.ζ
    ∂e = model.dirty.η * ∂xᵈ - ∂h
    ∂y = geometry.yz * ∂z + SA.SVector(zero(q), geometry.yv, zero(q)) + geometry.yvᶜ * ∂vᶜ
    ∂w = model.household.ν * geometry.n^inversefrisch(model.household) * ∂n - (1 - model.climate.γ * allocation.e) * ∂y + model.government.υ * xᵈ * ∂xᵈ + model.dirty.ζ * h * ∂h + model.climate.γ * geometry.y * ∂e
    ∂f = SA.SVector(zero(q), one(q), zero(q)) - (xᵈ * ∂p - allocation.e * ∂τ) / r

    allocation, ∂f, ∂w / r, ∂τ
end

committedcomplementarity(a, b) = hypot(a, b) - a - b
