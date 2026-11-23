"Derivatives with respect to inherited clean capacity, dirty value and the current tax."
struct CommittedDerivative{T} <: StaticArrays.FieldVector{3, T}
    xᶜ::T
    vᵈ::T
    τ::T
end

StaticArrays.similar_type(::Type{<:CommittedDerivative}, ::Type{T}, ::StaticArrays.Size{(3,)}) where {T} = CommittedDerivative{T}

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
    creditkink::Bool
    dirtykink::Bool
    singular::Bool
end

"Intratemporal welfare exᶜludes installation costs, which are fixed when choosing the tax."
committedflow(model, allocation) = v(allocation.n, model.household) + model.υ * allocation.xᵈ^2 / 2 - allocation.y * (1 - model.climate.γ * model.dirty.η * allocation.xᵈ)

"Analytic partial derivatives before imposing energy-market clearing."
function committedgeometry(model, z, vᵈ; creditbranch=:auto)
    @unpack household, firm, bank = model

    ϕ = inversefrisch(household)
    barL = lendinglimit(vᵈ, bank)
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

    T = promote_type(typeof(labourlevel), typeof(price), typeof(output), typeof(vᵈ))

    CommittedGeometry{T}(labourlevel, price, z * labourlevel, output, nz, nv, pz, pv, xz, xv, yz, yv, log(unconstrained / constrained))
end

"Differentiate market clearing on a specified credit and dirty-utilisation branch."
function committedstaticderivatives(model, allocation, τ; creditbranch=:auto, dirtybranch=:auto, geometry=committedgeometry(model, allocation.z, allocation.vᵈ; creditbranch))
    @unpack household, dirty = model

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

committedtaxderivative(derivatives, λq, λvᵈ) = derivatives.welfare.τ - λq * derivatives.p.τ - λvᵈ * derivatives.profit.τ

"Use a Clarke derivative at a minimising kink, without changing the underlying market regimes."
function committedpolicyderivatives(model, allocation, τ, λq, λvᵈ, geometry=committedgeometry(model, allocation.z, allocation.vᵈ))
    margin = allocation.p - model.dirty.η * τ

    roundoff = 64 * eps(one(float(margin)))
    creditkink = abs(geometry.creditgap) <= roundoff
    shutdown = abs(margin) <= roundoff * max(one(margin), abs(allocation.p))
    capacitykink = abs(margin - model.dirty.γᵈ * model.dirty.kᵈ) <= roundoff * max(one(margin), abs(margin))
    lowerbound = τ <= roundoff * max(one(τ), abs(τ))

    # Away from regime switches there is only one derivative, using the already computed geometry.
    if !creditkink && !capacitykink
        derivatives = committedstaticderivatives(model, allocation, τ; dirtybranch=shutdown ? :shutdown : :auto, geometry)
        slope = committedtaxderivative(derivatives, λq, λvᵈ)
        foc = lowerbound ? max(-slope, zero(slope)) : shutdown ? zero(slope) : abs(slope)

        return derivatives, foc, creditkink, shutdown, false
    end

    creditbranches = creditkink ? (lowerbound ? (:slack,) : (:slack, :binding)) : (:auto,)
    dirtybranches = shutdown ? (:shutdown,) : capacitykink ? (lowerbound ? (:interior,) : (:interior, :capacity)) : (:auto,)
    selected = committedstaticderivatives(model, allocation, τ; creditbranch=first(creditbranches), dirtybranch=first(dirtybranches))
    selectedslope = committedtaxderivative(selected, λq, λvᵈ)
    left = selected
    right = selected
    leftslope = selectedslope
    rightslope = selectedslope

    for creditbranch in creditbranches, dirtybranch in dirtybranches
        creditbranch === first(creditbranches) && dirtybranch === first(dirtybranches) && continue
        derivatives = committedstaticderivatives(model, allocation, τ; creditbranch, dirtybranch)
        slope = committedtaxderivative(derivatives, λq, λvᵈ)

        if abs(slope) < abs(selectedslope)
            selected = derivatives
            selectedslope = slope
        end
        if slope < leftslope
            left = derivatives
            leftslope = slope
        end
        if slope > rightslope
            right = derivatives
            rightslope = slope
        end
    end

    if !lowerbound && !shutdown && leftslope < 0 < rightslope
        weight = rightslope / (rightslope - leftslope)
        selected = committedblend(left, right, weight)
    end

    slope = committedtaxderivative(selected, λq, λvᵈ)
    foc = lowerbound ? max(-slope, zero(slope)) : shutdown ? zero(slope) : abs(slope)

    # Simultaneous switches can admit several valid convex combinations; retain that diagnostic.
    variants = length(creditbranches) * length(dirtybranches)
    singular = variants > 2 || (variants > 1 && iszero(leftslope) && iszero(rightslope))

    selected, foc, creditkink, shutdown || capacitykink, singular
end

"Derivative of the reduced Hamiltonian with respect to the implemented input ratio."
function committedpolicyslope(model, xᶜ, vᵈ, z, λq, λvᵈ; creditbranch=:auto)
    geometry = committedgeometry(model, z, vᵈ; creditbranch)
    xd = geometry.x - xᶜ

    model.household.ν * geometry.n^inversefrisch(model.household) * geometry.nz - (1 - model.climate.γ * model.dirty.η * xd) * geometry.yz + (model.υ * xd + model.climate.γ * model.dirty.η * geometry.y - λvᵈ * model.dirty.γᵈ * xd) * geometry.xz - λq * geometry.pz
end

"Follow one stationary policy branch; compare boundary Hamiltonians only in diagnostics."
function committedpolicy(model, xᶜ, vᵈ, q, λq, λvᵈ; zguess=one(xᶜ))
    @unpack dirty = model

    lower, upper = committedpolicybounds(model, xᶜ, vᵈ)

    # The projected FOC includes the two utilisation endpoints and keeps trial ratios positive.
    residual = @closure z -> begin
        ratio = clamp(z, lower, upper)
        slope = committedpolicyslope(model, xᶜ, vᵈ, ratio, λq, λvᵈ)
        z - clamp(ratio - slope, lower, upper)
    end

    # A fixed reference avoids evaluation-order-dependent policies inside the BVP Jacobian.
    z = if lower == upper
        lower
    else
        try
            Roots.find_zero(residual, clamp(zguess, lower, upper), Roots.Order0())
        catch exception
            exception isa Roots.ConvergenceFailed || rethrow()
            Roots.find_zero(residual, (lower, upper), Roots.Brent())
        end
    end

    z = clamp(z, lower, upper)
    geometry = committedgeometry(model, z, vᵈ)
    xd = clamp(geometry.x - xᶜ, zero(xᶜ), dirty.kᵈ)
    τ = max((geometry.p - dirty.γᵈ * xd) / dirty.η, zero(xᶜ))
    allocation = committedallocationat(model, xᶜ, vᵈ, q, z, τ; geometry)
    derivatives, foc, creditkink, dirtykink, singular = committedpolicyderivatives(model, allocation, τ, λq, λvᵈ, geometry)
    hamiltonian = committedflow(model, allocation) - λq * allocation.p - λvᵈ * allocation.profit
    T = promote_type(typeof(τ), typeof(hamiltonian), typeof(foc))

    CommittedPolicy{typeof(allocation), typeof(derivatives), T}(allocation, τ, derivatives, hamiltonian, foc, creditkink, dirtykink, singular)
end

"Feasible input-ratio endpoints without constructing allocations or their derivatives."
function committedpolicybounds(model, xᶜ, vᵈ)
    @unpack household, firm, bank, dirty = model

    L̄ = lendinglimit(vᵈ, bank)
    labourlevel = @closure z -> labour(z, L̄, household, firm, bank.δᴸ, household.r)
    untaxedresidual = @closure z -> energymarketresidual(z, labourlevel(z), xᶜ, zero(xᶜ), household, firm, dirty)
    bracket = committedratiolowerbound(model, xᶜ, vᵈ)
    upper = Roots.find_zero(untaxedresidual, (bracket, 100.), Roots.Brent())
    allcleanresidual = @closure z -> energydemand(z, labourlevel(z)) - xᶜ
    lower = Roots.find_zero(allcleanresidual, (bracket, upper), Roots.Brent())

    lower, upper
end

"Positive gaps flag a boundary or nearby ratio with a lower Hamiltonian than the selected policy."
function committedboundarygap(model, policy, λq, λvᵈ)
    allocation = policy.allocation
    xᶜ, vᵈ, q = allocation.xᶜ, allocation.vᵈ, allocation.q
    lower, upper = committedpolicybounds(model, xᶜ, vᵈ)
    uppergeometry = committedgeometry(model, upper, vᵈ)
    untaxed = committedallocationat(model, xᶜ, vᵈ, q, upper, zero(xᶜ); geometry=uppergeometry)
    hamiltonian(a) = committedflow(model, a) - λq * a.p - λvᵈ * a.profit
    lowergeometry = committedgeometry(model, lower, vᵈ)
    clean = committedallocationat(model, xᶜ, vᵈ, q, lower, lowergeometry.p / model.dirty.η; geometry=lowergeometry)
    minimumvalue = min(policy.hamiltonian, hamiltonian(untaxed), hamiltonian(clean))

    # Also detect local maxima and check the capacity-rent endpoint at the untaxed ratio.
    step = cbrt(eps(one(allocation.z))) * max(one(allocation.z), abs(allocation.z))
    for z in (max(lower, allocation.z - step), min(upper, allocation.z + step), upper)
        geometry = committedgeometry(model, z, vᵈ)
        xd = clamp(geometry.x - xᶜ, zero(xᶜ), model.dirty.kᵈ)
        τ = max((geometry.p - model.dirty.γᵈ * xd) / model.dirty.η, zero(xᶜ))
        candidate = committedallocationat(model, xᶜ, vᵈ, q, z, τ; geometry)
        minimumvalue = min(minimumvalue, hamiltonian(candidate))
    end

    max(policy.hamiltonian - minimumvalue, zero(policy.hamiltonian))
end
