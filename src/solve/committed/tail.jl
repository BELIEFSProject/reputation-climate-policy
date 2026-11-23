"Stationary continuation on the interior dirty-production branch."
function committedtail(model, xᶜ, vᵈ, q)
    @unpack household, firm, bank, dirty = model
    r = household.r

    # Extend the algebraic tail at trial iterates; admissibility is checked after solving.
    xᶜ = max(xᶜ, eps(one(xᶜ)))
    vᵈ = max(vᵈ, eps(one(vᵈ)))
    xᵈ = sqrt(2r * vᵈ / dirty.γᵈ)
    L̄ = lendinglimit(vᵈ, bank)
    residual = @closure z -> energydemand(z, labour(z, L̄, household, firm, bank.δᴸ, r)) - xᶜ - xᵈ
    bracket = committedratiolowerbound(model, xᶜ + xᵈ, vᵈ)
    z = Roots.find_zero(residual, (bracket, oftype(bracket, 100)), Roots.Brent())
    geometry = committedgeometry(model, z, vᵈ)
    τ = (geometry.p - dirty.γᵈ * xᵈ) / dirty.η

    committedallocationat(model, xᶜ, vᵈ, min(q, model.clean.c), z, τ; geometry)
end

"Gradients of the stationary clean value and welfare value, allowing the tail tax to adjust."
function committedtailderivatives(model, allocation, geometry=committedgeometry(model, allocation.z, allocation.vᵈ))
    @unpack household, dirty = model
    r = household.r
    xᵈ = sqrt(2r * allocation.vᵈ / dirty.γᵈ)
    ∂xᵈ = r / (dirty.γᵈ * xᵈ)
    valuezero = zero(xᵈ)

    ∂z = CommittedDerivative(inv(geometry.xz), (∂xᵈ - geometry.xv) / geometry.xz, valuezero)
    ∂n = geometry.nz * ∂z + CommittedDerivative(valuezero, geometry.nv, valuezero)
    ∂p = geometry.pz * ∂z + CommittedDerivative(valuezero, geometry.pv, valuezero)
    ∂y = geometry.yz * ∂z + CommittedDerivative(valuezero, geometry.yv, valuezero)
    ∂w = household.ν * geometry.n^inversefrisch(household) * ∂n - (1 - model.climate.γ * dirty.η * xᵈ) * ∂y + CommittedDerivative(valuezero, (model.υ * xᵈ + model.climate.γ * dirty.η * geometry.y) * ∂xᵈ, valuezero)

    ∂p / r, ∂w / r
end

"Recover endpoint multipliers for diagnostics without adding collocation variables."
function committedtailmultipliers(model, allocation, λxᶜ, λq, geometry=committedgeometry(model, allocation.z, allocation.vᵈ))
    ∂q, ∂g = committedtailderivatives(model, allocation, geometry)
    derivatives = committedstaticderivatives(model, allocation, (allocation.p - model.dirty.γᵈ * allocation.xᵈ) / model.dirty.η; geometry)
    ψq = (∂g.xᶜ - λxᶜ) / ∂q.xᶜ
    ψvᵈ = (derivatives.welfare.τ - ψq * derivatives.p.τ) / derivatives.profit.τ

    ψq, ψvᵈ, λq - ψq
end
