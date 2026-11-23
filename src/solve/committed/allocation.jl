"Static competitive allocation under full-information commitment."
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
    q::T
    investment::T
    welfare::T
    profit::T
end

function equilibriumclearing(z, (vᵈ, price, model))
    L̄ = lendinglimit(vᵈ, model.bank)

    n = labour(z, L̄, model.household, model.firm, model.bank.δᴸ, model.household.r)
    
    return energyprice(z, n, model.household, model.firm) - price
end

"Reference allocation at the stationary-price identity p=rq."
function committedallocation(model, xᶜ, vᵈ, q, τ; zmax = 100.)
    @unpack household, firm, dirty, clean, bank = model
    
    price = household.r * q
    objective = Base.Fix2(equilibriumclearing, (vᵈ, price, model))
    z = Roots.find_zero(objective, (1e-3, zmax))
    
    return committedallocationat(model, xᶜ, vᵈ, q, z, τ)
end

function committedmarketclearing(z, (xᶜ, vᵈ, τ, model))
    @unpack household, firm, dirty, bank = model

    L̄ = lendinglimit(vᵈ, bank)
    n = labour(z, L̄, household, firm, bank.δᴸ, household.r)

    return energymarketresidual(z, n, xᶜ, τ, household, firm, dirty)
end

"Choose a positive lower bracket with energy demand below inherited clean output."
function committedratiolowerbound(model, xᶜ, vᵈ)
    @unpack household, bank = model

    L̄ = lendinglimit(vᵈ, bank)
    nmax = (L̄ / (bank.δᴸ * household.ν))^inv(1 + inversefrisch(household))

    return min(1e-3, xᶜ / (2 * nmax))
end

"Competitive transition allocation with q determined by future prices."
function committedmarketallocation(model, xᶜ, vᵈ, q, τ; zmax = 100.)
    objective = Base.Fix2(committedmarketclearing, (xᶜ, vᵈ, τ, model))

    z = Roots.find_zero(objective, (committedratiolowerbound(model, xᶜ, vᵈ), zmax))

    return committedallocationat(model, xᶜ, vᵈ, q, z, τ)
end

"Stationary reference at the marginal-investment boundary `p=rc`."
function committedstationary(model, τ; zmax = 100.)
    @unpack household, firm, dirty, clean, bank = model

    q = clean.c
    price = household.r * q
    xᵈ = dirtyoutput(price, τ, dirty)
    vᵈ = dirtyprofit(price, τ, xᵈ, dirty) / household.r

    L̄ = lendinglimit(vᵈ, bank)

    objective = Base.Fix2(equilibriumclearing, (vᵈ, price, model))
    z = Roots.find_zero(objective, (1e-3, zmax))

    xᶜ = energydemand(z, labour(z, L̄, household, firm, bank.δᴸ, household.r)) - xᵈ
    
    return committedallocationat(model, xᶜ, vᵈ, q, z, τ)
end


"Recover an allocation at a policy-selected input ratio without another market-clearing solve."
function committedallocationat(model, xᶜ, vᵈ, q, z, τ)
    @unpack household, firm, bank, dirty = model
    r = household.r

    barL = lendinglimit(vᵈ, bank)
    labourlevel = labour(z, barL, household, firm, bank.δᴸ, r)

    wagelevel = wage(z, labourlevel, household)
    price = energyprice(z, labourlevel, household, firm)
    dirtylevel = dirtyoutput(price, τ, dirty)
    demand = energydemand(z, labourlevel)
    output = y(labourlevel, z, firm)

    liquiditylevel = liquidity(z, labourlevel, household, firm, model.bank.δᴸ)
    shadow = max(shadowcost(z, labourlevel, household, firm, model.bank.δᴸ, r), zero(price))

    investment = cleaninvestment(q, model.clean)
    welfare = welfarecost(labourlevel, dirtylevel, investment, xᶜ, output, household, dirty, model.clean, model.υ, model.climate.γ)
    profit = dirtyprofit(price, τ, dirtylevel, dirty)

    CommittedAllocation(promote(z, labourlevel, wagelevel, price, demand, dirtylevel, xᶜ, output, liquiditylevel, barL, shadow, vᵈ, q, investment, welfare, profit)...)
end

"Tax sufficient to shut dirty production at every feasible subsequent clean capacity."
function committedtaxbound(model, xᶜ₀; zmax = 100.)
    @unpack household, firm, bank, dirty = model

    n = @closure z -> unconstrainedlabour(z, household, firm, household.r, bank.δᴸ)
    greenresidual = @closure z -> energydemand(z, n(z)) - xᶜ₀

    z = Roots.find_zero(greenresidual, (1e-3, zmax))

    return energyprice(z, n(z), household, firm) / dirty.η
end
