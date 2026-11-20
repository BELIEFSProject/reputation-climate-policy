"Static competitive allocation under full-information commitment."
struct CommittedAllocation{T}
    z::T
    n::T
    ω::T
    p::T
    x::T
    xd::T
    xc::T
    y::T
    L::T
    barL::T
    μ::T
    vd::T
    q::T
    investment::T
    welfare::T
    profit::T
end

"Bracket a positive scalar equilibrium root without fixing its economic domain."
function committedroot(residual, zguess)
    lower = upper = float(zguess)
    lower > zero(lower) || error("A committed equilibrium requires a positive initial input ratio.")
    flower = fupper = residual(lower)
    iszero(flower) && return lower

    for iteration in 1:128
        lower /= 2
        upper *= 2
        flower = residual(lower)
        fupper = residual(upper)
        iszero(flower) && return lower
        iszero(fupper) && return upper
        signbit(flower) != signbit(fupper) && return Roots.find_zero(residual, (lower, upper), Roots.Brent())
    end

    error("Could not bracket a positive committed equilibrium root; check the trial states and lending capacity.")
end

"Clear energy and labour markets while retaining the supplied forward-looking valuations."
function committedallocation(model, xc, vd, q, τ; zguess=one(xc))
    household = model.household
    firm = model.firm
    dirty = model.dirty
    r = household.r
    barL = lendinglimit(vd, model.bank)
    n(z) = labour(z, barL, household, firm, model.bank.δL, r)
    residual(z) = energymarketresidual(z, n(z), xc, τ, household, firm, dirty)
    z = committedroot(residual, zguess)
    labourlevel = n(z)
    wagelevel = wage(z, labourlevel, household)
    price = energyprice(z, labourlevel, household, firm)
    dirtylevel = dirtyoutput(price, τ, dirty)
    demand = energydemand(z, labourlevel)
    output = y(labourlevel, z, firm)
    liquiditylevel = liquidity(z, labourlevel, household, firm, model.bank.δL)
    shadow = max(shadowcost(z, labourlevel, household, firm, model.bank.δL, r), zero(price))
    investment = cleaninvestment(q, model.clean)
    welfare = welfarecost(labourlevel, dirtylevel, investment, xc, output, household, dirty, model.clean, model.υ, model.climate.γ)
    profit = dirtyprofit(price, τ, dirtylevel, dirty)
    CommittedAllocation(promote(z, labourlevel, wagelevel, price, demand, dirtylevel, xc, output, liquiditylevel, barL, shadow, vd, q, investment, welfare, profit)...)
end

"Tax sufficient to shut dirty production at every feasible subsequent clean capacity."
function committedtaxbound(model, xc0)
    household = model.household
    firm = model.firm
    n(z) = unconstrainedlabour(z, household, firm, household.r, model.bank.δL)
    z = committedroot(z -> energydemand(z, n(z)) - xc0, one(xc0))
    energyprice(z, n(z), household, firm) / model.dirty.η
end

"Stationary reference at the marginal-investment boundary p=rc, not an imposed transition endpoint."
function committedstationary(model, τ)
    household = model.household
    firm = model.firm
    price = household.r * model.clean.c
    xd = dirtyoutput(price, τ, model.dirty)
    vd = dirtyprofit(price, τ, xd, model.dirty) / household.r
    barL = lendinglimit(vd, model.bank)
    n(z) = labour(z, barL, household, firm, model.bank.δL, household.r)
    z = committedroot(z -> energyprice(z, n(z), household, firm) - price, one(price))
    xc = energydemand(z, n(z)) - xd
    committedallocation(model, xc, vd, model.clean.c, τ; zguess=z)
end
