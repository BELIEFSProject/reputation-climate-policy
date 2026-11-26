"Constrained and unconstrained labour supply at energy-labour ratio `z`."
function labour(z, L̄, household::AbstractHousehold, firm::AbstractFirm, deltaL, r)
    constrained = ((1 - firm.α) * L̄ / (deltaL * household.ν * f(z, firm)))^inv(1 + inversefrisch(household))
    unconstrained = unconstrainedlabour(z, household, firm, r, deltaL)
    min(constrained, unconstrained)
end

"Wage at cleared labour supply."
wage(z, n, household::AbstractHousehold) = household.ν * n^inversefrisch(household)

"Energy price at a given energy-labour ratio."
function energyprice(z, n, household::AbstractHousehold, firm::CESFirm)
    firm.α / (1 - firm.α) * household.ν * n^inversefrisch(household) * z^(firm.ρ - 1)
end

function energyprice(z, n, household::AbstractHousehold, firm::CobbDouglasFirm)
    firm.α / (1 - firm.α) * household.ν * n^inversefrisch(household) / z
end

function energyprice(z, n, household::AbstractHousehold, firm::LeontiefFirm)
    error("The Leontief energy price is not defined by an interior input ratio.")
end

"Total energy demand."
energydemand(z, n) = z * n

"Working capital required by a firm."
liquidity(n, x, ω, p, deltaL) = deltaL * (ω * n + p * x)

"Working capital expressed using equilibrium wages."
liquidity(z, n, household::AbstractHousehold, firm::AbstractFirm, deltaL) = deltaL * household.ν * n^(1 + inversefrisch(household)) * f(z, firm) / (1 - firm.α)

"Shadow cost of the borrowing constraint."
function shadowcost(z, n, household::LinearHousehold, firm::AbstractFirm, deltaL, r)
    firm.A * (1 - firm.α) * f(z, firm)^((1 - firm.ρ) / firm.ρ) / (deltaL * household.ν * n) - r - inv(deltaL)
end

function shadowcost(z, n, household::Household, firm::AbstractFirm, deltaL, r)
    firm.A * (1 - firm.α) * f(z, firm)^((1 - firm.ρ) / firm.ρ) / (deltaL * household.ν * n^household.ϕ) - r - inv(deltaL)
end

"Dirty energy market supply."
function dirtyoutput(z, n, p, expectedtax, household::AbstractHousehold, firm::AbstractFirm, dirty::DirtyEnergy)
    p = energyprice(z, n, household, firm)
    dirtyoutput(p, expectedtax, dirty)
end

"Dirty energy market supply computed from the equilibrium energy price."
dirtysupply(z, n, expectedtax, household, firm, dirty) = dirtyoutput(energyprice(z, n, household, firm), expectedtax, dirty)

"Energy-market-clearing residual."
energymarketresidual(z, n, xc, expectedtax, household, firm, dirty) = energydemand(z, n) - xc - dirtysupply(z, n, expectedtax, household, firm, dirty)

"Government welfare cost in the general-equilibrium model."
function welfarecost(n, xd, h, investment, xc, y, household::AbstractHousehold, dirty::DirtyEnergy, clean::CleanEnergy, government::Government, γ)
    νcost = household isa LinearHousehold ? household.ν * n^2 / 2 : v(n, household)
    νcost + government.υ * xd^2 / 2 + clean.c * investment + clean.ξ * investment^2 / 2 + dirty.ζ * h^2 / 2 - y * (1 - γ * (dirty.η * xd - h))
end
