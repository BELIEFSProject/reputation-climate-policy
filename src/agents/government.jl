Base.@kwdef struct Government{T <: Real}
    κₚ::T = defaultκₚ
    b::T = defaultbᴾ
end

implementationcost(τ, τᶜ, government::Government) = iszero(τᶜ) ? zero(τ + government.κₚ) : government.κₚ * (τ / τᶜ)^2 / 2
marginaltaxcost(τ, τᶜ, government::Government) = iszero(τᶜ) ? zero(τ + government.κₚ) : government.κₚ * τ / τᶜ^2

κ₀(a, household::LinearHousehold, firm::Firm, climate::Climate) = -firm.A^2 / (2 * household.ν) + climate.γ * firm.A^4 * (firm.η - a) / household.ν^2
κ₁(a, household::LinearHousehold, firm::Firm, climate::Climate) = -2 * climate.γ * firm.A^4 * (firm.η - a)^2 / household.ν^2
κ₂(a, household::LinearHousehold, firm::Firm, climate::Climate) = firm.A^2 * (firm.η - a)^2 / household.ν + 2 * climate.γ * firm.A^4 * (firm.η - a)^3 / household.ν^2

# Evaluate at the actual tax for reduced-form government costs, or expected tax for GE welfare.
simplifiedcost(τ, a, ȧ, household::LinearHousehold, firm::Firm, climate::Climate) = k(ȧ, firm) + κ₀(a, household, firm, climate) + κ₁(a, household, firm, climate) * τ + κ₂(a, household, firm, climate) * τ^2 / 2

function ∂ₐsimplifiedcost(τ, a, ȧ, household::LinearHousehold, firm::Firm, climate::Climate)
    output, gap = firm.A^2 / household.ν, firm.η - a
    return -climate.γ * output^2 * (1 - gap * τ) * (1 - 3gap * τ) - output * gap * τ^2
end

function w(τ, a, ȧ, household::AbstractHousehold, firm::Firm, climate::Climate)
    labour = n(τ, a, household, firm)
    damages = d(e(labour, a, firm), climate)

    return k(a, ȧ, firm) + v(labour, household) - y(labour, firm) * (1 - damages)
end

function statictax(a, household::AbstractHousehold, firm::Firm, climate::Climate)
    iszero(emissivity(a, firm)) && return zero(a)
    maxtax = inv(firm.η)
    residual(τ) = τ - 2d′(climate) * y(n(τ, a, household, firm), firm)
    residual(maxtax) ≤ 0 && return maxtax
    return Roots.find_zero(residual, (zero(maxtax), maxtax), Roots.Brent())
end

function statictax(a, household::LinearHousehold, firm::Firm, climate::Climate)
    ϵ = emissivity(a, firm)
    iszero(ϵ) && return zero(a)
    levycoefficient = 2d′(climate) * y′(firm)^2 / household.ν
    return clamp(levycoefficient / (1 + levycoefficient * ϵ), zero(a), inv(firm.η))
end

function Government(household::AbstractHousehold, firm::Firm, climate::Climate; implementation = defaultimplementation, b = defaultbᴾ)
    0 < implementation < 1 || throw(ArgumentError("The reference implementation fraction must lie between zero and one."))
    τᶜ = statictax(firm.a₀, household, firm, climate)
    referenceemissions = e(n(implementation * τᶜ, firm.a₀, household, firm), firm.a₀, firm)
    κₚ = b * referenceemissions * τᶜ / implementation
    return Government(promote(κₚ, b)...)
end
