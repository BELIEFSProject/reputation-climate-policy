# Metadata retained for saved solution compatibility; current objectives use the κ coefficients below.
Base.@kwdef struct Government{T <: Real}
    κₚ::T = defaultκₚ
    b::T = defaultbᴾ
end

function κ₀(a, household::LinearHousehold, firm::Firm, climate::Climate)
    -firm.A^2 / (2 * household.ν) + climate.γ * firm.A^4 * (firm.η - a) / household.ν^2
end
function κ₁(a, household::LinearHousehold, firm::Firm, climate::Climate)
    -2 * climate.γ * firm.A^4 * (firm.η - a)^2 / household.ν^2
end
function κ₂(a, household::LinearHousehold, firm::Firm, climate::Climate)
    firm.A^2 * (firm.η - a)^2 / household.ν + 2 * climate.γ * firm.A^4 * (firm.η - a)^3 / household.ν^2
end

# Evaluate at the actual tax for reduced-form government costs, or expected tax for GE welfare.
function simplifiedcost(τ, a, ȧ, household::LinearHousehold, firm::Firm, climate::Climate)
    k(ȧ, firm) + κ₀(a, household, firm, climate) + κ₁(a, household, firm, climate) * τ + κ₂(a, household, firm, climate) * τ^2 / 2
end

function ∂ₐsimplifiedcost(τ, a, _, household::LinearHousehold, firm::Firm, climate::Climate)
    output, gap = firm.A^2 / household.ν, firm.η - a
    return -climate.γ * output^2 * (1 - gap * τ) * (1 - 3gap * τ) - output * gap * τ^2
end
