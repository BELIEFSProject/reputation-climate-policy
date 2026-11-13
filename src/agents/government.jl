import Roots
import ForwardDiff as FD

Base.@kwdef struct Government{T <: Real}
    κᵇ::T = defaultκᵇ
    τ̄::T = defaultτ̄
end

"Signed behavioural cost relative to the conjectured policy, held fixed during a hidden deviation."
behaviouralcost(τ̂, τ, government::Government) = government.κᵇ * (τ̂ - τ) * ((τ̂ + τ) / 2 - government.τ̄)

function κ₀(a, household::LinearHousehold, firm::Firm, climate::Climate)
    -firm.A^2 / (2 * household.ν) + climate.γ * firm.A^4 * (firm.η - a) / household.ν^2
end

function κ₀(a, household::Household, firm::Firm, climate::Climate)
    labour = n(0, a, household, firm)
    output = y(labour, firm)
    return v(labour, household) - output + climate.γ * emissivity(a, firm) * output^2
end
function κ₁(a, household::LinearHousehold, firm::Firm, climate::Climate)
    -2 * climate.γ * firm.A^4 * (firm.η - a)^2 / household.ν^2
end
function κ₂(a, household::LinearHousehold, firm::Firm, climate::Climate)
    firm.A^2 * (firm.η - a)^2 / household.ν + 2 * climate.γ * firm.A^4 * (firm.η - a)^3 / household.ν^2
end

# Hidden deviations leave this resource cost evaluated at firms' expected tax.
# relative=true removes the zero-emissions flow algebraically to preserve numerical precision.
function simplifiedcost(τ, a, ȧ, household::LinearHousehold, firm::Firm, climate::Climate; relative = false)
    if relative
        output, gap = firm.A^2 / household.ν, emissivity(a, firm)
        return k(ȧ, firm) + output * (gap * τ)^2 / 2 + climate.γ * gap * (output * (1 - gap * τ))^2
    end
    k(ȧ, firm) + κ₀(a, household, firm, climate) + κ₁(a, household, firm, climate) * τ + κ₂(a, household, firm, climate) * τ^2 / 2
end

function simplifiedcost(τ, a, ȧ, household::Household, firm::Firm, climate::Climate; relative = false)
    labour = n(τ, a, household, firm)
    output = y(labour, firm)
    if relative && household.ϕ == 2
        referenceoutput = y(n(0, a, household, firm), firm)
        fraction = output / referenceoutput
        distortion = referenceoutput * (emissivity(a, firm) * τ)^2 * (fraction + 2) / (3 * (1 + fraction)^2)
        return k(ȧ, firm) + distortion + climate.γ * emissivity(a, firm) * output^2
    elseif relative
        return k(ȧ, firm) + v(labour, household) - output + climate.γ * emissivity(a, firm) * output^2 - κ₀(firm.η, household, firm, climate)
    end
    return k(ȧ, firm) + v(labour, household) - output + climate.γ * emissivity(a, firm) * output^2
end

function ∂ₐsimplifiedcost(τ, a, _, household::LinearHousehold, firm::Firm, climate::Climate)
    output, gap = firm.A^2 / household.ν, firm.η - a
    return -climate.γ * output^2 * (1 - gap * τ) * (1 - 3gap * τ) - output * gap * τ^2
end

function ∂ₐsimplifiedcost(τ, a, _, household::Household, firm::Firm, climate::Climate)
    output = y(n(τ, a, household, firm), firm)
    return firm.A * emissivity(a, firm) * (2climate.γ * output - τ) * ∂ₐn(τ, a, household, firm) - climate.γ * output^2
end

∂ₜsimplifiedcost(τ, a, household::LinearHousehold, firm::Firm, climate::Climate) = κ₁(a, household, firm, climate) + κ₂(a, household, firm, climate) * τ
∂ₜₜsimplifiedcost(τ, a, household::LinearHousehold, firm::Firm, climate::Climate) = κ₂(a, household, firm, climate)

function ∂ₜsimplifiedcost(τ, a, household::Household, firm::Firm, climate::Climate)
    gap = emissivity(a, firm)
    factor = 1 - gap * τ
    if factor ≤ 0
        return zero(τ + a)
    end
    output = y(n(τ, a, household, firm), firm)
    return gap^2 * output * (τ - 2climate.γ * output) / (household.ϕ * factor)
end

function ∂ₜₜsimplifiedcost(τ, a, household::Household, firm::Firm, climate::Climate)
    gap = emissivity(a, firm)
    factor = 1 - gap * τ
    if factor ≤ 0
        return zero(τ + a)
    end
    output = y(n(τ, a, household, firm), firm)
    elasticity = inv(household.ϕ)
    return elasticity * gap^2 * output / factor * (1 + 2climate.γ * elasticity * gap * output / factor + (1 - elasticity) * gap * (τ - 2climate.γ * output) / factor)
end

"Preferred static tax before imposing the committed announcement."
function statictax(a, household::AbstractHousehold, firm::Firm, climate::Climate)
    gap = emissivity(a, firm)
    if iszero(gap)
        return zero(a)
    end
    target = 2climate.γ * y(n(0, a, household, firm), firm)
    if household isa LinearHousehold
        return clamp(target / (1 + gap * target), 0, inv(firm.η))
    elseif household.ϕ == 2
        return clamp(2target / (sqrt((gap * target)^2 + 4) + gap * target), 0, inv(firm.η))
    end
    residual(τ) = τ - 2climate.γ * y(n(τ, a, household, firm), firm)
    upper = inv(firm.η)
    if residual(upper) ≤ 0
        return upper
    end
    return Roots.find_zero(residual, (zero(target), upper), Roots.Brent())
end
