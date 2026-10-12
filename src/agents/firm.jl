abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    a₀::T = a₀
    A::T = defaultA
    η::T = defaultηᴱ
    κ::T = defaultdietzϕ * y₀ / realfirmdiscount
    ξ::T = defaultξ
end

function y(n, firm::Firm)
    n * firm.A
end
function y′(firm::Firm)
    firm.A
end

function emissivity(a, firm::Firm)
    max(firm.η - a, 0)
end
function emissivity′(a, firm)
    ifelse(firm.η > a, -1, 0)
end

"Firm's wage offer"
function ω(τ, a, firm::Firm)
    firm.A * (1 - emissivity(a, firm) * τ)
end
function ∂ₜω(a, firm::Firm)
    -firm.A * emissivity(a, firm)
end
function ∂ₐω(τ, a, firm::Firm)
    -firm.A * τ * emissivity′(a, firm)
end

function e(n, a, firm::Firm)
    emissivity(a, firm) * y(n, firm)
end

function c(a, firm::Firm)
    firm.κ * a
end

c′(firm::Firm) = firm.κ
c′(_, firm::Firm) = c′(firm)

function k(a, ȧ, firm::Firm)
    c(a, firm) * ȧ + firm.ξ * ȧ^2 / 2
end
function ∂ₐk(ȧ, firm::Firm)
    ȧ * firm.κ
end

## Equilibrium
"Equilibrium labour demand given tax `τ` and abatement `a`."
function n(τ, a, household::AbstractHousehold, firm::Firm)
    n(ω(τ, a, firm), household)
end

function ∂ₜn(τ, a, household::Household, firm::Firm)
    n′(ω(τ, a, firm), household) * ∂ₜω(a, firm)
end
function ∂ₜn(a, household::LinearHousehold, firm::Firm)
    n′(household) * ∂ₜω(a, firm)
end

function ∂ₐn(τ, a, household::Household, firm::Firm)
    n′(ω(τ, a, firm), household) * ∂ₐω(τ, a, firm)
end

function ∂ₐn(τ, a, household::LinearHousehold, firm::Firm)
    n′(household) * ∂ₐω(τ, a, firm)
end

function α(q, a, household::AbstractHousehold, firm::Firm)
    da = ((q / household.r) - c(a, firm)) / firm.ξ

    return max(da, 0)
end