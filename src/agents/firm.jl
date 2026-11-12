abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    a₀::T = a₀
    A::T = defaultA
    η::T = defaultηᴱ
    κ::T = defaultκ
    ξ::T = defaultξ
end

function y(n, firm::Firm)
    n * firm.A
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
function ∂ₐω(τ, a, firm::Firm)
    -firm.A * τ * emissivity′(a, firm)
end

function e(n, a, firm::Firm)
    emissivity(a, firm) * y(n, firm)
end

# Constant installation price in the simplified model.
c(firm::Firm) = firm.κ * firm.a₀
k(ȧ, firm::Firm) = c(firm) * ȧ + firm.ξ * ȧ^2 / 2

## Equilibrium
"Equilibrium labour demand given tax `τ` and abatement `a`."
function n(τ, a, household::AbstractHousehold, firm::Firm)
    wage = ω(τ, a, firm)
    n(max(wage, zero(wage)), household)
end

function ∂ₐn(τ, a, household::Household, firm::Firm)
    ω(τ, a, firm) ≤ 0 && return zero(τ + a)
    n′(ω(τ, a, firm), household) * ∂ₐω(τ, a, firm)
end

function ∂ₐn(τ, a, household::LinearHousehold, firm::Firm)
    ω(τ, a, firm) ≤ 0 && return zero(τ + a)
    n′(household) * ∂ₐω(τ, a, firm)
end

α(q, firm::Firm) = max(q - c(firm), zero(q)) / firm.ξ
