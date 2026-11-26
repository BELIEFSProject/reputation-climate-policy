## Physical states and calendar time

struct CommittedParameters{M, T}
    model::M
    xᶜ₀::T
    zguess::T
end

"Physical state–costate RHS, ordered as (xᶜ,q,vᵈ,vᶜ,λxᶜ,λq,λvᵈ,λvᶜ)."
function dynamics(u::TU, p::CommittedParameters, t) where TU <: SA.StaticVector{8}
    model = p.model
    r = model.household.r
    xᶜ, q, vᵈ, vᶜ, λxᶜ, λq, λvᵈ, λvᶜ = u
    a, ∂w, ∂p, ∂π = committedpolicy(model, xᶜ, vᵈ, vᶜ, q, λq, λvᵈ, λvᶜ; zguess=p.zguess)
    ∂πᶜ = xᶜ * ∂p + SA.SVector(a.p, 0, 0, 0)
    ẋᶜ = a.investment
    q̇ = r * q - a.p
    v̇ᵈ = r * vᵈ - a.profit
    v̇ᶜ = r * vᶜ - cleanprofit(a.p, xᶜ, a.investment, model.clean)
    λ̇xᶜ = r * λxᶜ - ∂w[1] + λq * ∂p[1] + λvᵈ * ∂π[1] + λvᶜ * ∂πᶜ[1]
    # Green investment spending enters both welfare costs and green equity cash flows.
    # At q=c use the inactive limiting derivative.
    λ̇q = q > model.clean.c ? -((1 + λvᶜ) * q + λxᶜ) / model.clean.ξ : zero(q)
    λ̇vᵈ = -∂w[2] + λq * ∂p[2] + λvᵈ * ∂π[2] + λvᶜ * ∂πᶜ[2]
    λ̇vᶜ = -∂w[3] + λq * ∂p[3] + λvᵈ * ∂π[3] + λvᶜ * ∂πᶜ[3]

    SA.SVector(ẋᶜ, q̇, v̇ᵈ, v̇ᶜ, λ̇xᶜ, λ̇q, λ̇vᵈ, λ̇vᶜ)
end

function initialboundary(u::TU, p::CommittedParameters) where TU <: SA.StaticVector{8}
    SA.SVector(u[1] - p.xᶜ₀, u[6], committedcomplementarity(u[7], u[3]), u[8])
end

"Endpoint multipliers on non-negative tax and zero investment, without dividing by emissions."
function committedterminal(u::SA.StaticVector{8}, p::CommittedParameters)
    model = p.model
    xᶜ, q, vᵈ, vᶜ, λxᶜ, λq, λvᵈ, λvᶜ = u
    a, ∂f, ∂g, ∂τ = committedtail(model, xᶜ, vᵈ, q)
    wedge = ∂g - SA.SVector(λxᶜ + q * λvᶜ, λvᵈ, λq + xᶜ * λvᶜ)
    if iszero(∂τ[1])
        throw(DomainError(∂τ[1], "The stationary endpoint price chart is singular."))
    end
    # The profit envelope gives ∂f=ev-xᵈ*eq+(e/r)∂τ; cancel these terms analytically.
    valuationmultiplier = -wedge[2] + wedge[1] * ∂τ[2] / ∂τ[1]
    taxmultiplier = wedge[1] / ∂τ[1] + a.e * valuationmultiplier / model.household.r
    investmentmultiplier = -wedge[3] + a.xᵈ * valuationmultiplier + wedge[1] * ∂τ[3] / ∂τ[1]

    a, taxmultiplier, investmentmultiplier
end

"Terminal valuation and KKT conditions for a stationary tail with partial compliance."
function terminalboundary(u::TU, p::CommittedParameters) where TU <: SA.StaticVector{8}
    a, taxmultiplier, investmentmultiplier = committedterminal(u, p)

    SA.SVector(u[3] - a.profit / p.model.household.r, u[4] - u[2] * u[1], committedcomplementarity(taxmultiplier, a.τ), committedcomplementarity(investmentmultiplier, p.model.clean.c - u[2]))
end
