abstract type AbstractDirtyEnergy{T <: Real} end

struct DirtyEnergy{T <: Real} <: AbstractDirtyEnergy{T}
    kᵈ::T
    γᵈ::T
    η::T
    ζ::T
end

function DirtyEnergy(; calibration = Calibration(), kᵈ = calibration.kᵈ, γᵈ = calibration.γᵈ, η = calibration.η, ζ = calibration.ζ)
    return DirtyEnergy(promote(kᵈ, γᵈ, η, ζ)...)
end

DirtyEnergy(calibration::Calibration; kwargs...) =
    DirtyEnergy(; calibration, kwargs...)

function dirtyoutput(p, expectedtax, de::DirtyEnergy)
    clamp(max((p - de.η * expectedtax) / de.γᵈ, p / (de.γᵈ + de.ζ * de.η^2)), zero(p + expectedtax), de.kᵈ)
end

"Compliance chosen before the levy increment, from its conditional expected rate."
dirtyabatement(expectedtax, xd, de::DirtyEnergy) = clamp(expectedtax / de.ζ, zero(expectedtax + xd), de.η * xd)

dirtyemissions(expectedtax, xd, de::DirtyEnergy) = de.η * xd - dirtyabatement(expectedtax, xd, de)

"Conditional mean dirty cash flow, including abatement and avoided permit payments."
function dirtyprofit(p, expectedtax, xd, de::DirtyEnergy)
    h = dirtyabatement(expectedtax, xd, de)
    (p - de.η * expectedtax) * xd - de.γᵈ * xd^2 / 2 + expectedtax * h - de.ζ * h^2 / 2
end

"Constant-policy present value of dirty-sector profits."
function dirtyvalue(p, expectedtax, xd, r, de::DirtyEnergy)
    dirtyprofit(p, expectedtax, xd, de) / r
end

"Inverse permit demand at a given energy price; zero prices allow excess issuance."
function permitprice(a, p, de::DirtyEnergy)
    if a < 0
        throw(DomainError(a, "Permit issuance must be non-negative."))
    end
    max(min(de.ζ * (de.η * p - de.γᵈ * a) / (de.γᵈ + de.ζ * de.η^2), de.ζ * (de.η * de.kᵈ - a)), zero(a + p))
end

abstract type AbstractCleanEnergy{T <: Real} end

struct CleanEnergy{T <: Real} <: AbstractCleanEnergy{T}
    c::T
    ξ::T
end

function CleanEnergy(; calibration = Calibration(),
                     c = calibration.cˣ,
                     ξ = calibration.ξˣ)
    return CleanEnergy(promote(c, ξ)...)
end

CleanEnergy(calibration::Calibration; kwargs...) =
    CleanEnergy(; calibration, kwargs...)

function cleaninvestment(qᵉ, ce::CleanEnergy)
    max((qᵉ - ce.c) / ce.ξ, 0)
end

"Constant-policy clean marginal investment value."
cleanmarginalvalue(p, investment, r, ce::CleanEnergy) = p / r

"Clean equity cash flow, after installation and adjustment spending."
cleanprofit(p, xᶜ, investment, clean::CleanEnergy) = p * xᶜ - clean.c * investment - clean.ξ * investment^2 / 2
