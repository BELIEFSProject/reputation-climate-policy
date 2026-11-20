abstract type AbstractDirtyEnergy{T <: Real} end

struct DirtyEnergy{T <: Real} <: AbstractDirtyEnergy{T}
    kᵈ::T
    γᵈ::T
    η::T
end

function DirtyEnergy(; calibration = Calibration(),
                     kᵈ = calibration.kᵈ,
                     γᵈ = calibration.γᵈ,
                     η = calibration.η)
    return DirtyEnergy(promote(kᵈ, γᵈ, η)...)
end

DirtyEnergy(calibration::Calibration; kwargs...) =
    DirtyEnergy(; calibration, kwargs...)

function dirtyoutput(p, τᵉ, de::DirtyEnergy)
    clamp((p - de.η * τᵉ) / de.γᵈ, 0, de.kᵈ)
end

"Dirty-sector profit at a given utilisation rate."
function dirtyprofit(p, τᵉ, xd, de::DirtyEnergy)
    (p - de.η * τᵉ) * xd - de.γᵈ * xd^2 / 2
end

"Constant-policy present value of dirty-sector profits."
function dirtyvalue(p, τᵉ, xd, r, de::DirtyEnergy)
    dirtyprofit(p, τᵉ, xd, de) / r
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
cleanmarginalvalue(p, xc_dot, r, ce::CleanEnergy) =
    (p - ce.c * xc_dot) / r