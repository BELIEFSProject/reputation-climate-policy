abstract type AbstractDirtyEnergy{T <: Real} end

struct DirtyEnergy{T}
    kᵈ::T
    γᵈ::T
    η::T
end

function dirtyoutput(p, τᵉ, de::DirtyEnergy)
    clamp((p - de.η * τᵉ) / de.γᵈ, 0, de.kᵈ)
end

abstract type AbstractCleanEnergy{T <: Real} end

struct CleanEnergy{T}
    c::T
    ξ::T
end

function cleaninvestment(qᵉ, ce::CleanEnergy)
    max((qᵉ - ce.c) / ce.ξ, 0)
end