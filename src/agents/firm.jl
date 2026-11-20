abstract type AbstractFirm{T <: Real} end

struct Firm{T} <: AbstractFirm{T}
    A::T # Total factor poductivity
    α::T # Energy share
    ρ::T # CES elasticity of substitution
end

function F(n, x, firm::Firm)
    @unpack A, α, ρ = firm

    return A * (α * x^ρ + (1 - α) * n^ρ)^inv(ρ)
end

function f(z, firm::Firm)
    @unpack α, ρ = firm

    return 1 + α * (z^ρ - 1)
end
