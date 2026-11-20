abstract type AbstractFirm{T <: Real} end

struct CESFirm{T <: Real} <: AbstractFirm{T}
    A::T
    α::T
    ρ::T
end

struct CobbDouglasFirm{T <: Real} <: AbstractFirm{T}
    A::T
    α::T
end

struct LeontiefFirm{T <: Real} <: AbstractFirm{T}
    A::T
    α::T
end

const Firm = CESFirm

function CESFirm(; calibration = Calibration(), A = calibration.A,
                 α = calibration.α, ρ = calibration.ρ)
    return CESFirm(promote(A, α, ρ)...)
end

CESFirm(calibration::Calibration; kwargs...) = CESFirm(; calibration, kwargs...)

function CobbDouglasFirm(; calibration = Calibration(),
                         A = calibration.A, α = calibration.α)
    return CobbDouglasFirm(promote(A, α)...)
end

CobbDouglasFirm(calibration::Calibration; kwargs...) =
    CobbDouglasFirm(; calibration, kwargs...)

function LeontiefFirm(; calibration = Calibration(),
                      A = calibration.A, α = calibration.α)
    return LeontiefFirm(promote(A, α)...)
end

LeontiefFirm(calibration::Calibration; kwargs...) =
    LeontiefFirm(; calibration, kwargs...)

function F(n, x, firm::CESFirm)
    firm.A * (firm.α * x^firm.ρ + (1 - firm.α) * n^firm.ρ)^inv(firm.ρ)
end

function F(n, x, firm::CobbDouglasFirm)
    firm.A * x^firm.α * n^(1 - firm.α)
end

function F(n, x, firm::LeontiefFirm)
    firm.A * min(x / firm.α, n / (1 - firm.α))
end

function f(z, firm::CESFirm)
    1 + firm.α * (z^firm.ρ - 1)
end

f(z, firm::CobbDouglasFirm) = z^firm.α

function f(z, firm::LeontiefFirm)
    min(z / firm.α, inv(1 - firm.α))
end

function inputratio(p, ω, firm::CESFirm)
    ((1 - firm.α) / firm.α * p / ω)^inv(firm.ρ - 1)
end

inputratio(p, ω, firm::CobbDouglasFirm) =
    firm.α / (1 - firm.α) * ω / p

inputratio(_, _, ::LeontiefFirm) = error("The Leontief input ratio is not unique.")

function y(n, z, firm::AbstractFirm)
    firm.A * n * f(z, firm)^inv(firm.ρ)
end

function y(n, z, firm::CobbDouglasFirm)
    firm.A * n * f(z, firm)
end

function y(n, z, firm::LeontiefFirm)
    firm.A * n * f(z, firm)
end
