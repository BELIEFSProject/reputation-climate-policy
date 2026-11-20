abstract type AbstractBank{T <: Real} end

struct Bank{T <: Real} <: AbstractBank{T}
    k̄::T
    θ::T
    κₗ::T
    κᵈ::T
end

function Bank(; calibration = Calibration(), k̄ = calibration.k̄,
              θ = calibration.θ, κₗ = calibration.κₗ, κᵈ = calibration.κᵈᵇ)
    Bank(promote(k̄, θ, κₗ, κᵈ)...)
end

Bank(calibration::Calibration; kwargs...) = Bank(; calibration, kwargs...)

"Total bank equity."
bankequity(k̄, θ, vd) = k̄ + θ * vd

"Minimum equity requirement for loans and dirty assets."
liquidityrequirement(L, θ, vd, bank::Bank) =
    bank.κₗ * L + bank.κᵈ * θ * vd

"Maximum feasible lending at given dirty-sector value."
lendinglimit(vd, bank::Bank) =
    (bank.k̄ + (1 - bank.κᵈ) * bank.θ * vd) / bank.κₗ
