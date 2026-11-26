abstract type AbstractBank{T <: Real} end

struct Bank{T <: Real} <: AbstractBank{T}
    k̄::T
    θ::T
    θᶜ::T
    κₗ::T
    κᵈ::T
    κᶜ::T
    δᴸ::T
end

function Bank(; calibration = Calibration(), k̄ = calibration.k̄, θ = calibration.θ, θᶜ = calibration.θᶜ, κₗ = calibration.κₗ, κᵈ = calibration.κᵈᵇ, κᶜ = calibration.κᶜᵇ, δᴸ = calibration.δᴸ)
    Bank(promote(k̄, θ, θᶜ, κₗ, κᵈ, κᶜ, δᴸ)...)
end

Bank(calibration::Calibration; kwargs...) = Bank(; calibration, kwargs...)

"Total bank equity, including fixed ownership shares in both energy sectors."
bankequity(vᵈ, vᶜ, bank::Bank) = bank.k̄ + bank.θ * vᵈ + bank.θᶜ * vᶜ

"Minimum equity requirement for loans and energy equity."
liquidityrequirement(L, vᵈ, vᶜ, bank::Bank) = bank.κₗ * L + bank.κᵈ * bank.θ * vᵈ + bank.κᶜ * bank.θᶜ * vᶜ

"Maximum lending supported by the two energy valuations."
lendinglimit(vᵈ, vᶜ, bank::Bank) = (bank.k̄ + (1 - bank.κᵈ) * bank.θ * vᵈ + (1 - bank.κᶜ) * bank.θᶜ * vᶜ) / bank.κₗ
