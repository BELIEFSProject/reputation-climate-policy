struct Government{T <: Real}
    κᵇ::T
    τ̄::T
    υ::T # Welfare weight on dirty operating costs; γᵈ gives the resource-cost benchmark.
    ψ::T
end

function Government(; calibration = Calibration(), κᵇ = calibration.κᵇ, τ̄ = calibration.τ̄, υ = calibration.υ, ψ = calibration.ψ)
    Government(promote(κᵇ, τ̄, υ, ψ)...)
end

Government(calibration::Calibration; kwargs...) =
    Government(; calibration, kwargs...)

"Firms' expected tax under public belief `φ`."
expectedtax(τᶜ, τⁿᶜ, φ) = φ * τᶜ + (1 - φ) * τⁿᶜ

"Private flow cost, excluded from household welfare and zero along the committed policy."
policycost(τ, τᶜ, government::Government) = government.ψ * (τ - τᶜ)^2 / 2

"Diffusion coefficient for posterior beliefs."
function beliefdiffusion(φ, τᶜ, τⁿᶜ, signal::Signal)
    φ * (1 - φ) * signal.ϵ / signal.σ * (τᶜ - τⁿᶜ)
end

"Drift of posterior beliefs under a hidden policy."
function beliefdrift(τ̂, expectedtax, σφ, signal::Signal)
    signal.ϵ / signal.σ * σφ * (τ̂ - expectedtax)
end
