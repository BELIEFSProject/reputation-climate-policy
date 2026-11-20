struct Government{T <: Real}
    κᵇ::T
    τ̄::T
end

function Government(; calibration = Calibration(), κᵇ = calibration.κᵇ,
                    τ̄ = calibration.τ̄)
    Government(promote(κᵇ, τ̄)...)
end

Government(calibration::Calibration; kwargs...) =
    Government(; calibration, kwargs...)

"Firms' expected tax under public belief `φ`."
expectedtax(τᶜ, τⁿᶜ, φ) = φ * τᶜ + (1 - φ) * τⁿᶜ

"Diffusion coefficient for posterior beliefs."
function beliefdiffusion(φ, τᶜ, τⁿᶜ, signal::Signal)
    φ * (1 - φ) * signal.ϵ / signal.σ * (τᶜ - τⁿᶜ)
end

"Drift of posterior beliefs under a hidden policy."
function beliefdrift(τ̂, τᵉ, σφ, signal::Signal)
    signal.ϵ / signal.σ * σφ * (τ̂ - τᵉ)
end
