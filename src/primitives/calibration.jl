"Calibration targets and model primitives; keyword overrides recompute dependent defaults."
Base.@kwdef struct Calibration{T <: Real}
    taxfactor::T = 1e-3 # USD/tCO₂e to trillion USD/GtCO₂e.
    r::T = 0.04
    ν::T = 0.70
    ϕ::T = 1.0
    A::T = 1.0
    α::T = 0.30
    ρ::T = -0.50
    γ::T = 0.15
    τ̄::T = 0.25
    κᵇ::T = 1.0

    ϵ::T = 1.0
    σ::T = 0.38

    # General-equilibrium stranded-assets calibration.
    kᵈ::T = 1.0
    γᵈ::T = 0.25
    η::T = 1.0
    cˣ::T = 4.0
    ξˣ::T = 100.0
    υ::T = 0.25
    δL::T = 0.25
    κₗ::T = 0.10
    κᵈᵇ::T = 0.20
    θ::T = 0.01
    k̄::T = 0.018
end

Calibration(values::Real...) = Calibration{promote_type(map(typeof, values)...)}(values...)
