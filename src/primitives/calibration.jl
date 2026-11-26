"Illustrative annual calibration and model primitives; keyword overrides recompute dependent defaults."
Base.@kwdef struct Calibration{T <: Real}
    taxfactor::T = 1e-3 # USD/tCO₂e to trillion USD/GtCO₂e.
    r::T = 0.04
    ν::T = 0.70
    ϕ::T = 1.0
    A::T = 1.0
    α::T = 0.30
    ρ::T = -0.50
    γ::T = 0.08
    τ̄::T = 0.25
    κᵇ::T = 1.0
    ψ::T = 0.10 # Policymaker's private cost of departure from the announcement.

    ϵ::T = 1.0
    σ::T = 0.38

    # General-equilibrium stranded-assets calibration.
    kᵈ::T = 1.0
    γᵈ::T = 0.25
    η::T = 1.0
    ζ::T = 1.0 # Intratemporal compliance-abatement cost.
    cˣ::T = 4.0
    ξˣ::T = 100.0
    υ::T = γᵈ # Government weight on dirty operating costs.
    δᴸ::T = 0.25
    κₗ::T = 0.10
    κᵈᵇ::T = 0.20
    θ::T = 0.005
    θᶜ::T = θ # Common ownership share in the current model.
    κᶜᵇ::T = 0.10
    k̄::T = 0.008 # Non-energy bank equity, denoted v̄ in the note.
end
