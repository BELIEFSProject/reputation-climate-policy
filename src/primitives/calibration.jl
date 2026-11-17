"Calibration targets and model primitives; keyword overrides recompute dependent defaults."
Base.@kwdef struct Calibration{T <: Real}
    taxfactor::T = 1e-3 # USD/tCO₂e to trillion USD/GtCO₂e.
    e₀::T = 3.3
    ẽ₀::T = e₀ / 0.8591885441527447
    y₀::T = 23.035
    r::T = 0.015

    A::T = y₀
    ν::T = A
    ϕ::T = 1.0
    η::T = ẽ₀ / A
    a₀::T = (ẽ₀ - e₀) / A
    γ::T = 2.1994e-3

    τ̄::T = 10taxfactor
    deviationtax::T = 50taxfactor
    deviationcut::T = 10taxfactor
    deviationbenefit::T = 1e-6
    κᵇ::T = deviationbenefit * y₀ / (deviationcut * (deviationtax - τ̄ - deviationcut / 2))

    investmentshare::T = 0.015
    adjustmenthorizon::T = 24.0
    investmentrate::T = (η - a₀) / adjustmenthorizon
    adjustmentweight::T = (adjustmenthorizon + expm1(-r * adjustmenthorizon) / r) / (2adjustmenthorizon)

    c::T = (investmentshare * y₀ / investmentrate - adjustmentweight * γ * y₀^2 / r) / (1 - adjustmentweight)
    κ::T = c / a₀
    ξ::T = 2(investmentshare * y₀ - c * investmentrate) / investmentrate^2

    ϵ::T = 1.0
    σ::T = 0.38
end

Calibration(values::Real...) = Calibration{promote_type(map(typeof, values)...)}(values...)
