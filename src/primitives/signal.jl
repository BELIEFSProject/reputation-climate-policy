struct Signal{T <: Real}
    ϵ::T
    σ::T
end

function Signal(; calibration = Calibration(), ϵ = calibration.ϵ, σ = calibration.σ)
    return Signal(promote(ϵ, σ)...)
end

Signal(calibration::Calibration; kwargs...) = Signal(; calibration, kwargs...)


μ(τ, signal::Signal) = signal.ϵ * τ