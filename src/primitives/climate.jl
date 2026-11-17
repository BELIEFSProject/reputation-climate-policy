struct Climate{T <: Real}
    γ::T
end

function Climate(; calibration = Calibration(), γ = calibration.γ)
    return Climate(promote(γ)...)
end

Climate(calibration::Calibration; kwargs...) = Climate(; calibration, kwargs...)


function d(e, climate::Climate)
    climate.γ * e
end
