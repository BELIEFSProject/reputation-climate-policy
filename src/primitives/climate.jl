Base.@kwdef struct Climate{T <: Real}
    γ::T = 1.0997e-3
end

function d(e, climate::Climate)
    climate.γ * e
end
function d′(climate::Climate)
    climate.γ
end