Base.@kwdef struct Climate{T <: Real}
    γ::T = defaultγ
end

function d(e, climate::Climate)
    climate.γ * e
end
