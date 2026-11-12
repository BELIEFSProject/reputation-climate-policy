abstract type AbstractHousehold{T <: Real} end
struct Household{T} <: AbstractHousehold{T}
    ν::T
    ϕ::T
    r::T

    function Household(; ν::T = defaultν, ϕ::T = defaultφᴸ, r::T = realhouseholddiscount) where T
        isone(ϕ) ? LinearHousehold(ν, r) : new{T}(ν, ϕ, r)
    end

    function Household(ν, ϕ, r)
        Household(; ν, ϕ, r)
    end
end


function v(n, household::Household)
    household.ν * n^(1 + household.ϕ) / (1 + household.ϕ)
end

"Labour supply given wage `ω`"
function n(ω, household::Household)
    (ω / household.ν)^inv(household.ϕ)
end
function n′(ω, household::Household)
    ((ω / household.ν)^(1 / household.ϕ - 1)) / (household.ϕ * household.ν)
end

# Specialise unit-elastic labour supply.
struct LinearHousehold{T} <: AbstractHousehold{T}
    ν::T
    r::T
end

function v(n, household::LinearHousehold)
    household.ν * n^2 / 2
end

function n(ω, household::LinearHousehold)
    ω / household.ν
end
function n′(household::LinearHousehold)
    inv(household.ν)
end

inversefrisch(household::Household) = household.ϕ
inversefrisch(::LinearHousehold) = 1
