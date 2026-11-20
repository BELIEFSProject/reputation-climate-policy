abstract type AbstractHousehold{T <: Real} end
struct Household{T} <: AbstractHousehold{T}
    ν::T
    ϕ::T
    r::T

    function Household(; calibration = Calibration(), ν::T = calibration.ν,
                       ϕ::T = calibration.ϕ, r::T = calibration.r) where T
        (ϕ ≈ 1) ? LinearHousehold(ν, r) : new{T}(ν, ϕ, r)
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

Household(calibration::Calibration; kwargs...) = Household(; calibration, kwargs...)

"Unconstrained labour-market-clearing condition at energy-labour ratio `z`."
function unconstrainedlabour(z, household::LinearHousehold, firm::CESFirm, r, deltaL)
    firm.A * (1 - firm.α) * f(z, firm)^((1 - firm.ρ) / firm.ρ) / (household.ν * (1 + r * deltaL))
end

function unconstrainedlabour(z, household::LinearHousehold, firm::CobbDouglasFirm, r, deltaL)
    firm.A * (1 - firm.α) * z^firm.α / (household.ν * (1 + r * deltaL))
end

function unconstrainedlabour(z, household::LinearHousehold, firm::LeontiefFirm, r, deltaL)
    error("The Leontief labour supply at an interior energy-labour ratio is not unique.")
end

function unconstrainedlabour(z, household::Household, firm::CESFirm, r, deltaL)
    (firm.A * (1 - firm.α) * f(z, firm)^((1 - firm.ρ) / firm.ρ) / (household.ν * (1 + r * deltaL)))^inv(household.ϕ)
end

function unconstrainedlabour(z, household::Household, firm::CobbDouglasFirm, r, deltaL)
    (firm.A * (1 - firm.α) * z^firm.α / (household.ν * (1 + r * deltaL)))^inv(household.ϕ)
end

function unconstrainedlabour(z, household::Household, firm::LeontiefFirm, r, deltaL)
    error("The Leontief labour supply at an interior energy-labour ratio is is not unique.")
end
