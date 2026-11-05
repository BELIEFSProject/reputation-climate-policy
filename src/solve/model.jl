abstract type AbstractModel end

struct Model{H <: Household, F, C} <: AbstractModel
    household::H
    firm::F
    climate::C
end

"Model parameters specialised to linear labour supply, corresponding to `ϕ = 1`."
struct LinearLabourModel{H <: LinearHousehold, F, C} <: AbstractModel
    household::H
    firm::F
    climate::C
end

function Model(household::H, firm::F, climate::C) where {H, F, C}
    isa(household, LinearHousehold) ? 
        LinearLabourModel{H, F, C}(household, firm, climate) :
        Model{H, F, C}(household, firm, climate)
end

struct AffineGridMap{T}
    lower::T
    upper::T
end

span(grid::AffineGridMap) = grid.upper - grid.lower
(grid::AffineGridMap)(z) = iszero(z) ? grid.lower : isone(z) ? grid.upper : grid.lower + span(grid) * z
normalise(x, grid::AffineGridMap) = (x - grid.lower) / span(grid)
