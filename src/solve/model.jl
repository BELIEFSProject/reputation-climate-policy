struct AffineGridMap{T}
    lower::T
    upper::T
end

span(grid::AffineGridMap) = grid.upper - grid.lower
(grid::AffineGridMap)(z) = iszero(z) ? grid.lower : isone(z) ? grid.upper : grid.lower + span(grid) * z
normalise(x, grid::AffineGridMap) = (x - grid.lower) / span(grid)

"General-equilibrium model with household, production, energy, banking and belief objects."
struct GeneralEquilibriumModel{H, F, D, CN, C, B, S, G}
    household::H
    firm::F
    dirty::D
    clean::CN
    climate::C
    bank::B
    signal::S
    government::G
end

function GeneralEquilibriumModel(calibration::Calibration)
    household = Household(calibration)
    firm = CESFirm(calibration)
    dirty = DirtyEnergy(calibration)
    clean = CleanEnergy(calibration)
    climate = Climate(calibration)
    bank = Bank(calibration)
    signal = Signal(calibration)
    government = Government(calibration)
    GeneralEquilibriumModel(household, firm, dirty, clean, climate, bank,
                            signal, government)
end
