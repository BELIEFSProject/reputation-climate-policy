const CtoCO₂ = 44 / 12
const taxfactor = 1e9 * 1e-12 # USD / tCO2e to trillion USD / GtCO2e
const scctotax = CtoCO₂ * taxfactor # USD / tC to trillion USD / GtC

const ẽ₀ = 41.9 # [GtCO2e / year] gross emissions before abatement in 2026
const e₀ = 36.0
const y₀ = 197.231 # [trillion USD / year]

const realhouseholddiscount = 1.5 / 100
const realfirmdiscount = realhouseholddiscount

# The zero-tax allocation is normalised to n(0) = 1, y(0) = y₀, and gross emissions ẽ₀.
const defaultA = y₀
const defaultν = defaultA
const defaultφᴸ = 1.
const defaultηᴱ = ẽ₀ / defaultA
const a₀ = (ẽ₀ - e₀) / defaultA

const defaultscc = 66 * taxfactor # [trillion USD / GtCO2e]
const dicescc = 3 * 66 / 1000
const defaultγ = 2.1994e-3 # 1.0997e-3

# Illustrative private utility: 80% implementation at the static Pigouvian tax at a₀, not at τ₀.
const defaultbᴾ = 0.01
const defaultimplementation = 0.8
const defaultreferencetax = (2defaultγ * defaultA^2 / defaultν) / (1 + 2defaultγ * defaultA^2 * (defaultηᴱ - a₀) / defaultν)
const defaultreferenceemissions = e₀ * (1 - (defaultηᴱ - a₀) * defaultimplementation * defaultreferencetax)
const defaultκₚ = defaultbᴾ * defaultreferenceemissions * defaultreferencetax / defaultimplementation # [trillion USD-equivalent / year]

const defaultdietzϕ = 3e-5 # Dietz-Venmans MAC slope in abatement flow y₀a
const dietzφ = defaultdietzϕ
const abatementdepreciation = 0.08
const defaultadjustmenthorizon = 70. # [years]
const defaultξ = defaultdietzϕ * y₀^3 * defaultadjustmenthorizon^2 / (1 + realfirmdiscount * defaultadjustmenthorizon)
const τ₀ = 90 * taxfactor

const σ̂ = 0.38
