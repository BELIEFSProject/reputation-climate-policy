const CtoCO₂ = 44 / 12
const CtoCO2 = CtoCO₂
const taxfactor = 1e9 * 1e-12 # USD / tCO2e to trillion USD / GtCO2e
const scctotax = CtoCO₂ * taxfactor # USD / tC to trillion USD / GtC

const ẽ₀ = 41.9 # [GtCO2e / year] gross emissions before abatement in 2026
const e₀ = 36.0
const y₀ = 197.231 # [trillion USD / year]

const realhouseholddiscount = 1e-2
const realfirmdiscount = realhouseholddiscount

# The zero-tax allocation is normalised to n(0) = 1, y(0) = y₀, and gross emissions ẽ₀.
const defaultA = y₀
const defaultν = defaultA
const defaultφᴸ = 1.
const defaultηᴱ = ẽ₀ / defaultA
const a₀ = (ẽ₀ - e₀) / defaultA

const defaultscc = 66 * taxfactor # [trillion USD / GtCO2e]
const dicescc = 3 * 66 / 1000
const defaultdietzϕ = 3e-5 # Dietz-Venmans MAC slope ϕ
const dietzφ = defaultdietzϕ
const abatementdepreciation = 2e-2
const defaultadjustmenthorizon = 30. # [years]
# [tUSD year³ / GtCO2e²]
const defaultξ = defaultdietzϕ * y₀ * defaultadjustmenthorizon^2 / (
    1 + realfirmdiscount * defaultadjustmenthorizon
)
const τ₀ = 50 * taxfactor

const σ̂ = 0.38
