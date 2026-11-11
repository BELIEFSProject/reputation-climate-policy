const taxfactor = 1e9 * 1e-12 # USD / tCO2e to trillion USD / GtCO2e

const e₀ = 3.3
const ẽ₀ = e₀ / 0.8591885441527447 # [GtCO2e / year] gross emissions before abatement in 2026
const y₀ = 23.035 # [trillion USD / year]

const realhouseholddiscount = 1.5e-2
const realfirmdiscount = realhouseholddiscount

# The zero-tax allocation is normalised to n(0) = 1, y(0) = y₀, and gross emissions ẽ₀.
const defaultA = y₀
const defaultν = defaultA
const defaultφᴸ = 1.
const defaultηᴱ = ẽ₀ / defaultA
const a₀ = (ẽ₀ - e₀) / defaultA

# Damages
const defaultγ = 2.1994e-3

# Legacy metadata defaults retained for compatibility with saved solutions.
const defaultbᴾ = 0.01
const defaultimplementation = 0.8
const defaultreferencetax = (2defaultγ * defaultA^2 / defaultν) / (1 + 2defaultγ * defaultA^2 * (defaultηᴱ - a₀) / defaultν)
const defaultreferenceemissions = e₀ * (1 - (defaultηᴱ - a₀) * defaultimplementation * defaultreferencetax)
const defaultκₚ = defaultbᴾ * defaultreferenceemissions * defaultreferencetax / defaultimplementation # [trillion USD-equivalent / year]

const defaultdietzϕ = 3e-5 # Dietz-Venmans MAC slope in abatement flow y₀a
const defaultadjustmenthorizon = 70. # [years]
const defaultξ = defaultdietzϕ * y₀^3 * defaultadjustmenthorizon^2 / (1 + realfirmdiscount * defaultadjustmenthorizon)

const σ̂ = 0.38
