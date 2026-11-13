const taxfactor = 1e9 * 1e-12 # USD / tCO2e to trillion USD / GtCO2e

const e₀ = 3.3
const ẽ₀ = e₀ / 0.8591885441527447 # [GtCO2e / year] gross emissions before abatement in 2026
const y₀ = 23.035 # [trillion USD / year]

const realhouseholddiscount = 1.5e-2
const realfirmdiscount = realhouseholddiscount

# The zero-tax allocation is normalised to n(0) = 1, y(0) = y₀, and gross emissions ẽ₀.
const defaultA = y₀
const defaultν = defaultA
const defaultϕᴸ = 1. # Inverse Frisch elasticity.
const defaultηᴱ = ẽ₀ / defaultA
const a₀ = (ẽ₀ - e₀) / defaultA

# Damages
const defaultγ = 2.1994e-3

# A hidden cut from 50 to 40 USD/t gives a private utility gain of one millionth of annual output.
# This is an illustrative preference normalisation, separate from resource costs.
const defaultτ̄ = 10 * taxfactor
const defaultdeviationtax = 50 * taxfactor
const defaultdeviationcut = 10 * taxfactor
const defaultdeviationbenefit = 1e-6
const defaultκᵇ = defaultdeviationbenefit * y₀ / (defaultdeviationcut * (defaultdeviationtax - defaultτ̄ - defaultdeviationcut / 2))

# Illustrative EU targets: 1.5% of output at constant progress from 2026 to 2050.
# Gross investment evidence motivates the scale, rather than identifying net resource costs.
const defaultinvestmentshare = 0.015
const defaultadjustmenthorizon = 24. # Calibration target; the BVP still chooses its stopping time.
const defaultinvestmentrate = (defaultηᴱ - a₀) / defaultadjustmenthorizon
const defaultadjustmentweight = (defaultadjustmenthorizon + expm1(-realfirmdiscount * defaultadjustmenthorizon) / realfirmdiscount) / (2defaultadjustmenthorizon)

# Match investment expenditure and the approximate optimal transition without labour wedges.
const defaultc = (defaultinvestmentshare * y₀ / defaultinvestmentrate - defaultadjustmentweight * defaultγ * y₀^2 / realfirmdiscount) / (1 - defaultadjustmentweight)
const defaultκ = defaultc / a₀ # Retain c(firm) = firm.κ * firm.a₀ and the existing Firm representation.
const defaultξ = 2(defaultinvestmentshare * y₀ - defaultc * defaultinvestmentrate) / defaultinvestmentrate^2

const σ̂ = 0.38
