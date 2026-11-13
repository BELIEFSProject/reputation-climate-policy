## Setup
using Revise

import Printf

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/utils/arguments.jl")

## Calibration
household, firm, government, signal, climate = initmodels()
output = y(n(0, firm.a₀, household, firm), firm)

# Counterfactual private utility gain; it is neither an equilibrium payment nor a resource cost.
τreference = defaultdeviationtax
taxcut = defaultdeviationcut
benefitshare = defaultdeviationbenefit
κᵇ = benefitshare * output / (taxcut * (τreference - government.τ̄ - taxcut / 2))
if !isfinite(κᵇ) || κᵇ ≤ 0
    error("The calibration cut must reduce the distance from the preferred tax.")
end
preference = Government(; κᵇ, τ̄ = government.τ̄)
benefit = -behaviouralcost(τreference - taxcut, τreference, preference)

## Diagnostics
Printf.@printf("Preferred tax: %.2f USD/tCO₂e; behavioural curvature: %.12g in model units\n", preference.τ̄ / taxfactor, preference.κᵇ)
Printf.@printf("A hidden cut from %.2f to %.2f USD/tCO₂e gives %.6g of annual output in private utility (%.3f million USD-equivalent/year).\n", τreference / taxfactor, (τreference - taxcut) / taxfactor, benefit / output, benefit * 1e6)
Printf.@printf("Equilibrium behavioural cost: %.1f\n", behaviouralcost(τreference, τreference, preference))
