## Setup
using Revise

import Printf
import JLD2
import Roots

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/dynamics/government.jl")
includet("../src/solve/model.jl")
includet("../src/solve/government/noncommittedmarkov.jl")
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

## Model
household, firm, government, signal, climate = initmodels()
model = Model(household, firm, climate)

## Announced policy
committedinitialtax = nothing # Use τ₀ to anchor the committed announcement.
committedintervals = 500
committedfile = joinpath(@__DIR__, "..", "data", "solutions", committedsolutionfilename(model; initialtax = committedinitialtax, intervals = committedintervals))
if !isfile(committedfile)
    error("Run scripts/committed.jl with the same model and numerical settings first: $committedfile")
end
announcedtaxpath = JLD2.jldopen(committedfile, "r") do file
    file["committedtaxpath"]
end
noncommittedhorizon = last(announcedtaxpath.ts) # The stationary tail is solved at this date.

Printf.@printf("Constant installation price c = %.6g; initial κ₀ = %.6g, κ₁ = %.6g, κ₂ = %.6g\n", c(firm), κ₀(firm.a₀, household, firm, climate), κ₁(firm.a₀, household, firm, climate), κ₂(firm.a₀, household, firm, climate))
Printf.@printf("Announced initial tax: %.3f USD/tCO₂\n", first(announcedtaxpath.taxpath) / taxfactor)

## Solve
# h is shared by the two normalised state directions; time steps are calendar years.
noncommittedgridstep = 1 / 40
noncommittedtimestep = 1.
noncommittedholdingtime = 1.

noncommittedsolution = solvenoncommitted(announcedtaxpath, model, government, signal; h = noncommittedgridstep, timestep = noncommittedtimestep, maxholdingtime = noncommittedholdingtime, horizon = noncommittedhorizon, verbose = 1)

## Diagnostics
noncommittedinitialvalues = first(noncommittedsolution.u)
noncommittedvaluegap = maximum(x -> x.valuegap, noncommittedsolution.diagnostics)
noncommittedbellmangap = maximum(x -> x.bellmangap, noncommittedsolution.diagnostics)
noncommittedpolicygap = maximum(x -> x.policygap, noncommittedsolution.diagnostics)
noncommittedmaximumimprovement = maximum(x -> x.policyimprovement, noncommittedsolution.diagnostics)
noncommittedtaxrange = extrema(noncommittedinitialvalues.τ) ./ taxfactor
noncommittedvaluationrange = extrema(noncommittedinitialvalues.q)

Printf.@printf("Initial tax [%.3f, %.3f] USD/tCO₂; expected investment value [%.6g, %.6g]\n", first(noncommittedtaxrange), last(noncommittedtaxrange), first(noncommittedvaluationrange), last(noncommittedvaluationrange))
Printf.@printf("Maximum value, Bellman and policy residuals: %.3e, %.3e, %.3e\n", noncommittedvaluegap, noncommittedbellmangap, noncommittedpolicygap)
Printf.@printf("Maximum hidden-deviation improvement: %.3e output/year\n", noncommittedmaximumimprovement)

## Save
noncommittedkey = noncommittedsolutionkey(noncommittedsolution)
noncommittedfile = joinpath(@__DIR__, "..", "data", "solutions", "noncommitted-simplified-$(last(split(noncommittedkey, '/'))).jld2")
if isfile(noncommittedfile)
    error("Non-committed result already exists at $noncommittedfile; preserving the saved solution.")
end
mkpath(dirname(noncommittedfile))
JLD2.jldopen(noncommittedfile, "w") do file
    file["model"] = model
    file["committedtaxpath"] = announcedtaxpath
    file["$noncommittedkey/solution"] = noncommittedsolution
    file["$noncommittedkey/government"] = government
    file["$noncommittedkey/signal"] = signal
end
@info "Saved non-committed equilibrium" noncommittedfile noncommittedkey
