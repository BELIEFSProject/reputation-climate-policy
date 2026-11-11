## Setup
using Revise

import Printf
import JLD2

import DifferentiationInterface as DI
import ForwardDiff as FD
import SparseArrays as SP
import SparseMatrixColorings as SMC

import ADTypes
import LinearSolve as LS
import NonlinearSolve as NLS
import SciMLBase

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/dynamics/government.jl")

includet("../src/solve/model.jl")
includet("../src/solve/government/noncommittedmarkov.jl")
includet("../src/solve/government/noncommitteddae.jl")
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

## Model
household, firm, government, signal, climate = initmodels()
model = Model(household, firm, climate)

## Announced policy
committedinitialtax = nothing # Set an inherited tax to anchor the committed announcement.
committedintervals = 500
committedfile = joinpath(@__DIR__, "..", "data", "solutions", committedsolutionfilename(model; initialtax = committedinitialtax, intervals = committedintervals))
if !isfile(committedfile)
    error("Run scripts/committed.jl with the same model and numerical settings first: $committedfile")
end

τᶜ = JLD2.jldopen(committedfile, "r") do file
    file["committedtaxpath"]
end

## Solve
# h is shared by both normalised states; announcement knots further split the implicit time steps.
h = 1 / 80
timestep = 0.1
savestep = 1.0 # Calendar years between saved values; independent of the internal time step.
horizon = last(τᶜ.ts) # The stationary tail is solved at this date.

noncommittedsolution = solvenoncommitteddae(τᶜ, model, government, signal; h, timestep, savestep, horizon, verbose = 1);

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
noncommittedfile = joinpath(@__DIR__, "..", "data", "solutions", "noncommitted-simplified-$(first(last(split(noncommittedkey, '/')), solutionhashlength)).jld2")
if isfile(noncommittedfile)
    error("Non-committed result already exists at $noncommittedfile; preserving the saved solution.")
end
mkpath(dirname(noncommittedfile))
JLD2.jldopen(noncommittedfile, "w") do file
    file["model"] = model
    file["committedtaxpath"] = τᶜ
    file["$noncommittedkey/solution"] = noncommittedsolution
    file["$noncommittedkey/government"] = government
    file["$noncommittedkey/signal"] = signal
end
@info "Saved non-committed equilibrium" noncommittedfile noncommittedkey
