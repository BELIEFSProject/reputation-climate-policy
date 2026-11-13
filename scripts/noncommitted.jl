## Setup
using Revise

import Printf
import JLD2
import ArgParse

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

## Arguments
argumentsettings = parameterargumentsettings()
ArgParse.@add_arg_table! argumentsettings begin
    "--h"
        arg_type = Float64
        default = 1 / 80
        help = "Shared normalised grid step in beliefs and abatement; must divide [0,1]."
    "--timestep"
        arg_type = Float64
        default = 0.05
        help = "Maximum backward implicit time step in calendar years."
end
arguments = ArgParse.parse_args(ARGS, argumentsettings; as_symbols = true)

## Model
household, firm, government, signal, climate = initmodels(arguments)
model = Model(household, firm, climate)
@info "Non-committed preferences" preferredtaxUSD = government.τ̄ / taxfactor curvature = government.κᵇ

## Announced policy
committedinitialtax = 10 * taxfactor # Set an inherited tax to anchor the committed announcement.
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
h = arguments[:h]
timestep = arguments[:timestep]
savestep = 1.0 # Calendar years between saved values; independent of the internal time step.
horizon = last(τᶜ.ts) # The stationary tail is solved at this date.
tolerance = 1e-6
policytolerance = 1e-6

ncsolution = solvencdae(τᶜ, model, government, signal; h, timestep, savestep, horizon, tolerance, policytolerance, verbose = 1);

## Diagnostics
ncinitialvalues = first(ncsolution.u)
ncvaluegap = maximum(x -> x.valuegap, ncsolution.diagnostics)
ncbellmangap = maximum(x -> x.bellmangap, ncsolution.diagnostics)
ncpolicygap = maximum(x -> x.policygap, ncsolution.diagnostics)
ncmaximumimprovement = maximum(x -> x.policyimprovement, ncsolution.diagnostics)
nctaxrange = extrema(ncinitialvalues.τ) ./ taxfactor
ncvaluationrange = extrema(ncinitialvalues.q)

Printf.@printf("Initial tax [%.3f, %.3f] USD/tCO₂; expected investment value [%.6g, %.6g]\n", first(nctaxrange), last(nctaxrange), first(ncvaluationrange), last(ncvaluationrange))
Printf.@printf("Maximum value, Bellman and policy residuals: %.3e, %.3e, %.3e\n", ncvaluegap, ncbellmangap, ncpolicygap)
Printf.@printf("Maximum hidden-deviation improvement: %.3e trillion USD-equivalent/year\n", ncmaximumimprovement)

## Save
nckey = ncsolutionkey(ncsolution)
ncfile = joinpath(@__DIR__, "..", "data", "solutions", "noncommitted-simplified-$(first(last(split(nckey, '/')), solutionhashlength)).jld2")
if isfile(ncfile)
    error("Non-committed result already exists at $ncfile; preserving the saved solution.")
end
mkpath(dirname(ncfile))
JLD2.jldopen(ncfile, "w") do file
    file["ncversion"] = ncversion
    file["model"] = model
    file["committedtaxpath"] = τᶜ
    file["$nckey/solution"] = ncsolution
    file["$nckey/government"] = government
    file["$nckey/signal"] = signal
end
@info "Saved non-committed equilibrium" ncfile nckey
