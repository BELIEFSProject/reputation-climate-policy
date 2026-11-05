## Setup
using Revise

import UnPack: @unpack

import Printf
import JLD2
import Roots
import BoundaryValueDiffEq as BVP
import SciMLBase
import QuadGK as QGK

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/dynamics/government.jl")
includet("../src/solve/model.jl")
includet("../src/solve/government/committed.jl")
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

## Model
household, firm, government, signal, climate = initmodels()
model = Model(household, firm, climate)

## Solve
initialtax = nothing
intervals = 500
filename = committedsolutionfilename(model; initialtax, intervals)
committedfile = joinpath(@__DIR__, "..", "data", "solutions", filename)

if isfile(committedfile)
    error("Committed benchmark already exists at $committedfile; preserving the saved solution.")
end

committedsolution = solvecommitted(model; initialtax, intervals)
committedtaxpath = CommittedTaxPath(committedsolution)

## Diagnostics
committedtaxtimes = committedtaxpath.ts
committedtaxes = committedtaxpath.taxpath ./ taxfactor # Repeated times display the tax jump.
committedabatementrange = extrema(committedsolution.a)
committedinvestmentrange = extrema(committedsolution.investment)
committedtaxrange = extrema(committedtaxes)
committedemissionspath = [e(n(τ, a, household, firm), a, firm) for (τ, a) in zip(committedsolution.tax, committedsolution.a)]
committedlevypath = committedsolution.tax .* committedemissionspath

Printf.@printf("Abatement [%.6g, %.6g]; investment [%.6g, %.6g]; tax [%.3f, %.3f] USD/tCO₂\n", first(committedabatementrange), last(committedabatementrange), first(committedinvestmentrange), last(committedinvestmentrange), first(committedtaxrange), last(committedtaxrange))
Printf.@printf("Full abatement at calendar time %.6g; welfare cost %.8g; boundary residual %.3e\n", committedsolution.stoppingtime, committedsolution.welfarecost, committedsolution.boundarygap)
Printf.@printf("Initial valuation %.6g; initial tax %.3f USD/tCO₂; contact tax %.3f → %.3f USD/tCO₂; return code %s\n", first(committedsolution.q), first(committedsolution.tax) / taxfactor, committedtaxpath(committedsolution.stoppingtime; side = :left) / taxfactor, committedtaxpath(committedsolution.stoppingtime) / taxfactor, string(committedsolution.retcode))

## Save
mkpath(dirname(committedfile))
JLD2.jldopen(committedfile, "w") do file
    file["model"] = model
    file["committedtaxpath"] = committedtaxpath
    file["committed/solution"] = committedsolution
    file["committed/welfarecost"] = committedsolution.welfarecost
end
@info "Saved committed benchmark" committedfile
