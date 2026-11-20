## Setup
using Revise

import Roots
import ForwardDiff as FD
import ADTypes
import ArgParse
import SHA

import UnPack: @unpack
import FastClosures: @closure

import Printf
import JLD2
import BoundaryValueDiffEq as BVP
import SciMLBase
import QuadGK as QGK

includet("../src/primitives/calibration.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/energy.jl")
includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/solve/model.jl")
includet("../src/dynamics/government.jl")

includet("../src/solve/government/committed.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

## Model
calibration = Calibration()
household, firm, government, signal, climate = initmodels(; calibration)
model = Model(household, firm, climate)

## Solve
τ₀ = 10 * calibration.taxfactor
intervals = 500
filename = committedsolutionfilename(model; initialtax = τ₀, intervals)
filepath = joinpath("data", "solutions", filename)

if isfile(filepath)
    error("Committed benchmark already exists at $filepath; preserving the saved solution.")
end

solution = solvecommitted(model; initialtax = τ₀, intervals)
taxpath = CommittedTaxPath(solution)

## Plot the tax
if isinteractive()
    import Plots

    normalisedtaxpath =  taxpath.taxpath ./ calibration.taxfactor

    fig = Plots.plot(taxpath.ts, normalisedtaxpath; xlabel = "Year", ylabel = "Policy", c = :darkred, xlims = extrema(taxpath.ts), yguidefontcolor = :darkred, label = false, linewidth = 2.5)

    afig = Plots.twinx(fig)
    Plots.plot!(afig, taxpath.ts, [firm.a₀, solution.a...]; c = :darkgreen, ylims = (0, 1.05 * firm.η), xlims = extrema(taxpath.ts), ylabel = "Abatemnet", yguidefontcolor = :darkgreen, label = false, linewidth = 2.5)

    Plots.hline!(afig, [firm.η]; c = :darkgreen, linestyle = :dash, label = "Zero emissions")

    fig
end

## Save
mkpath(dirname(filepath))
JLD2.jldopen(filepath, "w") do file
    file["model"] = model
    file["committedtaxpath"] = taxpath
    file["committed/solution"] = solution
    file["committed/welfarecost"] = solution.welfarecost
end
@info "Saved committed benchmark" filepath
