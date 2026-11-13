## Setup
using Revise

import FastClosures: @closure

import DotEnv
import JLD2
import Printf
import Colors
import CairoMakie
import SciMLBase
import UnPack: @unpack
import LaTeXStrings: @L_str
import LinearAlgebra as LA
import SparseArrays as SP
import OrdinaryDiffEq as ODE

DotEnv.load!()

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/primitives/climate.jl")

includet("../../src/agents/households.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/dynamics/government.jl")

includet("../../src/solve/model.jl")
includet("../../src/solve/government/committed.jl")
includet("../../src/solve/government/noncommittedmarkov.jl")
includet("../../src/solve/government/noncommitteddae.jl")
includet("../../src/dynamics/committed.jl")

includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")

includet("publication.jl")
includet("colours.jl")
includet("reputationutils.jl")

## Configuration
background = Colors.colorant"#FAFAFA"
CairoMakie.set_theme!(publicationtheme)
CairoMakie.update_theme!(backgroundcolor = background, Axis = (; backgroundcolor = background), Legend = (; backgroundcolor = background))

figuresize = (960, 540)
φ₀ = 0.9
Δt = 0.25
characteristicstep = 0.05
characteristictolerance = 1e-8

rootdirectory = normpath(joinpath(@__DIR__, "..", ".."))
datapath = get(ENV, "DATAPATH", joinpath(rootdirectory, "data"))
plotpath = get(ENV, "PLOTPATH", joinpath(rootdirectory, "figures"))
solutionfile = get(ENV, "REPUTATION_SOLUTION", "")

## Load the saved equilibrium
household, firm, government, signal, climate = initmodels()

if isempty(solutionfile)
    validfiles = filter(path -> startswith(basename(path), "noncommitted-simplified-") && endswith(path, ".jld2"), readdir(joinpath(datapath, "solutions"); join = true))
    sort!(validfiles; by = mtime, rev = true)
    equilibria = filter(!isnothing, loadreputationequilibrium.(validfiles))
    label = solutionlabel(household, firm, signal, climate)
    candidates = filter(equilibrium -> solutionlabel(equilibrium.solution.parameters.model.household, equilibrium.solution.parameters.model.firm, equilibrium.solution.parameters.signal, equilibrium.solution.parameters.model.climate) == label && equilibrium.solution.parameters.government == government, equilibria)
    if isempty(candidates)
        error("No saved non-committed equilibrium matches the calibration. Set REPUTATION_SOLUTION to select a saved file.")
    end

    baseline = first(candidates)
else
    baseline = loadreputationequilibrium(solutionfile)
    if isnothing(baseline)
        error("REPUTATION_SOLUTION must contain a current non-committed Markov equilibrium.")
    end
end

parameters = baseline.solution.parameters
model = parameters.model
firm, household = model.firm, model.household
grid = parameters.grid
horizon = last(baseline.solution.t)

if maximum(d -> max(d.valuegap, d.bellmangap), baseline.solution.diagnostics) > parameters.tolerance || maximum(d -> d.policygap, baseline.solution.diagnostics) > parameters.policytolerance
    error("Saved equilibrium residuals exceed tolerance in $(baseline.filename).")
end

runlabel = join(("characteristic-v1", ncsolutionkey(baseline.solution), string(φ₀), string(Δt), string(characteristicstep), string(characteristictolerance)), "|")
@info "Plotting commitment comparisons" source = baseline.filename noise = parameters.signal.σ horizon = horizon

## Commitment and non-commitment transitions

@info "Comparing committed and non-committed transitions" initialbelief = φ₀
figurepath = publicationfigurepath(plotpath, household, firm, model.climate, "commitment", runlabel)
mkpath(figurepath)
benchmark = loadreputationbenchmark(baseline, joinpath(datapath, "solutions"))
ts, paths, diagnostics = simulatecommitment(baseline; initialbelief = φ₀, timestep = Δt, diagnostics = true)
characteristic = simulatecharacteristic(baseline.solution, ts; timestep = characteristicstep, tolerance = characteristictolerance)
benchmarkvalues = zeros(eltype(ts), length(ts), 10)

for (k, t) in enumerate(ts)
    if t < benchmark.stoppingtime
        index = min(searchsortedlast(benchmark.t, t), length(benchmark.t) - 1)
        weight = (t - benchmark.t[index]) / (benchmark.t[index + 1] - benchmark.t[index])
        a = clamp((1 - weight) * benchmark.a[index] + weight * benchmark.a[index + 1], firm.a₀, firm.η)
        q = (1 - weight) * benchmark.q[index] + weight * benchmark.q[index + 1]
    else
        a, q = firm.η, c(firm)
    end

    side = k < length(ts) && ts[k + 1] == t ? :left : :right
    τ = parameters.τᶜ(t; side)
    nₜ = n(τ, a, household, firm)
    eₜ = e(nₜ, a, firm)
    cumulative = k == 1 ? 0 : benchmarkvalues[k - 1, 10] + (t - ts[k - 1]) * eₜ
    benchmarkvalues[k, :] .= (1, τ, τ, a, q, max(q - c(firm), 0) / firm.ξ, nₜ, y(nₜ, firm), eₜ, cumulative)
end

cases = ("Ramsey (continuous BVP)", "Known commitment (Markov)", "Committed (uncertain type)", "Non-committed (uncertain type)")
colours = (defaultpalette[:guide], defaultpalette[:committed], defaultpalette[:abatement], defaultpalette[:damages])
columns = (2, 3, 1, 4, 5, 6, 7, 8, 9, 10)
scales = (taxfactor, taxfactor, 1, firm.η, c(firm), firm.η, n(0, firm.a₀, household, firm), 1, 1, 1)
series = [hcat(benchmarkvalues[:, column], paths[:, column, :]) ./ scale for (column, scale) in zip(columns, scales)]
titles = ("Mean carbon tax rate", "Expected carbon policy", "Reputation", "Abatement", "Value of abatement investment", "Abatement investment", "Labour", "Output", "Emissions", "Cumulative emissions")
labels = ("USD/tCO₂e", "USD/tCO₂e", "Probability of commitment", "Share of abatement capacity", L"$q/c$", "Share of capacity installed per year", "Labour / zero-tax labour", "Trillion USD/year", "GtCO₂e/year", "GtCO₂e")
filenames = ("policy", "expected-policy", "belief", "abatement", "valuation", "investment", "labour", "output", "emissions", "cumulative-emissions")

for panel in eachindex(series)
    figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
    axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Years since announcement", ylabel = labels[panel], title = titles[panel], limits = ((0, horizon), (0, nothing)))
    if panel in (3, 4, 6, 7)
        axis.ytickformat = percenttickformat
    end

    if panel in (3, 4)
        CairoMakie.ylims!(axis, 0, 1.02)
    end

    for case in 2:4
        CairoMakie.lines!(axis, ts, series[panel][:, case]; color = colours[case], linestyle = case == 2 ? :dash : :solid, linewidth = publicationdefault(:medianlinewidth), label = cases[case])
    end

    CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
    savepublicationfigure(joinpath(figurepath, filenames[panel]), figure)
end

gapfig = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
gapaxis = CairoMakie.Axis(gapfig[1, 1]; xlabel = "Years since announcement", ylabel = "USD/tCO₂e", title = "Tax shortfall before full abatement", limits = ((0, horizon), (0, nothing)))
for case in 2:3
    CairoMakie.lines!(gapaxis, ts, diagnostics[:, 4, case] ./ taxfactor; color = colours[case + 1], linewidth = publicationdefault(:medianlinewidth), label = cases[case + 1])
end
CairoMakie.Legend(gapfig[2, 1], gapaxis; orientation = :horizontal, tellwidth = false)
savepublicationfigure(joinpath(figurepath, "policy-gap"), gapfig)

open(joinpath(figurepath, "diagnostics.csv"), "w") do file
    println(file, "case,time,capacitymass,taxbelowcapacityUSD,expectedtaxbelowcapacityUSD,gapbelowcapacityUSD,activegapcontributionUSD")
    for case in 1:3, k in eachindex(ts)
        capacity = diagnostics[k, 1, case]
        values = diagnostics[k, 2:4, case] ./ taxfactor
        println(file, join((cases[case + 1], ts[k], capacity, values..., (1 - capacity) * values[3]), ","))
    end
end

for (column, scale, title, ylabel, filename) in zip((4, 5), (firm.η, c(firm)), ("Known-commitment abatement check", "Known-commitment valuation check"), ("Share of abatement capacity", L"$q/c$"), ("benchmark-abatement", "benchmark-valuation"))
    checkfig = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
    checkaxis = CairoMakie.Axis(checkfig[1, 1]; xlabel = "Years since announcement", ylabel, title, limits = ((0, horizon), (0, nothing)))
    if column == 4
        checkaxis.ytickformat = percenttickformat
        CairoMakie.ylims!(checkaxis, 0, 1.02)
    end
    for (values, colour, style, caption) in zip((benchmarkvalues, characteristic, paths[:, :, 1]), (defaultpalette[:guide], defaultpalette[:abatement], defaultpalette[:committed]), (:dash, :dot, :solid), (cases[1], "Known commitment (characteristic)", cases[2]))
        CairoMakie.lines!(checkaxis, ts, values[:, column] ./ scale; color = colour, linestyle = style, linewidth = publicationdefault(:medianlinewidth), label = caption)
    end
    CairoMakie.Legend(checkfig[2, 1], checkaxis; orientation = :horizontal, tellwidth = false)
    savepublicationfigure(joinpath(figurepath, filename), checkfig)
end

open(joinpath(figurepath, "paths.csv"), "w") do file
    println(file, "case,time,taxUSD,expectedtaxUSD,reputation,abatementshare,valuationratio,investmentshare,labourratio,output,emissions,cumulativeemissions")
    for case in 1:4, k in eachindex(ts)
        println(file, join((cases[case], ts[k], (values[k, case] for values in series)...), ","))
    end
    for k in eachindex(ts)
        println(file, join(("Known commitment (characteristic)", ts[k], (characteristic[k, column] / scale for (column, scale) in zip(columns, scales))...), ","))
    end
end

open(joinpath(figurepath, "benchmark.csv"), "w") do file
    println(file, "time,h,timestep,ramseyabatement,characteristicabatement,markovabatement,ramseyvaluation,characteristicvaluation,markovvaluation")
    for k in eachindex(ts)
        println(file, join((ts[k], grid.h, parameters.timestep, benchmarkvalues[k, 4] / firm.η, characteristic[k, 4] / firm.η, paths[k, 4, 1] / firm.η, benchmarkvalues[k, 5] / c(firm), characteristic[k, 5] / c(firm), paths[k, 5, 1] / c(firm)), ","))
    end
end

@info "Commitment comparison" terminalabatement = series[4][end, :] terminalemissions = series[9][end, :] cumulativeemissions = series[10][end, :]
@info "Commitment comparison gaps" maximumabatementgap = maximum(abs, paths[:, 4, 2] .- paths[:, 4, 3]) / firm.η maximumemissionsgap = maximum(abs, paths[:, 9, 2] .- paths[:, 9, 3]) terminalcapacitymass = diagnostics[end, 1, :]
@info "Known-commitment benchmark gaps" gridstep = grid.h timestep = parameters.timestep characteristicabatementgap = maximum(abs, characteristic[:, 4] .- benchmarkvalues[:, 4]) / firm.η chainabatementgap = maximum(abs, paths[:, 4, 1] .- characteristic[:, 4]) / firm.η characteristicvaluationgap = maximum(abs, characteristic[:, 5] .- benchmarkvalues[:, 5]) / c(firm)

@info "Saved commitment comparison figures" directory = figurepath
