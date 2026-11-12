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
φ₀ = 0.5
Δt = 0.25

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
    candidates = filter(equilibrium -> solutionlabel(equilibrium.solution.parameters.model.household, equilibrium.solution.parameters.model.firm, equilibrium.solution.parameters.signal, equilibrium.solution.parameters.model.climate) == label, equilibria)
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

basefigurepath = joinpath(plotpath, splitext(basename(baseline.filename))[1], "reputation")
@info "Plotting commitment comparisons" source = baseline.filename noise = parameters.signal.σ horizon = horizon

## Commitment and non-commitment transitions

@info "Comparing committed and non-committed transitions" initialbelief = φ₀
figurepath = joinpath(basefigurepath, "commitment")
mkpath(figurepath)
benchmark = loadreputationbenchmark(baseline, joinpath(datapath, "solutions"))
ts, paths, diagnostics = simulatecommitment(baseline; initialbelief = φ₀, timestep = Δt, diagnostics = true)
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
    caption = panel == 1 ? "Includes net-zero states, where the tax rate has no effect on emissions" : Printf.@sprintf("Common Markov grid; private-information prior %.0f%%; no additional news shock", 100φ₀)
    CairoMakie.Label(figure[3, 1], caption; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    savepublicationfigure(joinpath(figurepath, filenames[panel]), figure)
end

gapfig = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
gapaxis = CairoMakie.Axis(gapfig[1, 1]; xlabel = "Years since announcement", ylabel = "USD/tCO₂e", title = "Tax shortfall before full abatement", limits = ((0, horizon), (0, nothing)))
for case in 2:3
    CairoMakie.lines!(gapaxis, ts, diagnostics[:, 4, case] ./ taxfactor; color = colours[case + 1], linewidth = publicationdefault(:medianlinewidth), label = cases[case + 1])
end
CairoMakie.Legend(gapfig[2, 1], gapaxis; orientation = :horizontal, tellwidth = false)
CairoMakie.Label(gapfig[3, 1], L"$\mathbb{E}[\tau^{\mathrm{c}}-\tau\mid a<\eta]$; net-zero states excluded"; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
savepublicationfigure(joinpath(figurepath, "policy-gap"), gapfig)

open(joinpath(figurepath, "diagnostics.csv"), "w") do file
    println(file, "case,time,capacitymass,taxbelowcapacityUSD,expectedtaxbelowcapacityUSD,gapbelowcapacityUSD,activegapcontributionUSD")
    for case in 1:3, k in eachindex(ts)
        capacity = diagnostics[k, 1, case]
        values = diagnostics[k, 2:4, case] ./ taxfactor
        println(file, join((cases[case + 1], ts[k], capacity, values..., (1 - capacity) * values[3]), ","))
    end
end

checkfig = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
checkaxis = CairoMakie.Axis(checkfig[1, 1]; xlabel = "Years since announcement", ylabel = "Share of abatement capacity", title = "Known-commitment benchmark check", ytickformat = percenttickformat, limits = ((0, horizon), (0, 1.02)))

for case in 1:2
    CairoMakie.lines!(checkaxis, ts, series[4][:, case]; color = colours[case], linestyle = case == 1 ? :dash : :solid, linewidth = publicationdefault(:medianlinewidth), label = cases[case])
end

CairoMakie.Legend(checkfig[2, 1], checkaxis; orientation = :horizontal, tellwidth = false)
CairoMakie.Label(checkfig[3, 1], Printf.@sprintf("Same announcement and primitives; Markov grid h = %.3g", grid.h); fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
savepublicationfigure(joinpath(figurepath, "benchmark-abatement"), checkfig)

open(joinpath(figurepath, "paths.csv"), "w") do file
    println(file, "case,time,taxUSD,expectedtaxUSD,reputation,abatementshare,valuationratio,investmentshare,labourratio,output,emissions,cumulativeemissions")
    for case in 1:4, k in eachindex(ts)
        println(file, join((cases[case], ts[k], (values[k, case] for values in series)...), ","))
    end
end

@info "Commitment comparison" terminalabatement = series[4][end, :] terminalemissions = series[9][end, :] cumulativeemissions = series[10][end, :]
@info "Commitment comparison gaps" maximumabatementgap = maximum(abs, paths[:, 4, 2] .- paths[:, 4, 3]) / firm.η maximumemissionsgap = maximum(abs, paths[:, 9, 2] .- paths[:, 9, 3]) terminalcapacitymass = diagnostics[end, 1, :]

@info "Saved commitment comparison figures" directory = figurepath
