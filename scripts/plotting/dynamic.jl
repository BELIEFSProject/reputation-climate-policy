## Setup

using Revise

import DotEnv
import JLD2
import Printf
import Colors
import CairoMakie
import SciMLBase
import Random
import Statistics
import LogExpFunctions as LE
import StochasticDiffEq as SDE
import LinearAlgebra as LA
import SparseArrays as SP
import LaTeXStrings: @L_str

DotEnv.load!()

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/primitives/climate.jl")

includet("../../src/agents/households.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/dynamics/government.jl")
includet("../../src/solve/model.jl")
includet("../../src/solve/government/noncommittedmarkov.jl")
includet("../../src/solve/government/noncommitteddae.jl")
includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")
includet("../../src/dynamics/markovsimulation.jl")

includet("publication.jl")
includet("colours.jl")
includet("reputationutils.jl")

## Configuration

φ₀ = 0.5
npaths = 500
seed = 2026
Δt = 0.01
saveinterval = 0.25
interval = (0.1, 0.9)
samplepaths = 6
committed = false

background = Colors.colorant"#FAFAFA"
figuresize = (960, 540)
CairoMakie.set_theme!(publicationtheme)
CairoMakie.update_theme!(backgroundcolor = background, Axis = (; backgroundcolor = background), Legend = (; backgroundcolor = background))

rootdirectory = normpath(joinpath(@__DIR__, "..", ".."))
datapath = get(ENV, "DATAPATH", joinpath(rootdirectory, "data"))
plotpath = get(ENV, "PLOTPATH", joinpath(rootdirectory, "figures"))
solutionfile = get(ENV, "REPUTATION_SOLUTION", "")

## Load matching noise equilibria

household, firm, government, signal, climate = initmodels()
files = filter(path -> startswith(basename(path), "noncommitted-simplified-") && endswith(path, ".jld2"), readdir(joinpath(datapath, "solutions"); join = true))
sort!(files; by = mtime, rev = true)
equilibria = filter(!isnothing, loadreputationequilibrium.(files))

if isempty(solutionfile)
    label = solutionlabel(household, firm, signal, climate)
    candidates = filter(equilibrium -> solutionlabel(equilibrium.solution.parameters.model.household, equilibrium.solution.parameters.model.firm, equilibrium.solution.parameters.signal, equilibrium.solution.parameters.model.climate) == label, equilibria)
    if isempty(candidates)
        error("No saved equilibrium matches the calibration. Run scripts/noncommitted.jl or set REPUTATION_SOLUTION.")
    end
    baseline = first(candidates)
else
    baseline = loadreputationequilibrium(solutionfile)
    if isnothing(baseline)
        error("REPUTATION_SOLUTION must contain a current non-committed Markov equilibrium.")
    end
end

matches = filter(equilibrium -> comparablereputationequilibrium(equilibrium, baseline), equilibria)
pushfirst!(matches, baseline)
matches = unique(equilibrium -> equilibrium.solution.parameters.signal.σ, matches)
sort!(matches; by = equilibrium -> equilibrium.solution.parameters.signal.σ)

for equilibrium in matches
    p = equilibrium.solution.parameters
    if maximum(d -> max(d.valuegap, d.bellmangap), equilibrium.solution.diagnostics) > p.tolerance || maximum(d -> d.policygap, equilibrium.solution.diagnostics) > p.policytolerance
        error("Saved equilibrium residuals exceed tolerance in $(equilibrium.filename).")
    end
end

if length(matches) < 2
    @warn "Only one matching noise equilibrium is available; solve other noise calibrations separately for a comparison."
end

parameters = baseline.solution.parameters
household, firm = parameters.model.household, parameters.model.firm
horizon = last(baseline.solution.t)
σs = [equilibrium.solution.parameters.signal.σ for equilibrium in matches]
τᶜ = parameters.τᶜ
jumps = unique([τᶜ.ts[k] for k in 1:(length(τᶜ.ts) - 1) if τᶜ.ts[k] == τᶜ.ts[k + 1] && τᶜ.taxpath[k] != τᶜ.taxpath[k + 1] && 0 < τᶜ.ts[k] ≤ horizon])
times = sort!(unique!(vcat(collect(0:saveinterval:horizon), horizon, jumps)))
ts = [t for t in times for _ in 1:(t in jumps ? 2 : 1)]
priors = (fill(φ₀, npaths), clamp.(rand(Random.Xoshiro(seed), npaths), eps(), 1 - eps()))
cases = ("fixed", "uniform")

## Simulate the physical belief diffusion

# The cache identity includes every source equilibrium and simulation setting.
cachelabel = join(("physical-belief-sde-v1", join(noncommittedsolutionkey.(getproperty.(matches, :solution)), ","), string(φ₀), string(npaths), string(seed), string(Δt), string(saveinterval), string(committed)), "|")
key = bytes2hex(SHA.sha256(cachelabel))
tag = first(key, solutionhashlength)
cachepath = joinpath(datapath, "simulations", "dynamic-$tag.jld2")
figurepath = joinpath(plotpath, "dynamic-$tag")
mkpath(dirname(cachepath))

if isfile(cachepath)
    paths = JLD2.jldopen(cachepath, "r") do file
        if file["dynamics/key"] != key
            error("Simulation cache identity does not match $cachepath.")
        end
        [[file["dynamics/$case/$k/paths"] for k in eachindex(matches)] for case in cases]
    end
    @info "Loaded stochastic transitions" cache = cachepath
else
    paths = [[begin
        @info "Simulating stochastic transitions" case = cases[case] noise = σs[k] trajectories = npaths
        simulatedynamics(equilibrium, priors[case], ts; timestep = Δt, seed, committed)
    end for (k, equilibrium) in enumerate(matches)] for case in eachindex(cases)]

    JLD2.jldopen(cachepath, "w") do file
        file["dynamics/key"] = key
        file["dynamics/t"] = ts
        file["dynamics/noise"] = σs
        file["dynamics/committed"] = committed
        file["dynamics/seed"] = seed
        file["dynamics/timestep"] = Δt
        file["dynamics/sources"] = getproperty.(matches, :filename)
        for (case, name) in enumerate(cases)
            file["dynamics/$name/initialbeliefs"] = priors[case]
            for k in eachindex(matches)
                file["dynamics/$name/$k/paths"] = paths[case][k]
            end
        end
    end
end

## Diagnostics

mandates = [τᶜ(t; side = row < length(ts) && ts[row + 1] == t ? :left : :right) for (row, t) in enumerate(ts)]
for (case, name) in enumerate(cases), k in eachindex(matches)
    values = paths[case][k]
    σᵩ = (matches[k].solution.parameters.signal.ϵ / σs[k]) .* values[:, 1, :] .* (mandates .- values[:, 3, :])
    @info "Stochastic belief dispersion" case = name noise = σs[k] initialsd = Statistics.std(values[1, 1, :]) maximumsd = maximum(Statistics.std(values[:, 1, :]; dims = 2)) maximumvolatility = maximum(abs, σᵩ)
end

## Separate state and policy figures

gradient = CairoMakie.cgrad([defaultpalette[:committed], defaultpalette[:abatement], defaultpalette[:damages]])
colours = [gradient[(k - 1) / max(length(matches) - 1, 1)] for k in eachindex(matches)]
columns = (2, 3, 1, 4, 5, 6, 7, 8, 9, 10)
scales = (taxfactor, taxfactor, 1, firm.η, c(firm), firm.η, n(0, firm.a₀, household, firm), 1, 1, 1)
titles = ("Carbon policy", "Expected carbon policy", "Reputation", "Abatement", "Value of abatement investment", "Abatement investment", "Labour", "Output", "Emissions", "Cumulative emissions")
labels = ("USD/tCO₂e", "USD/tCO₂e", "Probability of commitment", "Share of abatement capacity", L"$q/c$", "Share of capacity installed per year", "Labour / zero-tax labour", "Trillion USD/year", "GtCO₂e/year", "GtCO₂e")
filenames = ("policy", "expected-policy", "belief", "abatement", "valuation", "investment", "labour", "output", "emissions", "cumulative-emissions")
mkpath(figurepath)

open(joinpath(figurepath, "summary.csv"), "w") do file
    println(file, "case,noise,time,variable,lower,median,upper")
    for (case, name) in enumerate(cases), panel in eachindex(columns)
        figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
        axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Years since announcement", ylabel = labels[panel], title = titles[panel], limits = ((0, horizon), (0, nothing)))
        if panel in (3, 4, 6, 7)
            axis.ytickformat = percenttickformat
        end
        if panel in (3, 4)
            CairoMakie.ylims!(axis, 0, 1.02)
        end

        for k in eachindex(matches)
            values = paths[case][k][:, columns[panel], :] ./ scales[panel]
            quantiles = reduce(hcat, [Statistics.quantile(view(values, row, :), [interval[1], 0.5, interval[2]]) for row in axes(values, 1)])
            lower, median, upper = eachrow(quantiles)
            CairoMakie.band!(axis, ts, lower, upper; color = (colours[k], publicationdefault(:intervalopacity)))
            for trajectory in 1:min(samplepaths, npaths)
                CairoMakie.lines!(axis, ts, values[:, trajectory]; color = (colours[k], publicationdefault(:samplepathopacity)), linewidth = publicationdefault(:samplepathlinewidth))
            end
            CairoMakie.lines!(axis, ts, median; color = colours[k], linewidth = publicationdefault(:medianlinewidth), label = Printf.@sprintf("σ = %.2f", σs[k]))
            for row in eachindex(ts)
                println(file, join((name, σs[k], ts[row], filenames[panel], lower[row], median[row], upper[row]), ","))
            end
        end

        priorlabel = case == 1 ? Printf.@sprintf("Initial belief %.0f%%", 100φ₀) : "Initial beliefs uniform on (0,1)"
        typelabel = committed ? "Committed government" : "Non-committed government"
        caption = Printf.@sprintf("%s; %s\nMedian and %.0f–%.0f%% bands; %d trajectories", typelabel, priorlabel, 100interval[1], 100interval[2], npaths)
        CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
        CairoMakie.Label(figure[3, 1], caption; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
        directory = joinpath(figurepath, name)
        mkpath(directory)
        savepublicationfigure(joinpath(directory, filenames[panel]), figure)
    end
end

@info "Saved stochastic dynamics figures" directory = figurepath cache = cachepath noise = σs
