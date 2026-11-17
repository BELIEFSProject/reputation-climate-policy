## Setup
using Revise

import Roots
import ForwardDiff as FD
import ADTypes
import ArgParse
import SHA
import StaticArrays as SA

import DotEnv
import JLD2
import Printf
import UnPack: @unpack
import FastClosures: @closure
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

includet("../../src/primitives/calibration.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/primitives/climate.jl")
includet("../../src/agents/households.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")
includet("../../src/solve/model.jl")
includet("../../src/dynamics/government.jl")

includet("../../src/solve/government/committed.jl")
includet("../../src/dynamics/ramsey.jl")
includet("../../src/solve/government/nc/grid.jl")
includet("../../src/solve/government/nc/parameters.jl")
includet("../../src/solve/government/nc/policy.jl")
includet("../../src/solve/government/nc/operator.jl")
includet("../../src/solve/government/nc/solve.jl")
includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")
includet("../../src/dynamics/markov/types.jl")
includet("../../src/dynamics/markov/observables.jl")
includet("../../src/dynamics/markov/ensemble.jl")

includet("publication.jl")
includet("colours.jl")
includet("common/results.jl")
includet("common/loading.jl")
includet("common/projection.jl")

calibration = Calibration()
taxfactor = calibration.taxfactor

## Configuration
φ₀ = 0.9
npaths = 10_000
seed = 2026
Δt = 0.01
saveinterval = 0.25
interval = (0.1, 0.9)
samplepaths = 20
figureσs = [0.19, 0.38, 0.76, 1.52]

background = Colors.colorant"#FAFAFA"
figuresize = (960, 540)
CairoMakie.set_theme!(publicationtheme)
CairoMakie.update_theme!(backgroundcolor = background, Axis = (; backgroundcolor = background), Legend = (; backgroundcolor = background))

rootdirectory = pwd()
datapath = get(ENV, "DATAPATH", joinpath(rootdirectory, "data"))
plotpath = get(ENV, "PLOTPATH", joinpath(rootdirectory, "figures"))
solutionfile = get(ENV, "REPUTATION_SOLUTION", "")

## Load matching noise equilibria
household, firm, government, signal, climate = initmodels(; calibration)
baseline, matches = loadequilibria(joinpath(datapath, "solutions"), Model(household, firm, climate), government, signal; filename = solutionfile)
filter!(eq -> eq.solution.parameters.signal.σ ∈ figureσs, matches)

parameters = baseline.solution.parameters
household, firm = parameters.model.household, parameters.model.firm
horizon = last(baseline.solution.t)
σs = [equilibrium.solution.parameters.signal.σ for equilibrium in matches]
τᶜ = parameters.τᶜ
jumps = unique([τᶜ.ts[k] for k in 1:(length(τᶜ.ts) - 1) if τᶜ.ts[k] == τᶜ.ts[k + 1] && τᶜ.taxpath[k] != τᶜ.taxpath[k + 1] && 0 < τᶜ.ts[k] ≤ horizon])
times = sort!(unique!(vcat(collect(0:saveinterval:horizon), horizon, jumps)))
ts = [t for t in times for _ in 1:(t in jumps ? 2 : 1)]
baselineyear = 2020.
yeartime = ts .+ baselineyear
priors = (fill(φ₀, npaths), clamp.(rand(Random.Xoshiro(seed), npaths), eps(), 1 - eps()))
cases = ("fixed", "uniform")

## Ramsey benchmark

benchmark = loadreputationbenchmark(baseline, joinpath(datapath, "solutions"))
benchmarkvalues = ramsey(benchmark, τᶜ, ts)

## Simulate the physical belief diffusion
runlabel = join(("ensemble-v1", join(ncsolutionkey.(getproperty.(matches, :solution)), ","), string(φ₀), string(npaths), string(seed), string(Δt), string(saveinterval)), "|")
figurepaths = [publicationfigurepath(plotpath, household, firm, parameters.model.climate, "dynamic-$name", runlabel) for name in cases]

paths = [[simulatedynamics(eq, φ, ts; timestep = Δt, seed) for eq in matches] for φ in priors]

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
titles = (L"Policy $\tau$", L"Expected policy $\mathbb{E}_{t, \phi}[\tau]$", L"Reputation $\phi$", L"Installed batement $a_t$", L"Value of abatement investment $q_t$", L"Abatement investment $\dot{a}_t$", L"Labour $n_t$", L"Output $y_t$", L"Emissions $e_t$", L"Cumulative emissions $\int^t_{0} e_t \text{d}t$")
labels = ("USD/tCO₂e", "USD/tCO₂e", "Probability of commitment", "Share of abatement capacity", L"$q/c$", "Share of capacity installed per year", "Labour / zero-tax labour", "Trillion USD/year", "GtCO₂e/year", "GtCO₂e")
filenames = ("policy", "expected-policy", "belief", "abatement", "valuation", "investment", "labour", "output", "emissions", "cumulative-emissions")

allfigures = CairoMakie.Figure[]
for (case, name) in enumerate(cases)
    directory = figurepaths[case]
    mkpath(directory)
    open(joinpath(directory, "summary.csv"), "w") do file
        println(file, "case,government,noise,time,variable,lower,median,upper")
        for panel in eachindex(columns)
            figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
            axis = CairoMakie.Axis(figure[1, 1]; xlabel = L"Year $t$", ylabel = labels[panel], title = titles[panel], limits = ((0, horizon), (0, nothing)), xtickformat = ts -> map(t -> Printf.@sprintf("%d", t + baselineyear), ts))
            CairoMakie.xlims!(axis, 0., floor(horizon))

            if panel in (3, 4, 6, 7)
                axis.ytickformat = percenttickformat
            end
            if panel in (3, 4)
                CairoMakie.ylims!(axis, 0, 1.02)
            end

            committedvalues = benchmarkvalues[:, columns[panel]] ./ scales[panel]
            CairoMakie.lines!(axis, ts, committedvalues; color = defaultpalette[:guide], linestyle = :dash, linewidth = publicationdefault(:medianlinewidth), label = "Committed")
            for row in eachindex(ts)
                println(file, join((name, "Committed", NaN, ts[row], filenames[panel], committedvalues[row], committedvalues[row], committedvalues[row]), ","))
            end

            for k in eachindex(matches)
                values = paths[case][k][:, columns[panel], :] ./ scales[panel]
                quantiles = reduce(hcat, [Statistics.quantile(view(values, row, :), [interval[1], 0.5, interval[2]]) for row in axes(values, 1)])
                lower, median, upper = eachrow(quantiles)
                CairoMakie.band!(axis, ts, lower, upper; color = (colours[k], publicationdefault(:intervalopacity)))
                for trajectory in 1:min(samplepaths, npaths)
                    CairoMakie.lines!(axis, ts, values[:, trajectory]; color = (colours[k], publicationdefault(:samplepathopacity)), linewidth = publicationdefault(:samplepathlinewidth), linestyle = :solid)
                end
                CairoMakie.lines!(axis, ts, median; color = colours[k], linewidth = publicationdefault(:medianlinewidth), linestyle = :solid, label = Printf.@sprintf("nc, σ = %.2f", σs[k]))
                for row in eachindex(ts)
                    println(file, join((name, "nc", σs[k], ts[row], filenames[panel], lower[row], median[row], upper[row]), ","))
                end
            end

            CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, nbanks = 2, tellwidth = false)
            savepublicationfigure(joinpath(directory, filenames[panel]), figure)

            push!(allfigures, figure)
        end
    end
end

@info "Saved stochastic dynamics figures" directories = figurepaths noise = σs

## Simulate dynamics with different ϕ₀
φ₀s = [0.1, 0.2, 0.5, 0.85, 0.9, 0.95]
eqbaseline = baseline
initcolours =  reverse([gradient[(k - 1) / max(length(φ₀s) - 1, 1)] for k in eachindex(φ₀s)])
dynamics = simulateensembledynamics(eqbaseline, φ₀s, ts; trajectories = 1_000)
montecarlodir = publicationfigurepath(plotpath, household, firm, parameters.model.climate, "dynamic-monte-carlo", runlabel)
if !isdir(montecarlodir)
    mkdir(montecarlodir)
end

begin
    dynamicfigures = CairoMakie.Figure[]
    for (panel, column) in enumerate(columns)
        figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
        axis = CairoMakie.Axis(figure[1, 1]; xlabel = L"Year $t$", ylabel = labels[panel], title = titles[panel], limits = ((0, horizon), (0, nothing)), xtickformat = ts -> map(t -> Printf.@sprintf("%d", t + baselineyear), ts))

        if panel in (3, 4, 6, 7)
            axis.ytickformat = percenttickformat
        end

        if panel in (3, 4)
            CairoMakie.ylims!(axis, 0, 1.02)
        elseif panel == 6
            CairoMakie.ylims!(axis, 0., 0.08)
        else
            ymax = max(
                maximum(abs, dynamics[:, column, :, :]),
                maximum(abs, benchmarkvalues[:, column])
            ) ./ scales[panel]
            CairoMakie.ylims!(axis, 0., ceil(ymax))
        end

        CairoMakie.xlims!(axis, 0., floor(horizon))

        committedvalues = benchmarkvalues[:, column] ./ scales[panel]
        CairoMakie.lines!(axis, ts, committedvalues; color = defaultpalette[:guide], linestyle = :dash, linewidth = publicationdefault(:medianlinewidth), label = "Committed")

        for (j, φ₀) in enumerate(φ₀s)
            values = dynamics[:, column, j, :] ./ scales[panel]
            quantiles = hcat(map(row -> Statistics.quantile(row, [interval[1], 0.5, interval[2]]), eachrow(values))...)

            lower, median, upper = eachrow(quantiles)
            CairoMakie.band!(axis, ts, lower, upper; color = (initcolours[j], publicationdefault(:intervalopacity)))
            for trajectory in 1:min(samplepaths, npaths)
                CairoMakie.lines!(axis, ts, values[:, trajectory]; color = (initcolours[j], publicationdefault(:samplepathopacity)), linewidth = publicationdefault(:samplepathlinewidth), linestyle = :solid)
            end

            CairoMakie.lines!(axis, ts, median; color = initcolours[j], linewidth = publicationdefault(:medianlinewidth), linestyle = :solid, label = Printf.@sprintf("ϕ₀ = %.2f", φ₀))
        end

        CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, nbanks = 2, tellwidth = false)
        savepublicationfigure(joinpath(montecarlodir, "$(filenames[panel])"), figure)

        push!(dynamicfigures, figure)
    end
end

@info "Saved Monte Carlo dynamics figures" directory = montecarlodir
