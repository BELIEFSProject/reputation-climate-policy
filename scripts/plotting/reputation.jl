## Setup

using Revise

import DotEnv
import JLD2
import Printf
import Colors
import CairoMakie
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

includet("publication.jl")
includet("colours.jl")
includet("reputationutils.jl")

## Configuration

background = Colors.colorant"#FAFAFA"
CairoMakie.set_theme!(publicationtheme)
CairoMakie.update_theme!(backgroundcolor = background, Axis = (; backgroundcolor = background), Legend = (; backgroundcolor = background))

figuresize = (960, 540)
φ₀ = 0.5
oddsratio = 0.1
Δt = 0.25
nearzeroshare = 0.95
nshocks = 21

cases = ("Baseline learning", "Additional adverse news")
colours = (defaultpalette[:committed], defaultpalette[:damages])

rootdirectory = normpath(joinpath(@__DIR__, "..", ".."))
datapath = get(ENV, "DATAPATH", joinpath(rootdirectory, "data"))
plotpath = get(ENV, "PLOTPATH", joinpath(rootdirectory, "figures"))
solutionfile = get(ENV, "REPUTATION_SOLUTION", "")

## Load matching equilibria

household, firm, government, signal, climate = initmodels()

validfiles = filter(path -> startswith(basename(path), "noncommitted-simplified-") && endswith(path, ".jld2"), readdir(joinpath(datapath, "solutions"); join = true))
sort!(validfiles; by = mtime, rev = true)
equilibria = filter(!isnothing, loadreputationequilibrium.(validfiles))

if isempty(solutionfile)
    label = solutionlabel(household, firm, signal, climate)
    candidates = filter(equilibrium -> solutionlabel(equilibrium.solution.parameters.model.household, equilibrium.solution.parameters.model.firm, equilibrium.solution.parameters.signal, equilibrium.solution.parameters.model.climate) == label, equilibria)

    if isempty(candidates)
        error("No current non-committed equilibrium matches the calibration. Run scripts/noncommitted.jl, or set REPUTATION_SOLUTION to a saved solution file.")
    end

    baseline = first(candidates)
else
    baseline = loadreputationequilibrium(solutionfile)
    if isnothing(baseline)
        error("REPUTATION_SOLUTION must contain a current non-committed Markov equilibrium.")
    end

    if all(equilibrium -> equilibrium.filename != baseline.filename, equilibria)
        pushfirst!(equilibria, baseline)
    end
end

parameters = baseline.solution.parameters
model = parameters.model
firm, household = model.firm, model.household
signal = parameters.signal
grid = parameters.grid
horizon = last(baseline.solution.t)

tₛ = baseline.solution.t[argmin(abs.(baseline.solution.t .- min(10, horizon / 4)))]

matches = filter(equilibrium -> equilibrium.filename != baseline.filename && comparablereputationequilibrium(equilibrium, baseline), equilibria)
pushfirst!(matches, baseline)
matches = unique(equilibrium -> equilibrium.solution.parameters.signal.σ, matches)
sort!(matches; by = equilibrium -> equilibrium.solution.parameters.signal.σ)
baseindex = findfirst(equilibrium -> equilibrium.solution.parameters.signal.σ == signal.σ, matches)

for equilibrium in matches
    p = equilibrium.solution.parameters
    if maximum(d -> max(d.valuegap, d.bellmangap), equilibrium.solution.diagnostics) > p.tolerance || maximum(d -> d.policygap, equilibrium.solution.diagnostics) > p.policytolerance
        error("Saved equilibrium residuals exceed tolerance in $(equilibrium.filename). Recompute this equilibrium before plotting.")
    end
end

figurepath = joinpath(plotpath, splitext(basename(baseline.filename))[1], "reputation")
mkpath(figurepath)

@info "Plotting reputation and the transition" source = baseline.filename noise = signal.σ horizon = horizon newsdate = tₛ

## Forward distributions under the non-committed type

projections = map(matches) do equilibrium
    @info "Projecting the solved Markov chain" noise = equilibrium.solution.parameters.signal.σ
    projectreputation(equilibrium; initialbelief = φ₀, losstime = tₛ, oddsratio = oddsratio, timestep = Δt, nearzeroshare = nearzeroshare)
end

projection = projections[baseindex]
eventindex = findfirst(==(tₛ), projection.t)
ā = projection.abatement[eventindex, 1]
φ̄ = projection.belief[eventindex, 1]
j = argmin([abs(physicalstate(grid, 1, j)[2] - ā) for j in 1:size(grid, 2)])
i = argmin([abs(physicalstate(grid, i, 1)[1] - φ̄) for i in 1:size(grid, 1)])
a = physicalstate(grid, 1, j)[2]
φ = physicalstate(grid, i, 1)[1]
_, objects, _ = reputationoperator(baseline, tₛ)
slices = (1:size(grid, 1)) .+ (j - 1) * size(grid, 1)
φvalues = objects[slices, 1]
τᶜ = parameters.τᶜ(tₛ)
value = baseline.solution(tₛ)
q = value.q[:, j] ./ c(firm)
slicetitle = Printf.@sprintf("At year %.1f and %.0f%% abatement", tₛ, 100a / firm.η)

## Tax flatness: intrinsic preferences, observed pooling and reputation

preferred, policygap, preferencegap, taxwedge, spatialregimes = reputationtaxdiagnostics(baseline, value)
abatementshare = [physicalstate(grid, 1, j)[2] / firm.η for j in 1:(size(grid, 2) - 1)]
regimesmap = (preferencegap[:, 1:end-1], policygap[:, 1:end-1], taxwedge[:, 1:end-1])

const regimetitles = ("Downward-deviation incentive", "Distance from announcement", "Reputational tax wedge")
const regimenames = ("incentive", "announcement-gap", "reputation-wedge")
const regimecaptions = (L"$\tau^{\mathrm{c}}-\tau_P(a)$; capacity boundary excluded", L"$\tau^{\mathrm{c}}-\tau$; capacity boundary excluded", L"$\tau-\min\{\tau_P(a),\tau^{\mathrm{c}}\}$; capacity boundary excluded")

regimecolours = (CairoMakie.cgrad([defaultpalette[:abatement], background, defaultpalette[:damages]]), CairoMakie.cgrad([background, defaultpalette[:damages]]), CairoMakie.cgrad([defaultpalette[:damages], background, defaultpalette[:abatement]]))

for panel in 1:3
    figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
    axis = CairoMakie.Axis(figure[1, 1]; xlabel = L"Reputation $\phi$", ylabel = "Abatement / capacity", title = Printf.@sprintf("%s at year %.1f", regimetitles[panel], tₛ), xticks = 0:0.25:1, xtickformat = percenttickformat, ytickformat = percenttickformat)
    values = regimesmap[panel] ./ taxfactor
    bound = max(1, maximum(abs, values))
    limits = panel == 2 ? (0, bound) : (-bound, bound)
    image = CairoMakie.heatmap!(axis, φvalues, abatementshare, values; colormap = regimecolours[panel], colorrange = limits)
    for (case, index) in ((1, eventindex), (2, eventindex + 1))
        CairoMakie.scatter!(axis, [projection.belief[index, case]], [projection.abatement[index, case] / firm.η]; marker = case == 1 ? :circle : :xcross, markersize = 10, color = colours[case], label = cases[case])
    end

    CairoMakie.Colorbar(figure[1, 2], image; label = "USD/tCO₂e", width = 12)
    CairoMakie.Legend(figure[2, 1:2], axis; orientation = :horizontal, tellwidth = false)
    CairoMakie.Label(figure[3, 1:2], regimecaptions[panel]; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)

    panelpath = joinpath(figurepath, "tax-regime-map-$(regimenames[panel])")
    savepublicationfigure(panelpath, figure)
end

regimefigures = [CairoMakie.Figure(size = figuresize, backgroundcolor = background) for _ in 1:3]
incentiveaxis = CairoMakie.Axis(regimefigures[1][1, 1]; xlabel = "Years since announcement", ylabel = "Abatement / capacity", title = "Where is deviation attractive?", ytickformat = percenttickformat, limits = ((0, horizon), (first(abatementshare), 1)))

announcementtimes = unique(parameters.τᶜ.ts)
incentivevalues = [(parameters.τᶜ(t; side = :left) - preferred[j]) / taxfactor for t in announcementtimes, j in 1:(length(preferred) - 1)]
incentivebounds = max(1, maximum(abs, incentivevalues))
incentivemap = CairoMakie.heatmap!(incentiveaxis, announcementtimes, abatementshare, incentivevalues; colormap = regimecolours[1], colorrange = (-incentivebounds, incentivebounds))
CairoMakie.Colorbar(regimefigures[1][1, 2], incentivemap; label = L"$\tau^{\mathrm{c}}-\tau_P(a)$ [USD/tCO₂e]", width = 12)

if minimum(incentivevalues) < 0 < maximum(incentivevalues)
    CairoMakie.contour!(incentiveaxis, announcementtimes, abatementshare, incentivevalues; levels = [0], color = defaultpalette[:guide], linewidth = publicationdefault(:guidelinewidth))
end

poolaxis = CairoMakie.Axis(regimefigures[2][1, 1]; xlabel = "Years since announcement", ylabel = "Share below capacity", title = "Policy at the announcement", ytickformat = percenttickformat, limits = ((0, horizon), (0, 1)))
wedgeaxis = CairoMakie.Axis(regimefigures[3][1, 1]; xlabel = "Years since announcement", ylabel = "Mean absolute wedge [USD/tCO₂e]", title = "Size of the reputational wedge", limits = ((0, horizon), (0, nothing)))

for case in 1:2
    CairoMakie.lines!(incentiveaxis, projection.t, projection.abatement[:, case] ./ firm.η; color = colours[case], linewidth = publicationdefault(:medianlinewidth), label = cases[case])
    for (column, linestyle) in ((2, :solid), (1, :dash))
        shares = [1 - projection.regimes[k, 5, case] > 1e-10 ? projection.regimes[k, column, case] / (1 - projection.regimes[k, 5, case]) : NaN for k in eachindex(projection.t)]
        CairoMakie.lines!(poolaxis, projection.t, shares; color = colours[case], linestyle, linewidth = publicationdefault(:medianlinewidth), label = column == 2 ? cases[case] : nothing)
    end

    wedges = [1 - projection.regimes[k, 5, case] > 1e-10 ? projection.taxwedge[k, 2, case] / ((1 - projection.regimes[k, 5, case]) * taxfactor) : NaN for k in eachindex(projection.t)]
    CairoMakie.lines!(wedgeaxis, projection.t, wedges; color = colours[case], linewidth = publicationdefault(:medianlinewidth), label = cases[case])
end

const exposurenames = ("incentive", "pooling", "wedge")
const exposurecaptions = ("Green: no intrinsic incentive to cut tax; curves show mean abatement", "Solid: observed pooling; dashed: intrinsic preference; shares reflect discrete grid crossings", "Mean absolute tax wedge conditional on being below full abatement")

for (panel, axis) in enumerate((incentiveaxis, poolaxis, wedgeaxis))
    figure = regimefigures[panel]
    CairoMakie.vlines!(axis, [tₛ]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
    CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
    CairoMakie.Label(figure[3, 1], exposurecaptions[panel]; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    savepublicationfigure(joinpath(figurepath, "tax-regime-exposure-$(exposurenames[panel])"), figure)
end

## Reputation and current decisions

policyfigures = [CairoMakie.Figure(size = figuresize, backgroundcolor = background) for _ in 1:3]
policyaxes = [CairoMakie.Axis(figure[1, 1]; xlabel = L"Reputation $\phi$", title = title, ylabel = ylabel, xticks = 0:0.25:1, xtickformat = percenttickformat, limits = ((0, 1), nothing)) for (figure, title, ylabel) in zip(policyfigures, ("Actual and expected policy", "Value of abatement investment", "Investment"), ("USD/tCO₂e", L"$q/c$", "Capacity installed per year"))]

CairoMakie.lines!(policyaxes[1], φvalues, objects[slices, 2] ./ taxfactor; color = defaultpalette[:committed], linewidth = publicationdefault(:medianlinewidth), label = L"Actual $\tau$")
CairoMakie.lines!(policyaxes[1], φvalues, objects[slices, 3] ./ taxfactor; color = defaultpalette[:abatement], linestyle = :dash, linewidth = publicationdefault(:medianlinewidth), label = L"Expected $\tau^e$")
CairoMakie.hlines!(policyaxes[1], [τᶜ / taxfactor]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth), label = L"Announcement $\tau^{\mathrm{c}}$")
CairoMakie.lines!(policyaxes[2], φvalues, q; color = defaultpalette[:abatement], linewidth = publicationdefault(:medianlinewidth))
CairoMakie.hlines!(policyaxes[2], [1]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
CairoMakie.lines!(policyaxes[3], φvalues, objects[slices, 4] ./ firm.η; color = defaultpalette[:abatement], linewidth = publicationdefault(:medianlinewidth))
policyaxes[3].ytickformat = percenttickformat
CairoMakie.Legend(policyfigures[1][2, 1], policyaxes[1]; orientation = :horizontal, tellwidth = false)

const policynames = ("policy", "investment-value", "investment")

for (panel, figure) in enumerate(policyfigures)
    CairoMakie.Label(figure[3, 1], slicetitle; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)

    paneldir = joinpath(figurepath, "reputation-mechanism-$(policynames[panel])")
    savepublicationfigure(paneldir, figure)
end

## An unexpected reputation loss and the transition

transitionseries = (projection.belief, projection.tax ./ taxfactor, projection.abatement ./ firm.η, projection.cumulativeemissions)

const transitiontitles = ("Reputation", "Climate policy", "Abatement progress", "Cumulative emissions")
const transitionlabels = ("Probability of commitment", "USD/tCO₂e", "Share of abatement capacity", "GtCO₂e")
const transitionnames = ("belief", "policy", "abatement", "emissions")

for panel in 1:4
    figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
    axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Years since announcement", ylabel = transitionlabels[panel], title = transitiontitles[panel], limits = ((0, horizon), nothing))
    if panel == 1 || panel == 3
        axis.ytickformat = percenttickformat
        CairoMakie.ylims!(axis, 0, 1)
    end

    CairoMakie.vlines!(axis, [tₛ]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
    for case in 1:2
        CairoMakie.lines!(axis, projection.t, transitionseries[panel][:, case]; color = colours[case], linewidth = publicationdefault(:medianlinewidth), label = cases[case])
    end

    CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
    CairoMakie.Label(figure[0, 1], Printf.@sprintf("Unexpected news at year %.1f: commitment odds fall by %.0f%%", tₛ, 100(1 - oddsratio)); fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)
    CairoMakie.Label(figure[3, 1], "Expected paths conditional on a non-committed government"; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    savepublicationfigure(joinpath(figurepath, "reputation-transition-$(transitionnames[panel])"), figure)
end

## News timing along the baseline abatement transition

@info "Comparing reputation shocks at different abatement stages" shocks = nshocks
timingresult = reputationtiming(baseline, projection; shocks = nshocks, initialbelief = φ₀, oddsratio = oddsratio, timestep = Δt, nearzeroshare = nearzeroshare)

timingendpoint = timingresult.stationary ? "Steady-state" : "Terminal"
timingprogresslabel = timingresult.stationary ? "Baseline long-run abatement progress at the shock" : "Baseline terminal abatement progress at the shock"
timingemissions = 100 .* hcat(timingresult.terminalemissions ./ timingresult.baselineemissions .- 1, timingresult.cumulativeemissions ./ timingresult.baselinecumulativeemissions .- 1)
timingabatement = 100 .* (timingresult.terminalabatement ./ timingresult.baselineabatement .- 1)

timingseries = (reshape(timingemissions[:, 2], :, 1), reshape(timingabatement, :, 1))
timinglabels = (("Cumulative emissions to the common horizon",), ("$timingendpoint abatement",))
timingtitles = ("Effect of reputation-shock timing on cumulative emissions", "Effect of reputation-shock timing on abatement")
timingnames = ("emissions", "abatement")
timingticks = unique(round.(Int, range(1, length(timingresult.t); length = min(6, length(timingresult.t)))))

for panel in 1:2
    figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
    axis = CairoMakie.Axis(figure[1, 1]; xlabel = timingprogresslabel, ylabel = "Change relative to baseline [%]", xticks = 0:0.2:1, xtickformat = percenttickformat, yautolimitmargin = (0.05, 0.15), limits = ((0, 1), nothing))
    CairoMakie.hlines!(axis, [0]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
    for column in axes(timingseries[panel], 2)
        CairoMakie.scatterlines!(axis, timingresult.progress, timingseries[panel][:, column]; color = colours[3 - column], linestyle = column == 1 ? :solid : :dash, linewidth = publicationdefault(:medianlinewidth), markersize = 7, label = timinglabels[panel][column])
    end

    for k in timingticks
        alignment = k == first(timingticks) ? :left : k == last(timingticks) ? :right : :center
        CairoMakie.text!(axis, timingresult.progress[k], timingseries[panel][k, 1]; text = Printf.@sprintf("%.1f", timingresult.emissions[k]), align = (alignment, :bottom), offset = (0, 10), fontsize = 12)
    end

    CairoMakie.Label(figure[-1, 1], timingtitles[panel]; fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)
    CairoMakie.Label(figure[0, 1], "Point labels: pre-shock baseline emissions [GtCO₂e/year]"; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
    CairoMakie.Label(figure[3, 1], Printf.@sprintf("Commitment odds fall by %.0f%%; common horizon %.1f years; true government non-committed", 100(1 - oddsratio), horizon); fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    savepublicationfigure(joinpath(figurepath, "reputation-shock-timing-$(timingnames[panel])"), figure)
end

open(joinpath(figurepath, "reputation-shock-timing.csv"), "w") do file
    println(file, "shocktime,baselineprogress,baselineshockemissions,terminalabatement,terminalemissions,cumulativeemissions,relativeabatementpercent,relativeemissionspercent,relativecumulativeemissionspercent,baselineabatement,baselineemissions,baselinecumulativeemissions,stationarytail")
    for k in eachindex(timingresult.t)
        Printf.@printf(file, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n", timingresult.t[k], timingresult.progress[k], timingresult.emissions[k], timingresult.terminalabatement[k], timingresult.terminalemissions[k], timingresult.cumulativeemissions[k], timingabatement[k], timingemissions[k, 1], timingemissions[k, 2], timingresult.baselineabatement, timingresult.baselineemissions, timingresult.baselinecumulativeemissions, string(timingresult.stationary))
    end
end

@info "Shock-timing outcomes" stationarytail = timingresult.stationary emissionschange = extrema(timingemissions[:, 1]) cumulativechange = extrema(timingemissions[:, 2])

## The information channel of volatility curbing: a fixed-forecast diagnostic

σratios = collect(range(0.5, 2.0; length = 121))
σvalues = signal.σ .* σratios
node = i + (j - 1) * size(grid, 1)
τ = objects[node, 2]
Δτ = τᶜ - τ
∂ᵩuᵍ = 0.0

if 1 < i < size(grid, 1)
    ∂ᵩuᵍ = (value.uᵍ[i + 1, j] - value.uᵍ[i - 1, j]) / (2grid.h * span(grid.φ))
end

χvalues = signal.ϵ * Δτ ./ σvalues
σᵩ = φ * (1 - φ) .* χvalues
τ̂ = map(σvalues) do σ
    if a ≥ firm.η || iszero(τᶜ)
        return zero(τᶜ)
    end
    reputationterm = φ * (1 - φ) * (signal.ϵ / σ)^2 * Δτ * ∂ᵩuᵍ
    if household isa LinearHousehold
        return clamp(-(κ₁(a, household, firm, model.climate) + reputationterm) / κ₂(a, household, firm, model.climate), 0, τᶜ)
    end
    output = y(n(0, a, household, firm), firm)
    gap = firm.η - a
    target = 2model.climate.γ * output - 2reputationterm / (gap^2 * output)
    return target ≤ 0 ? zero(target) : clamp(2target / (sqrt((gap * target)^2 + 4) + gap * target), 0, τᶜ)
end
informationseries = (χvalues.^2 ./ 2, σᵩ, τ̂ ./ taxfactor)
informationtitles = ("Statistical separation", "Belief volatility", "Hidden best response")
informationlabels = (L"$\chi^2/2$ [per year]", L"$\sigma_\phi$ [per $\sqrt{\mathrm{year}}$]", "USD/tCO₂e")
informationnames = ("separation", "belief-volatility", "best-response")
informationcaption = iszero(Δτ) ? "Policy mimics at this state: the local signal carries no information about type" : "Policy forecasts and continuation values held fixed"

for panel in 1:3
    figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
    axis = CairoMakie.Axis(figure[1, 1]; xlabel = L"Signal noise $\sigma/\sigma_0$", ylabel = informationlabels[panel], title = informationtitles[panel], limits = (extrema(σratios), (0, nothing)))
    CairoMakie.vlines!(axis, [1]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
    CairoMakie.lines!(axis, σratios, informationseries[panel]; color = defaultpalette[:abatement], linewidth = publicationdefault(:medianlinewidth))
    CairoMakie.Label(figure[0, 1], Printf.@sprintf("The information channel at year %.1f, φ = %.0f%%", tₛ, 100φ); fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)
    CairoMakie.Label(figure[2, 1], informationcaption; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    savepublicationfigure(joinpath(figurepath, "ets-information-channel-$(informationnames[panel])"), figure)
end

## Equilibrium signal-noise comparisons, when matching solutions are available

ȳ = y(n(0, firm.η, household, firm), firm)
initialmass = reputationinitialmass(grid, φ₀, firm.a₀)
u₀ = LA.dot(initialmass, vec(first(baseline.solution.u).u))
uvalues = [LA.dot(initialmass, vec(first(equilibrium.solution.u).u)) for equilibrium in matches]
equilibriumσratios = [equilibrium.solution.parameters.signal.σ / signal.σ for equilibrium in matches]
emissionlevels = reduce(vcat, [reshape(projection.cumulativeemissions[end, :], 1, 2) for projection in projections])
completionlevels = reduce(vcat, [reshape(projection.nearzero[end, :], 1, 2) for projection in projections])
welfarelevels = hcat(uvalues .- u₀, uvalues .+ getproperty.(projections, :welfareloss) .- u₀) .* (100 / ȳ)

if length(matches) > 1
    comparisonseries = (emissionlevels, completionlevels, welfarelevels)
    comparisontitles = ("Emissions over the transition", "Probability of near-zero", "Household welfare cost")
    comparisonlabels = ("Expected cumulative GtCO₂e", Printf.@sprintf("Probability of %.0f%% abatement", 100nearzeroshare), "NPV cost [% of one year's output]")
    comparisonnames = ("emissions", "near-zero", "welfare")
    for panel in 1:3
        figure = CairoMakie.Figure(size = figuresize, backgroundcolor = background)
        axis = CairoMakie.Axis(figure[1, 1]; xlabel = L"Signal noise $\sigma/\sigma_0$", ylabel = comparisonlabels[panel], title = comparisontitles[panel], xticks = equilibriumσratios)
        if panel == 2
            axis.ytickformat = percenttickformat
            CairoMakie.ylims!(axis, 0, 1)
        end

        for case in 1:2
            CairoMakie.scatterlines!(axis, equilibriumσratios, comparisonseries[panel][:, case]; color = colours[case], linewidth = publicationdefault(:medianlinewidth), markersize = 9, label = cases[case])
        end

        CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
        CairoMakie.Label(figure[0, 1], "Market noise, credibility and the transition"; fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)
        CairoMakie.Label(figure[3, 1], "Each point uses an equilibrium solved for its signal noise; all other settings are matched"; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
        savepublicationfigure(joinpath(figurepath, "ets-equilibrium-comparison-$(comparisonnames[panel])"), figure)
    end
else
    @info "One matching noise calibration is available; the ETS figure reports the fixed-forecast information channel. Solve and save additional σ calibrations for the equilibrium comparison."
end

## Quantitative summary

open(joinpath(figurepath, "tax-regime-grid.csv"), "w") do file
    println(file, "signalnoise,time,announcement,staticminimum,staticmaximum,taxminimum,taxmaximum,beliefspread,abatementspread,intrinsicpoolinggridshare,observedpoolinggridshare,reputationaffectedgridshare,poolingruleerror,boundaryruleerror")
    for equilibrium in matches, value in equilibrium.solution.u
        p = equilibrium.solution.parameters
        preferred, policygap, preferencegap, wedge, flags = reputationtaxdiagnostics(equilibrium, value)
        taxes = view(value.τ, :, 1:(size(value.τ, 2) - 1))
        active = .!view(flags, :, 5)
        pooling = view(flags, :, 1)
        poolingerror = any(pooling) ? maximum(abs, vec(policygap)[pooling]) : NaN
        announcedtax = p.τᶜ(value.t)
        boundaryerror = maximum(abs, taxes[[1, end], :] .- reshape(min.(preferred[1:end-1], announcedtax), 1, :))
        beliefspread = maximum(maximum(view(taxes, :, j)) - minimum(view(taxes, :, j)) for j in axes(taxes, 2))
        abatementspread = maximum(maximum(view(taxes, i, :)) - minimum(view(taxes, i, :)) for i in axes(taxes, 1))
        Printf.@printf(file, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n", p.signal.σ, value.t, announcedtax, minimum(preferred[1:end-1]), maximum(preferred[1:end-1]), minimum(taxes), maximum(taxes), beliefspread, abatementspread, sum(flags[active, 1]) / sum(active), sum(flags[active, 2]) / sum(active), sum(flags[active, 3]) / sum(active), poolingerror, boundaryerror)
    end
end

open(joinpath(figurepath, "tax-regime-visited.csv"), "w") do file
    println(file, "signalnoise,case,time,phase,capacitymass,intrinsicpoolingmass,observedpoolingmass,reputationaffectedmass,interiorpolicymass,beliefvariance,signedtaxwedge,absolutetaxwedge")
    for (index, projection) in enumerate(projections), case in 1:2, k in eachindex(projection.t)
        phase = projection.t[k] != tₛ ? "regular" : k > 1 && projection.t[k - 1] == tₛ ? "after" : "before"
        Printf.@printf(file, "%.10g,%s,%.10g,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n", matches[index].solution.parameters.signal.σ, cases[case], projection.t[k], phase, projection.regimes[k, 5, case], projection.regimes[k, 1, case], projection.regimes[k, 2, case], projection.regimes[k, 3, case], projection.regimes[k, 4, case], projection.learningvariance[k, case], projection.taxwedge[k, 1, case], projection.taxwedge[k, 2, case])
    end
end

open(joinpath(figurepath, "tax-regime-exposure.csv"), "w") do file
    println(file, "signalnoise,case,yearsbelowcapacity,intrinsicpoolingexposure,observedpoolingexposure,reputationaffectedexposure,averagecapacitymass,maximumrmsbeliefvolatility,meantaxwedgeUSD,meanabsolutetaxwedgeUSD")
    for (index, projection) in enumerate(projections), case in 1:2
        Δts = diff(projection.t)
        activeyears = sum(Δts .* (1 .- projection.regimes[2:end, 5, case]))
        exposure = [activeyears > 0 ? sum(Δts .* projection.regimes[2:end, column, case]) / activeyears : NaN for column in 1:3]
        capacitymass = sum(Δts .* projection.regimes[2:end, 5, case]) / last(projection.t)
        volatility = sqrt(maximum(view(projection.learningvariance, :, case)))
        meanwedge = sum(Δts .* projection.taxwedge[2:end, 1, case]) / (activeyears * taxfactor)
        absolutewedge = sum(Δts .* projection.taxwedge[2:end, 2, case]) / (activeyears * taxfactor)
        Printf.@printf(file, "%.10g,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n", matches[index].solution.parameters.signal.σ, cases[case], activeyears, exposure[1], exposure[2], exposure[3], capacitymass, volatility, meanwedge, absolutewedge)
        Printf.@printf("σ=%.4g, %s: %.2f%% of pre-capacity state-time pools; reputation changes tax over %.2f%%; mean absolute wedge %.4g USD/tCO₂e.\n", matches[index].solution.parameters.signal.σ, cases[case], 100exposure[2], 100exposure[3], absolutewedge)
    end
end

open(joinpath(figurepath, "reputation-comparison.csv"), "w") do file
    println(file, "signalnoise,case,terminalreputation,terminalabatementshare,cumulativeemissions,probabilitynearzero,npvcostpercentoutput")
    for (index, projection) in enumerate(projections), case in 1:2
        Printf.@printf(file, "%.10g,%s,%.10g,%.10g,%.10g,%.10g,%.10g\n", matches[index].solution.parameters.signal.σ, cases[case], projection.belief[end, case], projection.abatement[end, case] / firm.η, projection.cumulativeemissions[end, case], projection.nearzero[end, case], welfarelevels[index, case])
    end
end

Printf.@printf("Reputation loss: Δ cumulative emissions = %.6g GtCO₂e; Δ terminal abatement = %.6g percentage points; NPV household cost = %.6g%% of one year's output.\n", projection.cumulativeemissions[end, 2] - projection.cumulativeemissions[end, 1], 100(projection.abatement[end, 2] - projection.abatement[end, 1]) / firm.η, 100projection.welfareloss / ȳ)
@info "Saved reputation figures and comparison table" directory = figurepath
