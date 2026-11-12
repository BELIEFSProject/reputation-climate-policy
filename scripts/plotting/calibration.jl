## Setup
using Revise

import DotEnv
import Printf
import Colors
import CairoMakie
import LaTeXStrings: @L_str

DotEnv.load!()

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/primitives/climate.jl")

includet("../../src/agents/households.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")

includet("publication.jl")
includet("colours.jl")

## Configuration

background = Colors.colorant"#FAFAFA"
figuresize = (960, 540)
CairoMakie.set_theme!(publicationtheme)
CairoMakie.update_theme!(backgroundcolor = background, Axis = (; backgroundcolor = background), Legend = (; backgroundcolor = background))

referencehorizon = 24. # Illustrative constant-rate completion, not the Ramsey stopping time.
τmax = 250. # USD/tCO₂e.

rootdirectory = normpath(joinpath(@__DIR__, "..", ".."))
plotpath = get(ENV, "PLOTPATH", joinpath(rootdirectory, "figures"))

household, firm, government, signal, climate = initmodels()
label = join(("calibration-v1", solutionlabel(household, firm, signal, climate), parameterstring(referencehorizon), parameterstring(τmax)), "_")
figurepath = joinpath(plotpath, "calibration-$(first(bytes2hex(SHA.sha256(label)), solutionhashlength))")
mkpath(figurepath)

n₀ = n(0, firm.a₀, household, firm)
output = y(n₀, firm)
τs = collect(range(0, min(τmax, inv(firm.η) / taxfactor); length = 301))
shares = collect(range(0, 1; length = 301))
abatement = firm.η .* shares
α₀ = (firm.η - firm.a₀) / referencehorizon
τsupply = household.r * c(firm) / output
τcontact = climate.γ * output
τpreferred = statictax(firm.a₀, household, firm, climate)

filenames = ("labour", "emissions", "welfare", "prices", "costs", "investment", "learning")
titles = ("Labour response to carbon policy", "Emissions and installed abatement", "The static welfare trade-off", "Tax incentives and the supporting price", "The cost of faster abatement", "Investment and its private valuation", "How quickly can policy types be distinguished?")
xlabels = ("Carbon tax [USD/tCO₂e]", "Installed abatement / capacity", "Carbon tax [USD/tCO₂e]", "Installed abatement / capacity", "Capacity installed per year", L"Investment valuation $q/c$", "Tax gap between types [USD/tCO₂e]")
ylabels = ("Reduction in labour [%]", "Emissions [GtCO₂e/year]", "Change in annual costs [% of zero-tax output]", "USD/tCO₂e", "Annual investment costs [% of zero-tax output]", "Capacity installed per year", "Years for a unit signal-to-noise ratio")
figures = [CairoMakie.Figure(size = figuresize, backgroundcolor = background) for _ in filenames]
plotaxes = [CairoMakie.Axis(figure[1, 1]; title, xlabel, ylabel) for (figure, title, xlabel, ylabel) in zip(figures, titles, xlabels, ylabels)]

## Labour supply

elasticities = sort(unique([inv(inversefrisch(household)), 0.5, 1.]))
colours = (defaultpalette[:abatement], defaultpalette[:committed], defaultpalette[:damages])
for (index, elasticity) in enumerate(elasticities)
    reference = Household(; ν = household.ν, ϕ = inv(elasticity), r = household.r)
    labour = [n(τ * taxfactor, firm.a₀, reference, firm) / n(0, firm.a₀, reference, firm) for τ in τs]
    baseline = elasticity == inv(inversefrisch(household))
    caption = Printf.@sprintf("Frisch %.2g%s", elasticity, baseline ? " (baseline)" : "")
    CairoMakie.lines!(plotaxes[1], τs, 100 .* (1 .- labour); color = colours[index], linestyle = baseline ? :solid : :dash, linewidth = publicationdefault(:medianlinewidth), label = caption)
end

## Emissions and the calibration reference

tplotaxes = (0., τpreferred, last(τs) * taxfactor)
for (index, τ) in enumerate(tplotaxes)
    emissions = [e(n(τ, a, household, firm), a, firm) for a in abatement]
    CairoMakie.lines!(plotaxes[2], shares, emissions; color = colours[index], linewidth = publicationdefault(:medianlinewidth), label = Printf.@sprintf("Tax %.0f USD/tCO₂e", τ / taxfactor))
end
CairoMakie.scatter!(plotaxes[2], [firm.a₀ / firm.η], [firm.e₀]; color = defaultpalette[:guide], markersize = 11, label = "Initial emissions reference")
plotaxes[2].xtickformat = percenttickformat

## Static welfare: no investment or continuation-value effects

labour = [n(τ * taxfactor, firm.a₀, household, firm) for τ in τs]
labourcost = [(v(nₜ, household) - y(nₜ, firm) - v(n₀, household) + output) / output for nₜ in labour]
damagecost = [d(e(nₜ, firm.a₀, firm), climate) * y(nₜ, firm) / output - d(e(n₀, firm.a₀, firm), climate) for nₜ in labour]
for (values, colour, caption) in zip((labourcost, damagecost, labourcost .+ damagecost), (defaultpalette[:committed], defaultpalette[:damages], defaultpalette[:abatement]), ("Labour distortion", "Change in climate damages", "Total cost change"))
    CairoMakie.lines!(plotaxes[3], τs, 100 .* values; color = colour, linewidth = publicationdefault(:medianlinewidth), label = caption)
end
CairoMakie.vlines!(plotaxes[3], [τpreferred / taxfactor]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth), label = "Static optimum")

## Prices near full abatement

# The static rule has a separate zero-rate convention exactly at capacity.
preferred = [statictax(a, household, firm, climate) / taxfactor for a in abatement[1:end-1]]
CairoMakie.lines!(plotaxes[4], shares[1:end-1], preferred; color = defaultpalette[:abatement], linewidth = publicationdefault(:medianlinewidth), label = "Preferred static tax")
CairoMakie.hlines!(plotaxes[4], [τcontact / taxfactor]; color = defaultpalette[:committed], linestyle = :dash, linewidth = publicationdefault(:guidelinewidth), label = "Pre-contact Ramsey price")
CairoMakie.hlines!(plotaxes[4], [τsupply / taxfactor]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth), label = "Permanent supporting price")
plotaxes[4].xtickformat = percenttickformat

## Installation and adjustment costs

rates = collect(range(0, max(0.05, 1.25α₀ / firm.η); length = 301))
installation = [c(firm) * rate * firm.η / output for rate in rates]
adjustment = [firm.ξ * (rate * firm.η)^2 / (2output) for rate in rates]
totalcost = [k(rate * firm.η, firm) / output for rate in rates]
for (values, colour, caption) in zip((installation, adjustment, totalcost), (defaultpalette[:committed], defaultpalette[:damages], defaultpalette[:abatement]), ("Installation", "Adjustment", "Total"))
    CairoMakie.lines!(plotaxes[5], rates, 100 .* values; color = colour, linewidth = publicationdefault(:medianlinewidth), label = caption)
end
CairoMakie.vlines!(plotaxes[5], [α₀ / firm.η]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth), label = Printf.@sprintf("Constant progress over %.0f years", referencehorizon))
plotaxes[5].xtickformat = percenttickformat

## Valuation and irreversible investment

qs = collect(range(0, 1 + firm.ξ * last(rates) * firm.η / c(firm); length = 301))
investment = [α(q * c(firm), firm) / firm.η for q in qs]
CairoMakie.lines!(plotaxes[6], qs, investment; color = defaultpalette[:abatement], linewidth = publicationdefault(:medianlinewidth), label = "Optimal investment")
CairoMakie.vlines!(plotaxes[6], [1]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth), label = L"Investment threshold $q=c$")
CairoMakie.hlines!(plotaxes[6], [α₀ / firm.η]; color = defaultpalette[:committed], linestyle = :dash, linewidth = publicationdefault(:guidelinewidth), label = Printf.@sprintf("Constant progress over %.0f years", referencehorizon))
plotaxes[6].ytickformat = percenttickformat

## Information content of a fixed tax gap

gaps = exp10.(range(-1, log10(τmax); length = 301))
noises = signal.σ .* (0.5, 1., 2.)
plotaxes[7].xscale = log10
plotaxes[7].yscale = log10
for (index, σ) in enumerate(noises)
    # Signal drift separation is ϵ*Δτ; its noise variance over t years is σ²*t.
    times = [(σ / (signal.ϵ * gap * taxfactor))^2 for gap in gaps]
    CairoMakie.lines!(plotaxes[7], gaps, times; color = colours[index], linestyle = index == 2 ? :solid : :dash, linewidth = publicationdefault(:medianlinewidth), label = Printf.@sprintf("σ = %.3g%s", σ, index == 2 ? " (baseline)" : ""))
end

## Export

captions = (
    Printf.@sprintf("Initial abatement fixed; ν/A = %.3g; zero-tax output = %.3g trillion USD/year", household.ν / firm.A, output),
    Printf.@sprintf("Zero-tax gross emissions = %.3g GtCO₂e/year; initial installed share = %.1f%%", e(n₀, 0, firm), 100firm.a₀ / firm.η),
    Printf.@sprintf("Relative to zero tax at initial abatement; γ = %.5g; investment held at zero", climate.γ),
    Printf.@sprintf("Capacity endpoint excluded; pre-contact minus supporting price = %.2f USD/tCO₂e", (τcontact - τsupply) / taxfactor),
    Printf.@sprintf("c = %.3g, ξ = %.3g; reference duration is illustrative, not a solved transition", c(firm), firm.ξ),
    L"$\alpha(q)=\max\{q-c,0\}/\xi$; investment is zero below the installation price",
    L"Fixed tax gap: $t=[\sigma/(\epsilon\Delta\tau)]^2$; no endogenous policy adjustment",
)

for (figure, axis, name, caption) in zip(figures, plotaxes, filenames, captions)
    CairoMakie.Legend(figure[2, 1], axis; orientation = :horizontal, tellwidth = false)
    CairoMakie.Label(figure[3, 1], caption; fontsize = publicationdefault(:annotationfontsize), color = defaultpalette[:guide], tellwidth = false)
    savepublicationfigure(joinpath(figurepath, name), figure)
end

open(joinpath(figurepath, "parameters.csv"), "w") do file
    println(file, "parameter,value")
    for (name, value) in (("A", firm.A), ("nu", household.ν), ("inversefrisch", inversefrisch(household)), ("r", household.r), ("eta", firm.η), ("a0", firm.a₀), ("e0", firm.e₀), ("kappa", firm.κ), ("xi", firm.ξ), ("c", c(firm)), ("gamma", climate.γ), ("epsilon", signal.ϵ), ("sigma", signal.σ), ("output", output), ("initialemissions", e(n₀, firm.a₀, firm)), ("initialdamagefraction", d(e(n₀, firm.a₀, firm), climate)), ("supportingtaxUSD", τsupply / taxfactor), ("precontacttaxUSD", τcontact / taxfactor), ("statictaxUSD", τpreferred / taxfactor), ("referencehorizon", referencehorizon))
        println(file, name, ",", value)
    end
end

@info "Saved calibration figures" directory = figurepath output = output emissions = e(n₀, firm.a₀, firm) supportingtax = τsupply / taxfactor
