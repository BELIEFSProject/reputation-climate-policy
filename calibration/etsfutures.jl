## Setup
using Revise

import XLSX
import DataFrames
import Dates
import Printf
import Statistics
import Optim
import CairoMakie
import LaTeXStrings: @L_str

includet("../scripts/plotting/publication.jl")
includet("locallevel.jl")

## Slide styling
# Moloch retains Beamer's default sans serif; use its Latin Modern counterpart.
slidefont = String(readchomp(`kpsewhich lmsans10-regular.otf`))
slideboldfont = String(readchomp(`kpsewhich lmsans10-bold.otf`))
isfile(slidefont) && isfile(slideboldfont) || error("Latin Modern Sans OpenType fonts could not be found in the TeX installation.")
slidesize = (600, 280)
slidetheme = merge(CairoMakie.Theme(fonts = CairoMakie.Attributes(regular = slidefont, bold = slideboldfont), fontsize = 18, Axis = CairoMakie.Attributes(titlesize = 16, xlabelsize = 18, ylabelsize = 18, xticklabelsize = 14, yticklabelsize = 14)), publicationtheme)

## Load data
etsdatapath = joinpath(@__DIR__, "data", "futures-table.xlsx")
# Closing prices are stored separately from volumes and bid/ask quotes.
data = DataFrames.DataFrame(XLSX.readtable(etsdatapath, "Close"))
data[!, 1] = Dates.Date.(data[!, 1])

for column in DataFrames.names(data)[2:end]
    # NaN preserves gaps where the workbook records an unavailable price as "NA".
    data[!, column] = map(x -> ismissing(x) || x == "NA" ? NaN : x isa Real ? Float64(x) : parse(Float64, x), data[!, column])
end

## Figure window
firstdate = Dates.Date(2021, 1, 1)
lastedate = Dates.Date(2026, 12, 31)
maturitynames = DataFrames.names(data)[2:end]
# Compute on all trading dates and maturities, then restrict only the visible axes.
dataplot = DataFrames.sort(data, 1)

## Plot
# Day offsets keep calendar labels explicit on the numerical plotting axis.
plotdates = dataplot[!, 1]
dayoffsets = Dates.value.(plotdates .- firstdate)
tickdates = collect(firstdate:Dates.Month(6):lastedate)
tickpositions = Dates.value.(tickdates .- firstdate)
ticklabels = Dates.format.(tickdates, "yyyy-mm")
minortickdates = collect(firstdate:Dates.Month(1):lastedate)
minortickpositions = Dates.value.(minortickdates .- firstdate)
colours = CairoMakie.resample_cmap(:viridis, length(maturitynames))

etsfigure = CairoMakie.with_theme(slidetheme) do
    figure = CairoMakie.Figure(size = slidesize, backgroundcolor = "#FAFAFA")
    axis = CairoMakie.Axis(figure[1, 1]; title = L"ETS Futures Prices $y^T_t$ by Maturity $T$", xlabel = "Date", ylabel = "Price € / tCO2e", backgroundcolor = "#FAFAFA", xticks = (tickpositions, ticklabels), xticklabelrotation = pi / 4)

    for (column, colour) in zip(maturitynames, colours)
        CairoMakie.lines!(axis, dayoffsets, dataplot[!, column]; color = colour, linewidth = publicationdefault(:samplepathlinewidth), label = column)
    end

    # CairoMakie.axislegend(axis; position = :lt, labelsize = 11)
    CairoMakie.xlims!(axis, 0, Dates.value(lastedate - firstdate))
    figure
end

CairoMakie.save("papers/presentations/img/future-raw.pdf", etsfigure)

## Exact time to maturity
# Match the R convention of assigning each December contract to 31 December.
maturitydates = [Dates.Date(parse(Int, last(split(column, '-'))), 12, 31) for column in maturitynames]
maxhorizon = 760
horizonmatrix = fill(NaN, length(plotdates), maxhorizon)

for (column, maturitydate) in zip(maturitynames, maturitydates)
    prices = dataplot[!, column]
    for i in eachindex(plotdates)
        isnan(prices[i]) && continue
        horizon = Dates.value(maturitydate - plotdates[i])
        1 <= horizon <= maxhorizon || continue
        horizonmatrix[i, horizon] = prices[i]
    end
end

coverage = vec(sum(.!isnan.(horizonmatrix); dims = 1))
horizonbands = 8
horizoncolormap = CairoMakie.cgrad(:viridis, horizonbands; categorical = true)
horizoncolours = [horizoncolormap[(h - 1) / max(maxhorizon - 1, 1)] for h in 1:maxhorizon]

horizonfigure = CairoMakie.with_theme(slidetheme) do
    figure = CairoMakie.Figure(size = slidesize, backgroundcolor = "#FAFAFA")
    axis = CairoMakie.Axis(figure[1, 1]; title = L"ETS Futures Price $x^h_t$ by Horizon $h$", xlabel = "Trading Date", ylabel = "Price (€/tCO₂)", backgroundcolor = "#FAFAFA", xticks = (tickpositions, ticklabels), xticklabelrotation = pi / 4, xminorticks = minortickpositions, xminorticksvisible = true)

    for h in 1:maxhorizon
        coverage[h] == 0 && continue
        available = .!isnan.(horizonmatrix[:, h])
        CairoMakie.scatter!(axis, dayoffsets[available], horizonmatrix[available, h]; color = horizoncolours[h], markersize = 4)
    end

    CairoMakie.Colorbar(figure[1, 2]; colormap = horizoncolormap, limits = (1, maxhorizon), label = L"Time to maturity $h$ [days]", labelsize = 16, ticklabelsize = 14, ticks = CairoMakie.LinearTicks(5), width = 10, spinewidth = 0)
    CairoMakie.xlims!(axis, 0, Dates.value(lastedate - firstdate))

    figure
end

CairoMakie.save("papers/presentations/img/future-horizons.pdf", horizonfigure)

## Common local level
locallevelfit = fithorizonlocallevel(horizonmatrix, plotdates)
Printf.@printf "Common-shock standard deviation: %.4f EUR/tCO2 per sqrt(calendar day)\n" sqrt(locallevelfit.processvariance)
Printf.@printf "Horizon observation-noise standard deviation: %.4f EUR/tCO2\n" sqrt(locallevelfit.observationvariance)
Printf.@printf "Local-level diffuse log likelihood: %.4f\n" locallevelfit.loglikelihood

locallevelfigure = CairoMakie.with_theme(slidetheme) do
    figure = CairoMakie.Figure(size = slidesize, backgroundcolor = "#FAFAFA")
    noiseaxis = CairoMakie.Axis(figure[1, 1]; title = L"Horizon noise $\sigma \text{d}z_t$", xlabel = "Trading Date", ylabel = "Price (€/tCO₂)", backgroundcolor = "#FAFAFA", xticks = (tickpositions, ticklabels), xticklabelrotation = pi / 4, xminorticks = minortickpositions, xminorticksvisible = true)
    levelcolour = horizoncolormap[0.5]

    CairoMakie.hlines!(noiseaxis, [0]; color = (levelcolour, 0.4), linewidth = publicationdefault(:samplepathlinewidth))
    for h in 1:maxhorizon
        coverage[h] == 0 && continue
        available = isfinite.(locallevelfit.noise[:, h])
        CairoMakie.scatter!(noiseaxis, dayoffsets[available], locallevelfit.noise[available, h]; color = horizoncolours[h], markersize = 4)
    end

    CairoMakie.Colorbar(figure[1, 2]; colormap = horizoncolormap, limits = (1, maxhorizon), label = L"Time to maturity $h$ [days]", labelsize = 16, ticklabelsize = 14, ticks = CairoMakie.LinearTicks(5), width = 10, spinewidth = 0)
    CairoMakie.xlims!(noiseaxis, 0, Dates.value(lastedate - firstdate))

    maxnoise = maximum(abs, filter(!isnan, locallevelfit.noise))
    CairoMakie.ylims!(noiseaxis, -maxnoise, maxnoise)

    figure
end

CairoMakie.save("papers/presentations/img/future-local-level.pdf", locallevelfigure)
