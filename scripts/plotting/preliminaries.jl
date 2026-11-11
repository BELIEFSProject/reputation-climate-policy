## Setup
using Revise

import DotEnv
DotEnv.load!()

import JLD2
import SciMLBase
import FastInterpolations as Itp
import LinearSolve
import SparseArrays
import StaticArrays as SA
import Statistics
import OrdinaryDiffEq as ODE
import UnPack: @unpack
import LaTeXStrings: @L_str
import Printf
import CairoMakie
import Colors

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/primitives/climate.jl")

includet("../../src/agents/households.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/dynamics/state.jl")
includet("../../src/dynamics/belief.jl")
includet("../../src/dynamics/firm.jl")
includet("../../src/dynamics/government.jl")

includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")

includet("../../src/solve/government/committed.jl")
includet("../../src/solve/government/noncommitted.jl")

includet("../../src/dynamics/simulation.jl")

includet("publication.jl")
includet("colours.jl")

CairoMakie.set_theme!(publicationtheme)

singlepanelsize = (420, 300)
combinedfiguresize = (900, 620)
plotpath = get(ENV, "PLOTPATH", "figures")
figurepath = joinpath(plotpath, "preliminaries")
ispath(figurepath) || mkpath(figurepath)
household, firm, government, signal, climate = initmodels()
## MAC curve
agrid = range(0, firm.e₀, 501)

begin
    macvalues = map(a -> c(a, firm), agrid)

    macfig = CairoMakie.Figure(size = singlepanelsize)
    macaxis = CairoMakie.Axis(
        macfig[1, 1];
        xlabel = L"Abatement $a_{i, t}$ [GtCO2e / year]",
        ylabel = "Output loss [% GDP / year]",
        limits = (extrema(agrid), (0, nothing)),
        ytickformat = values -> [Printf.@sprintf "%.2f%%" 100x for x in values],
        yticks = (0:0.25:2) ./ 100,
    )

    CairoMakie.lines!(macaxis, agrid, macvalues; color = defaultpalette[:mac], linewidth = publicationdefault(:medianlinewidth), label = L"Marginal abatement cost $c(a_{i, t})$")
    CairoMakie.axislegend(macaxis; position = :rt)
    savepublicationfigure(joinpath(figurepath, "marginal-abatement-costs"), macfig)

    macfig
end

## Calibration mechanisms
begin
    temperaturevalues = temperature.(mgrid, Ref(climate))
    initialtemperature = temperature(climate.m₀, climate)
    emissionsvalues = e.(agrid, Ref(firm))
    initialemissions = e(firm.a₀, firm)
    initialtax = sustainingtax(firm.a₀, firm)
    initialtaxdollars = initialtax / taxfactor
    netzerotaxdollars = sustainingtax(firm.e₀, firm) / taxfactor
    taxgrid = range(
        0.0,
        1.05 * max(initialtaxdollars, netzerotaxdollars);
        length = 501,
    )
    longrunabatement = min.(
        taxfactor .* taxgrid ./ (household.r * firm.κ),
        firm.e₀,
    )
    initialtaxabatement = min(
        initialtax / (household.r * firm.κ),
        firm.e₀,
    )
    φgrid = range(0.0, 1.0; length = 501)
    calibrationtaxgap = initialtax
    calibrationχ = χ(zero(calibrationtaxgap), calibrationtaxgap, signal)
    beliefdriftvalues = beliefdrift.(calibrationχ, φgrid)
    beliefdiffusionvalues = beliefdiffusion.(calibrationχ, φgrid)

    function plottemperaturecalibration!(position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Cumulative emissions $m$ [GtCO2e]",
            ylabel = "Temperature [°C]",
            title = L"Temperature response $T(m)$",
            limits = (extrema(mgrid), (0, nothing)),
            yticks = denseyticks,
        )
        CairoMakie.lines!(axis, mgrid, temperaturevalues; color = defaultpalette[:damages], linewidth = publicationdefault(:medianlinewidth), label = L"$T(m)=\zeta m$")
        CairoMakie.vlines!(axis, [climate.m₀]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.hlines!(axis, [initialtemperature]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.scatter!(axis, [climate.m₀], [initialtemperature]; color = defaultpalette[:damages], markersize = 14, strokewidth = 0, label = "Initial calibration")
        CairoMakie.axislegend(axis; position = :lt)

        axis
    end

    function plotemissionscalibration!(position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Abatement $a$ [GtCO2e/year]",
            ylabel = "Emissions [GtCO2e/year]",
            title = L"Residual emissions $e(a)$",
            limits = (extrema(agrid), (0, 1.05 * firm.e₀)),
            yticks = denseyticks,
        )
        CairoMakie.lines!(axis, agrid, emissionsvalues; color = defaultpalette[:emissions], linewidth = publicationdefault(:medianlinewidth), label = L"$e(a)=e_0-a$")
        CairoMakie.vlines!(axis, [firm.a₀]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.hlines!(axis, [initialemissions]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.scatter!(axis, [firm.a₀], [initialemissions]; color = defaultpalette[:emissions], markersize = 14, strokewidth = 0, label = "Initial calibration")
        CairoMakie.scatter!(axis, [firm.e₀], [0.0]; color = defaultpalette[:abatement], markersize = 14, strokewidth = 0, label = "Net zero")
        CairoMakie.axislegend(axis; position = :rt)

        axis
    end

    function plottaxabatementcalibration!(position)
        axis = CairoMakie.Axis(
            position;
            xlabel = "Carbon tax [USD/tCO2e]",
            ylabel = "Abatement [GtCO2e/year]",
            title = L"Long-run firm response $a^*(\tau)$",
            limits = (extrema(taxgrid), (0, 1.05 * firm.e₀)),
            yticks = denseyticks,
        )
        CairoMakie.lines!(axis, taxgrid, longrunabatement; color = defaultpalette[:committed], linewidth = publicationdefault(:medianlinewidth), label = L"$\tau=r_f c(a)$")
        CairoMakie.vlines!(axis, [initialtaxdollars]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.hlines!(axis, [firm.a₀, firm.e₀]; color = defaultpalette[:guide], linestyle = :dot, linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.scatter!(axis, [initialtaxdollars], [initialtaxabatement]; color = defaultpalette[:abatement], markersize = 14, strokewidth = 0, label = L"Initial sustaining tax")
        CairoMakie.text!(axis, last(taxgrid), firm.e₀; text = "Net zero", align = (:right, :bottom), offset = (0, 4), color = defaultpalette[:guide], fontsize = 14)
        CairoMakie.axislegend(axis; position = :lt)

        axis
    end

    function plotbeliefdriftcalibration!(position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Belief $\phi$",
            ylabel = "Belief drift",
            title = Printf.@sprintf(
                "Belief drift for a %.0f USD/tCO2e tax gap",
                calibrationtaxgap / taxfactor,
            ),
            limits = ((0, 1), nothing),
            xticks = 0:0.1:1,
            xtickformat = percenttickformat,
            yticks = denseyticks,
            ytickformat = values -> percentageformatter.(values),
        )
        CairoMakie.hlines!(axis, [0.0]; color = defaultpalette[:guide], linewidth = publicationdefault(:guidelinewidth))
        CairoMakie.lines!(axis, φgrid, beliefdriftvalues; color = defaultpalette[:damages], linewidth = publicationdefault(:medianlinewidth), label = L"$\mu_\phi(\phi)$")
        CairoMakie.axislegend(axis; position = :rb)

        axis
    end

    function plotbeliefdiffusioncalibration!(position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Belief $\phi$",
            ylabel = "Belief diffusion",
            title = Printf.@sprintf(
                "Belief diffusion for a %.0f USD/tCO2e tax gap",
                calibrationtaxgap / taxfactor,
            ),
            limits = ((0, 1), (0, nothing)),
            xticks = 0:0.1:1,
            xtickformat = percenttickformat,
            yticks = denseyticks,
            ytickformat = values -> percentageformatter.(values),
        )
        CairoMakie.lines!(axis, φgrid, beliefdiffusionvalues; color = defaultpalette[:committed], linewidth = publicationdefault(:medianlinewidth), label = L"$\sigma_\phi(\phi)$")
        CairoMakie.axislegend(axis; position = :rt)

        axis
    end

    calibrationfig = CairoMakie.Figure(size = combinedfiguresize)
    CairoMakie.Label(
        calibrationfig[0, 1:3],
        "Calibration mechanisms";
        fontsize = publicationdefault(:paneltitlefontsize),
        tellwidth = false,
    )
    plottemperaturecalibration!(calibrationfig[1, 1])
    plotemissionscalibration!(calibrationfig[1, 2])
    plottaxabatementcalibration!(calibrationfig[1, 3])
    beliefgrid = calibrationfig[2, 1:3] = CairoMakie.GridLayout()
    plotbeliefdriftcalibration!(beliefgrid[1, 1])
    plotbeliefdiffusioncalibration!(beliefgrid[1, 2])

    savepublicationfigure(
        joinpath(figurepath, "calibration-mechanisms"),
        calibrationfig,
    )

    individualpanels = (
        ("temperature-response", plottemperaturecalibration!),
        ("residual-emissions", plotemissionscalibration!),
        ("tax-abatement-response", plottaxabatementcalibration!),
        ("belief-drift", plotbeliefdriftcalibration!),
        ("belief-diffusion", plotbeliefdiffusioncalibration!),
    )
    for (filename, plotpanel!) in individualpanels
        panelfig = CairoMakie.Figure(size = singlepanelsize)
        plotpanel!(panelfig[1, 1])
        savepublicationfigure(joinpath(figurepath, filename), panelfig)
    end

    calibrationfig
end


## Non-committed tax coefficient
begin
    coefficienttaxmethod = OneShotTax()
    coefficientbeliefgrid = range(0.0, 1.0; length = 501)

    # At φ = 1/2, this upper bound makes the coefficient equal to one.
    maximumreputationvalue = 16 * (signal.σ / signal.ϵ)^2
    reputationvaluegrid = range(0.0, maximumreputationvalue; length = 501)
    reputationvaluebillions = 1_000 .* reputationvaluegrid

    taxcoefficientpercentage = [
        100 * inv(1 + household.r * government.δ / noncommittedtaxcoefficient(-p, φ, signal, coefficienttaxmethod))
        for φ in coefficientbeliefgrid, p in reputationvaluegrid
    ]
    coefficientlevels = range(0.0, 100.0; length = 11)

    coefficientfig = CairoMakie.Figure(size = (520, 340))
    coefficientaxis = CairoMakie.Axis(
        coefficientfig[1, 1];
        xlabel = L"Belief $\phi$",
        ylabel = L"Value of reputation $p$ [bn USD]",
        limits = ((0, 1), extrema(reputationvaluebillions)),
        xticks = 0:0.1:1,
        xtickformat = percenttickformat,
        yticks = denseyticks,
    )
    coefficientplot = CairoMakie.contourf!(
        coefficientaxis,
        coefficientbeliefgrid,
        reputationvaluebillions,
        taxcoefficientpercentage;
        levels = coefficientlevels,
        colormap = :YlGnBu_9,
    )
    CairoMakie.contour!(
        coefficientaxis,
        coefficientbeliefgrid,
        reputationvaluebillions,
        taxcoefficientpercentage;
        levels = coefficientlevels,
        color = (:white, 0.45),
        linewidth = 1.0,
    )
    CairoMakie.Colorbar(
        coefficientfig[1, 2],
        coefficientplot;
        label = L"Tax ratio $\tau_t / \tau^{\mathrm{c}}_t$",
        ticks = 0:20:100,
        tickformat = values -> [Printf.@sprintf "%.0f%%" value for value in values],
    )

    savepublicationfigure(
        joinpath(figurepath, "noncommitted-tax-coefficient"),
        coefficientfig,
    )

    coefficientfig
end
