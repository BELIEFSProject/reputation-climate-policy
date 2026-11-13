const publicationsettings = (;
    samplepathlinewidth = 1.0,
    medianlinewidth = 3.5,
    guidelinewidth = 2.0,
    samplepathopacity = 0.14,
    intervalopacity = 0.22,
    paneltitlefontsize = 20,
    annotationfontsize = 13,
)

const publicationtheme = CairoMakie.Theme(
    fontsize = 16,
    Axis = (;
        titlesize = 18,
        titlegap = 8,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xgridcolor = (:black, 0.08),
        ygridcolor = (:black, 0.08),
        topspinevisible = false,
        rightspinevisible = false,
    ),
    Legend = (;
        labelsize = 13,
        framevisible = false,
    ),
)

publicationdefault(key::Symbol) = getproperty(publicationsettings, key)
percenttickformat(values) = [Printf.@sprintf("%.0f%%", 100value) for value in values]

function publicationfigurepath(plotpath, household, firm, climate, experiment, runlabel)
    calibration = first(bytes2hex(SHA.sha256(solutionlabel(household, firm, climate))), solutionhashlength)
    run = first(bytes2hex(SHA.sha256(runlabel)), solutionhashlength)
    return joinpath(plotpath, "calibration-$calibration", "$experiment-$run")
end

function savepublicationfigure(basename, figure)
    CairoMakie.save("$basename.pdf", figure; pt_per_unit = 1)
    CairoMakie.save("$basename.png", figure; px_per_unit = 2)
    return figure
end
