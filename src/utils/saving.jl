committedfilename() = normpath(joinpath(@__DIR__, "..", "..", "data", "committed", "committed-" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS-sss") * ".jld2"))

"Save physical paths and model primitives without serialising solver closures."
function savecommitted(filename, solution, scale, model; valuations=nothing, valuationresiduals=nothing)
    if ispath(filename)
        error("Refusing to overwrite saved results: $(filename).")
    end
    mkpath(dirname(filename))
    states = map(u -> SA.SVector{8}(u) .* scale, solution.u)
    allocations = map(u -> first(committedpolicy(model, u[1], u[3], u[4], u[2], u[6], u[7], u[8]; zguess=solution.prob.p.parameters.zguess)), states)
    JLD2.jldopen(filename, "w") do file
        file["problem/φ₀"] = one(model.household.r)
        file["problem/horizon"] = solution.prob.p.horizon
        file["problem/tail"] = "Stationary allocation with interior dirty utilisation and partial compliance"
        file["solution/time"] = solution.t .* solution.prob.p.horizon
        file["solution/normalisedtime"] = solution.t
        file["solution/states"] = reduce(hcat, states)
        file["solution/scale"] = collect(scale)
        file["solution/order"] = ["xᶜ", "q", "vᵈ", "vᶜ", "λxᶜ", "λq", "λvᵈ", "λvᶜ"]
        file["diagnostics/retcode"] = string(solution.retcode)
        file["diagnostics/residual"] = maximum(abs, solution.resid)
        if !isnothing(valuations)
            file["diagnostics/valuations"] = collect(valuations)
        end
        if !isnothing(valuationresiduals)
            file["diagnostics/valuationresiduals"] = collect(valuationresiduals)
        end
        for name in fieldnames(eltype(allocations))
            file["solution/allocation/$(name)"] = map(a -> getfield(a, name), allocations)
        end
        for componentname in fieldnames(typeof(model))
            component = getfield(model, componentname)
            for name in fieldnames(typeof(component))
                file["parameters/$(componentname)/$(name)"] = getfield(component, name)
            end
        end
    end
    filename
end

"Reload a saved physical path as the existing (guess, scale) warm-start interface."
function loadcommittedguess(filename)
    time, states, scale = JLD2.jldopen(filename, "r") do file
        file["solution/normalisedtime"], file["solution/states"], SA.SVector{8}(file["solution/scale"])
    end
    guess = @closure s -> begin
        local j = clamp(searchsortedlast(time, s), 1, length(time) - 1)
        local weight = (s - time[j]) / (time[j + 1] - time[j])
        ((1 - weight) * SA.SVector{8}(view(states, :, j)) + weight * SA.SVector{8}(view(states, :, j + 1))) ./ scale
    end
    guess, scale
end
