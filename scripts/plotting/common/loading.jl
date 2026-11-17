function loadreputationequilibrium(filename)
    return JLD2.jldopen(filename, "r") do file
        if !haskey(file, "ncversion") || file["ncversion"] != ncversion || !haskey(file, "noncommitted")
            return nothing
        end
        entries = collect(keys(file["noncommitted"]))
        if length(entries) != 1
            return nothing
        end
        key = "noncommitted/$(only(entries))"
        solution = file["$key/solution"]
        if !(solution.parameters isa NCDAEParameters) || key != ncsolutionkey(solution)
            return nothing
        end
        return ReputationEquilibrium(solution, abspath(filename))
    end
end

"Find the saved Ramsey solution that generated this equilibrium's announcement."
function loadreputationbenchmark(equilibrium, directory)
    parameters = equilibrium.solution.parameters
    files = filter(path -> startswith(basename(path), "committed-simplified-") && endswith(path, ".jld2"), readdir(directory; join = true))
    sort!(files; by = mtime, rev = true)
    for filename in files
        benchmark = JLD2.jldopen(filename, "r") do file
            if !haskey(file, "committedtaxpath") || !haskey(file, "committed/solution")
                return nothing
            end
            path = file["committedtaxpath"]
            if path.ts != parameters.τᶜ.ts || path.taxpath != parameters.τᶜ.taxpath || path.tailtax != parameters.τᶜ.tailtax
                return nothing
            end
            solution = file["committed/solution"]
            model = solution.model
            if solutionlabel(model.household, model.firm, model.climate) != solutionlabel(parameters.model.household, parameters.model.firm, parameters.model.climate)
                return nothing
            end
            if !SciMLBase.successful_retcode(solution.retcode)
                error("The matching committed benchmark has return code $(solution.retcode): $filename")
            end
            return solution
        end
        if !isnothing(benchmark)
            return benchmark
        end
    end
    error("No saved committed benchmark matches the equilibrium's model and announced policy in $directory.")
end

function comparablereputationequilibrium(equilibrium, baseline)
    parameters, reference = equilibrium.solution.parameters, baseline.solution.parameters
    sameeconomy = solutionlabel(parameters.model.household, parameters.model.firm, parameters.model.climate) == solutionlabel(reference.model.household, reference.model.firm, reference.model.climate)
    sameannouncement = parameters.τᶜ.ts == reference.τᶜ.ts && parameters.τᶜ.taxpath == reference.τᶜ.taxpath && parameters.τᶜ.tailtax == reference.τᶜ.tailtax
    samenumerics = parameters.grid.h == reference.grid.h && parameters.timestep == reference.timestep && parameters.tolerance == reference.tolerance && parameters.policytolerance == reference.policytolerance && equilibrium.solution.t == baseline.solution.t
    return sameeconomy && sameannouncement && samenumerics && parameters.signal.ϵ == reference.signal.ϵ && parameters.government == reference.government
end

"Select one calibration and its comparable noise equilibria."
function loadequilibria(directory, model, government, signal; filename = "")
    files = filter(path -> startswith(basename(path), "noncommitted-simplified-") && endswith(path, ".jld2"), readdir(directory; join = true))
    sort!(files; by = mtime, rev = true)
    values = filter(!isnothing, loadreputationequilibrium.(files))

    label = solutionlabel(model.household, model.firm, signal, model.climate)
    candidates = filter(eq -> solutionlabel(eq.solution.parameters.model.household, eq.solution.parameters.model.firm, eq.solution.parameters.signal, eq.solution.parameters.model.climate) == label && eq.solution.parameters.government == government, values)
    baseline = isempty(filename) ? first(candidates) : loadreputationequilibrium(filename)

    matches = filter(eq -> comparablereputationequilibrium(eq, baseline), values)
    pushfirst!(matches, baseline)
    unique!(eq -> eq.solution.parameters.signal.σ, matches)
    sort!(matches; by = eq -> eq.solution.parameters.signal.σ)

    return baseline, matches
end
