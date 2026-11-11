struct ReputationEquilibrium{S}
    solution::S
    filename::String
end

struct ReputationProjection{V, M, R, T}
    t::V
    belief::M
    tax::M
    expectedtax::M
    investment::M
    abatement::M
    emissions::M
    cumulativeemissions::M
    nearzero::M
    regimes::R
    taxwedge::R
    learningvariance::M
    welfareloss::T
end

struct ReputationTiming{V, T}
    t::V
    progress::V
    emissions::V
    terminalabatement::V
    terminalemissions::V
    cumulativeemissions::V
    baselineabatement::T
    baselineemissions::T
    baselinecumulativeemissions::T
    stationary::Bool
end

"Regimes: intrinsic pooling, near-announcement policy, reputation wedge, interior policy, capacity."
function reputationtaxregimes(a, τ, mandate, parameters, tolerance)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    if a ≥ firm.η
        return false, false, false, false, true
    end
    preferred = (2climate.γ * firm.A^2 / household.ν) / (1 + 2climate.γ * firm.A^2 * (firm.η - a) / household.ν)
    return mandate ≤ preferred, abs(mandate - τ) ≤ tolerance, abs(τ - min(preferred, mandate)) > tolerance, tolerance < τ < mandate - tolerance, false
end

"Maps use the stored policy; the indeterminate static tax at full capacity is excluded."
function reputationtaxdiagnostics(equilibrium, value; tolerance = max(10equilibrium.solution.parameters.policytolerance * equilibrium.solution.parameters.taxscale, 32eps(eltype(value.τ)) * max(1, equilibrium.solution.parameters.taxscale)))
    parameters = equilibrium.solution.parameters
    household, firm, climate, grid = parameters.model.household, parameters.model.firm, parameters.model.climate, parameters.grid
    mandate = parameters.τᶜ(value.t)
    preferred = [j == size(grid, 2) ? NaN : (2climate.γ * firm.A^2 / household.ν) / (1 + 2climate.γ * firm.A^2 * (firm.η - physicalstate(grid, 1, j)[2]) / household.ν) for j in 1:size(grid, 2)]
    policygap, preferencegap, reputationwedge = (similar(value.τ) for _ in 1:3)
    regimes = falses(length(value.τ), 5)
    for j in 1:size(grid, 2), i in 1:size(grid, 1)
        a = physicalstate(grid, i, j)[2]
        node = i + (j - 1) * size(grid, 1)
        regimes[node, :] .= reputationtaxregimes(a, value.τ[i, j], mandate, parameters, tolerance)
        policygap[i, j] = j == size(grid, 2) ? NaN : mandate - value.τ[i, j]
        preferencegap[i, j] = mandate - preferred[j]
        reputationwedge[i, j] = value.τ[i, j] - min(preferred[j], mandate)
    end
    return preferred, policygap, preferencegap, reputationwedge, regimes
end

function loadreputationequilibrium(filename)
    return JLD2.jldopen(filename, "r") do file
        if !haskey(file, "noncommitted")
            return nothing
        end
        entries = collect(keys(file["noncommitted"]))
        if length(entries) != 1
            return nothing
        end
        key = "noncommitted/$(only(entries))"
        solution = file["$key/solution"]
        if !(solution.parameters isa NonCommittedDAEParameters) || key != noncommittedsolutionkey(solution)
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

function reputationweights(grid, φ, a)
    zφ = clamp(normalise(φ, grid.φ), 0, 1) * (size(grid, 1) - 1)
    za = clamp(normalise(a, grid.A), 0, 1) * (size(grid, 2) - 1)
    i, j = min(floor(Int, zφ) + 1, size(grid, 1) - 1), min(floor(Int, za) + 1, size(grid, 2) - 1)
    wφ, wa = zφ - (i - 1), za - (j - 1)
    indices = (CartesianIndex(i, j), CartesianIndex(i + 1, j), CartesianIndex(i, j + 1), CartesianIndex(i + 1, j + 1))
    weights = ((1 - wφ) * (1 - wa), wφ * (1 - wa), (1 - wφ) * wa, wφ * wa)
    return indices, weights
end

function reputationinitialmass(grid, φ, a)
    mass = zeros(typeof(grid.h), size(grid))
    indices, weights = reputationweights(grid, φ, a)
    for (index, weight) in zip(indices, weights)
        mass[index] += weight
    end
    return vec(mass)
end

"A negative public signal multiplies the posterior odds of commitment by oddsratio."
function reputationlossoperator(grid, oddsratio)
    rows, columns, weights = Int[], Int[], typeof(grid.h)[]
    for j in 1:size(grid, 2), i in 1:size(grid, 1)
        φ = physicalstate(grid, i, j)[1]
        posterior = oddsratio * φ / (1 - φ + oddsratio * φ)
        z = clamp(normalise(posterior, grid.φ), 0, 1) * (size(grid, 1) - 1)
        lower = min(floor(Int, z) + 1, size(grid, 1) - 1)
        weight = z - (lower - 1)
        node = i + (j - 1) * size(grid, 1)
        for (destination, probability) in ((lower, 1 - weight), (lower + 1, weight))
            push!(rows, destination + (j - 1) * size(grid, 1))
            push!(columns, node)
            push!(weights, probability)
        end
    end
    nodes = prod(size(grid))
    return SP.sparse(rows, columns, weights, nodes, nodes)
end

"The forward generator is the transpose of the solved LF chain, conditional on the true type."
function reputationoperator(equilibrium, t; side = :right, nearzeroshare = 0.95, committed = false)
    solution = equilibrium.solution
    parameters = solution.parameters
    model, grid = parameters.model, parameters.grid
    household, firm = model.household, model.firm
    value = solution(t)
    x = reshape(noncommitteddaestate(value, parameters), 4, size(grid)...)
    mandate = parameters.τᶜ(t; side)
    nodes = prod(size(grid))
    rows, columns, rates = Int[], Int[], eltype(value.q)[]
    sizehint!(rows, 4nodes)
    sizehint!(columns, 4nodes)
    sizehint!(rates, 4nodes)
    objects = Matrix{eltype(value.q)}(undef, nodes, 18)
    tolerance = max(10parameters.policytolerance * parameters.taxscale, 32eps(eltype(value.q)) * max(1, parameters.taxscale))
    Δφ, Δa = grid.h * span(grid.φ), grid.h * span(grid.A)

    for j in 1:size(grid, 2), i in 1:size(grid, 1)
        node = i + (j - 1) * size(grid, 1)
        φ, a = physicalstate(grid, i, j)
        τ, _ = noncommitteddaetax(x, parameters.taxscale * x[4, i, j], i, j, mandate, parameters; equilibrium = true)
        τᵉ = noncommittedexpectedtax(φ, τ, mandate)
        investment = j == size(grid, 2) ? 0 : α(value.q[i, j], firm)
        μᵩ, σᵩ = noncommittedbeliefmoments(τ, mandate, φ, parameters.signal)
        B = φ * (1 - φ) * (parameters.signal.ϵ / parameters.signal.σ)^2 * (mandate - τ)
        if committed
            μᵩ = B * (mandate - τᵉ)
        end
        variance = max(σᵩ^2 / (2Δφ^2), B * max(τᵉ, mandate - τᵉ) / (2Δφ))
        λ⁺ = i == size(grid, 1) ? 0 : max(variance + μᵩ / (2Δφ), 0)
        λ⁻ = i == 1 ? 0 : max(variance - μᵩ / (2Δφ), 0)
        λₐ = investment / Δa
        for (destination, rate) in ((node + 1, λ⁺), (node - 1, λ⁻), (node + size(grid, 1), λₐ))
            if rate > 0
                push!(rows, destination)
                push!(columns, node)
                push!(rates, rate)
            end
        end
        push!(rows, node)
        push!(columns, node)
        push!(rates, -(λ⁺ + λ⁻ + λₐ))
        # Regime diagnostics refer to the conjectured non-committed policy under either true type.
        regimes = reputationtaxregimes(a, τ, mandate, parameters, tolerance)
        preferred = (2model.climate.γ * firm.A^2 / household.ν) / (1 + 2model.climate.γ * firm.A^2 * (firm.η - a) / household.ν)
        wedge = a ≥ firm.η ? 0 : τ - min(preferred, mandate)
        labour = n(τᵉ, a, household, firm)
        actualtax = committed ? mandate : τ
        objects[node, :] .= (φ, actualtax, τᵉ, investment, a, e(labour, a, firm), a ≥ nearzeroshare * firm.η, regimes..., σᵩ^2, wedge, abs(wedge), value.q[i, j], labour, y(labour, firm))
    end
    return SP.sparse(rows, columns, rates, nodes, nodes), objects, value.u
end

"Columns: belief, tax, expected tax, a, q, investment, labour, output, emissions, cumulative emissions; cases: known commitment, uncertain committed, uncertain non-committed."
function simulatecommitment(equilibrium; initialbelief = 0.9, timestep = 0.25)
    solution = equilibrium.solution
    parameters = solution.parameters
    grid, firm = parameters.grid, parameters.model.firm
    horizon = last(solution.t)
    path = parameters.τᶜ
    jumps = unique([path.ts[k] for k in 1:(length(path.ts) - 1) if path.ts[k] == path.ts[k + 1] && path.taxpath[k] != path.taxpath[k + 1] && 0 < path.ts[k] ≤ horizon])
    mesh = sort!(unique!(vcat(collect(range(0, horizon; length = ceil(Int, horizon / timestep) + 1)), solution.t, path.ts)))
    timeprecision = 32eps(typeof(horizon)) * max(1, horizon)
    for date in jumps
        mesh[abs.(mesh .- date) .≤ timeprecision] .= date
    end
    sort!(unique!(mesh))
    mesh = mesh[vcat(true, diff(mesh) .> timeprecision)]
    mesh[end] = horizon
    times = sort!(vcat(mesh, jumps))
    T = eltype(first(solution.u).q)
    trajectories = zeros(T, length(times), 10, 3)
    initialmass = reputationinitialmass(grid, initialbelief, firm.a₀)
    probability = hcat(reputationinitialmass(grid, 1, firm.a₀), initialmass, initialmass)
    cumulative = zeros(T, 3)
    columns = (1, 2, 3, 5, 16, 4, 17, 18, 6)
    row = 0

    for k in eachindex(mesh)
        t = mesh[k]
        phases = t in jumps ? 2 : 1
        for committed in (true, false)
            cases = committed ? (1:2) : (3:3)
            generator, objects, _ = reputationoperator(equilibrium, t; side = :left, committed)
            if k > 1
                Δt = t - mesh[k - 1]
                probability[:, cases] = (LA.I - Δt * generator) \ view(probability, :, cases)
                for case in cases
                    mass = view(probability, :, case)
                    if minimum(mass) < -1e-10 || abs(sum(mass) - 1) > 1e-8
                        error("Commitment-comparison probabilities lost positivity or mass at t=$t, case=$case.")
                    end
                    mass .= max.(mass, 0)
                    mass ./= sum(mass)
                    cumulative[case] += Δt * LA.dot(view(objects, :, 6), mass)
                end
            end
            for phase in 1:phases
                if phase == 2
                    _, objects, _ = reputationoperator(equilibrium, t; committed)
                end
                for case in cases
                    for (column, objectcolumn) in enumerate(columns)
                        trajectories[row + phase, column, case] = LA.dot(view(objects, :, objectcolumn), view(probability, :, case))
                    end
                    trajectories[row + phase, 10, case] = cumulative[case]
                end
            end
        end
        row += phases
    end
    return times, trajectories
end

"Expected paths before and after unexpected news; cumulative emissions are a passive outcome."
function projectreputation(equilibrium; initialbelief = 0.75, losstime = 10.0, oddsratio = 0.25, timestep = 0.25, nearzeroshare = 0.95)
    solution = equilibrium.solution
    parameters = solution.parameters
    household, firm, grid = parameters.model.household, parameters.model.firm, parameters.grid
    horizon = last(solution.t)
    if !(0 < initialbelief < 1 && 0 < losstime < horizon && 0 < oddsratio < 1 && timestep > 0 && 0 < nearzeroshare ≤ 1)
        throw(ArgumentError("The prior, news date, odds ratio, projection step and abatement threshold must be admissible."))
    end
    timeprecision = 32eps(typeof(horizon)) * max(1, horizon)
    mesh = sort!(unique!(vcat(collect(range(0, horizon; length = ceil(Int, horizon / timestep) + 1)), solution.t, parameters.τᶜ.ts, losstime)))
    mesh = sort!(unique!([isapprox(t, losstime; rtol = 0, atol = timeprecision) ? losstime : t for t in mesh]))
    mesh = mesh[vcat(true, diff(mesh) .> timeprecision)]
    mesh[end] = horizon
    times = sort!(vcat(mesh, losstime))
    T = eltype(first(solution.u).q)
    belief, tax, expectedtax, investment, abatement, emissions, nearzero = ntuple(_ -> zeros(T, length(times), 2), 7)
    cumulativeemissions = zeros(T, length(times), 2)
    regimes = zeros(T, length(times), 5, 2)
    taxwedge = zeros(T, length(times), 2, 2)
    learningvariance = zeros(T, length(times), 2)
    metrics = (belief, tax, expectedtax, investment, abatement, emissions, nearzero)
    initialmass = reputationinitialmass(grid, initialbelief, firm.a₀)
    probability = hcat(initialmass, initialmass)
    lossoperator = reputationlossoperator(grid, oddsratio)
    cumulative = zeros(T, 2)
    welfareloss = zero(T)
    row = 0

    for k in eachindex(mesh)
        t = mesh[k]
        generator, objects, householdvalue = reputationoperator(equilibrium, t; side = :left, nearzeroshare)
        if k > 1
            Δt = t - mesh[k - 1]
            probability = (LA.I - Δt * generator) \ probability
            if minimum(probability) < -1e-10 || maximum(abs, vec(sum(probability; dims = 1)) .- 1) > 1e-8
                error("Forward probabilities lost positivity or mass at t=$t.")
            end
            probability .= max.(probability, 0)
            probability ./= sum(probability; dims = 1)
            cumulative .+= Δt .* vec(transpose(view(objects, :, 6)) * probability)
        end
        if parameters.τᶜ(t; side = :left) != parameters.τᶜ(t)
            _, objects, householdvalue = reputationoperator(equilibrium, t; nearzeroshare)
        end
        for phase in 1:(t == losstime ? 2 : 1)
            if phase == 2
                before = LA.dot(view(probability, :, 2), vec(householdvalue))
                probability[:, 2] = lossoperator * view(probability, :, 2)
                welfareloss = exp(-household.r * t) * (LA.dot(view(probability, :, 2), vec(householdvalue)) - before)
            end
            row += 1
            means = transpose(objects) * probability
            for (metric, values) in zip(metrics, eachrow(means))
                metric[row, :] .= values
            end
            regimes[row, :, :] .= view(means, 8:12, :)
            taxwedge[row, :, :] .= view(means, 14:15, :)
            learningvariance[row, :] .= view(means, 13, :)
            cumulativeemissions[row, :] .= cumulative
        end
    end
    return ReputationProjection(times, belief, tax, expectedtax, investment, abatement, emissions, cumulativeemissions, nearzero, regimes, taxwedge, learningvariance, welfareloss)
end

"Vary news timing on a common forward mesh; each probability column receives exactly one shock."
function reputationtiming(equilibrium, reference; shocks = 21, initialbelief = 0.9, oddsratio = 0.1, timestep = 0.25, nearzeroshare = 0.95)
    solution = equilibrium.solution
    parameters = solution.parameters
    firm, grid = parameters.model.firm, parameters.grid
    horizon = last(solution.t)
    referenceprogress = (reference.abatement[:, 1] .- firm.a₀) ./ (reference.abatement[end, 1] - firm.a₀)
    dates = sort!(unique([reference.t[argmin(abs.(referenceprogress .- progress))] for progress in range(0, 1; length = shocks)]))
    timeprecision = 32eps(typeof(horizon)) * max(1, horizon)
    mesh = sort!(unique!(vcat(collect(range(0, horizon; length = ceil(Int, horizon / timestep) + 1)), solution.t, parameters.τᶜ.ts, dates)))
    for date in dates
        mesh[abs.(mesh .- date) .≤ timeprecision] .= date
    end
    sort!(unique!(mesh))
    mesh = mesh[vcat(true, diff(mesh) .> timeprecision)]
    mesh[end] = horizon
    T = eltype(first(solution.u).q)
    probability = repeat(reputationinitialmass(grid, initialbelief, firm.a₀), 1, length(dates) + 1)
    cumulative = zeros(T, length(dates) + 1)
    abatement, emissions = zeros(T, length(dates)), zeros(T, length(dates))
    lossoperator = reputationlossoperator(grid, oddsratio)

    for k in eachindex(mesh)
        t = mesh[k]
        generator, objects, _ = reputationoperator(equilibrium, t; side = :left, nearzeroshare)
        if k > 1
            Δt = t - mesh[k - 1]
            probability = (LA.I - Δt * generator) \ probability
            if minimum(probability) < -1e-10 || maximum(abs, vec(sum(probability; dims = 1)) .- 1) > 1e-8
                error("Shock-timing probabilities lost positivity or mass at t=$t.")
            end
            probability .= max.(probability, 0)
            probability ./= sum(probability; dims = 1)
            cumulative .+= Δt .* vec(transpose(view(objects, :, 6)) * probability)
        end
        if parameters.τᶜ(t; side = :left) != parameters.τᶜ(t)
            _, objects, _ = reputationoperator(equilibrium, t; nearzeroshare)
        end
        event = searchsortedfirst(dates, t)
        if event ≤ length(dates) && dates[event] == t
            abatement[event] = LA.dot(view(objects, :, 5), view(probability, :, 1))
            emissions[event] = LA.dot(view(objects, :, 6), view(probability, :, 1))
            probability[:, event + 1] = lossoperator * view(probability, :, event + 1)
        end
    end

    generator, objects, _ = reputationoperator(equilibrium, horizon; nearzeroshare)
    terminalabatement = vec(transpose(view(objects, :, 5)) * probability)
    terminalemissions = vec(transpose(view(objects, :, 6)) * probability)
    progress = (abatement .- firm.a₀) ./ (terminalabatement[1] - firm.a₀)
    # Beliefs may still move at capacity while abatement and emissions remain stationary.
    abatementdrift = maximum(abs, transpose(generator) * view(objects, :, 5)) / firm.η
    emissionsdrift = maximum(abs, transpose(generator) * view(objects, :, 6)) / firm.e₀
    stationary = max(abatementdrift, emissionsdrift) ≤ 1e-10
    return ReputationTiming(dates, progress, emissions, terminalabatement[2:end], terminalemissions[2:end], cumulative[2:end], terminalabatement[1], terminalemissions[1], cumulative[1], stationary)
end
