"Simulate x=(log odds, a, cumulative emissions), one ensemble member per initial belief."
function simulatedynamics(equilibrium, φ₀s, ts; timestep = 0.01, seed = 2026, committed = false)
    solution = equilibrium.solution
    firm, τᶜ = solution.parameters.model.firm, solution.parameters.τᶜ

    fractions = [iszero(τᶜ(v.t)) ? zero.(v.τ) : v.τ ./ τᶜ(v.t) for v in solution.u]
    jumps = [τᶜ.ts[i] for i in 1:(length(τᶜ.ts) - 1) if τᶜ.ts[i] == τᶜ.ts[i + 1] && τᶜ.taxpath[i] != τᶜ.taxpath[i + 1] && first(ts) < τᶜ.ts[i] < last(ts)]
    stops = sort!(unique!(vcat(first(ts), jumps, last(ts))))
    times = sort!(unique!(vcat(ts, stops)))

    N = length(φ₀s)
    x₀ = [MarkovState(promote(log(φ) - log1p(-φ), firm.a₀, zero(firm.a₀))...) for φ in φ₀s]
    states = Matrix{eltype(x₀)}(undef, length(times), N)
    capacity = SciMLBase.DiscreteCallback((x, t, integrator) -> x.a > firm.η, integrator -> (integrator.u.a = firm.η); save_positions = (false, false))

    for i in 1:(length(stops) - 1)
        indices = findall(t -> stops[i] ≤ t ≤ stops[i + 1], times)
        p = MarkovParameters(solution, fractions, committed, stops[i + 1])
        problem = SciMLBase.SDEProblem(μ!, σ!, copy(x₀[1]), (stops[i], stops[i + 1]), p)
        initial = (prob, ctx) -> SciMLBase.remake(prob; u0 = copy(x₀[ctx.sim_id]))
        ensemble = SciMLBase.EnsembleProblem(problem; prob_func = initial, safetycopy = false)

        paths = SDE.solve(ensemble, SDE.EM(), SciMLBase.EnsembleThreads(); trajectories = N, seed = seed + i - 1, dt = timestep, adaptive = false, saveat = times[indices], save_start = true, save_end = true, save_everystep = false, dense = false, callback = capacity)

        for j in 1:N
            path = paths.u[j]
            if !SciMLBase.successful_retcode(path)
                error("Stochastic trajectory $j failed on $(problem.tspan): $(path.retcode).")
            end

            for (t, x) in zip(path.t, path.u)
                states[searchsortedfirst(times, t), j] = copy(x)
            end
            x₀[j] = copy(last(path.u))
        end
    end

    values = zeros(eltype(first(x₀)), length(ts), 10, N)
    p = MarkovParameters(solution, fractions, committed, last(ts))

    for j in 1:N, (i, t) in enumerate(ts)
        x = states[searchsortedfirst(times, t), j]
        side = i < length(ts) && ts[i + 1] == t ? :left : :right
        values[i, 1:9, j] .= observables(x, t, p; side)[1:9]
        values[i, 10, j] = x.m
    end

    return values
end

"Repeat each prescribed prior within the same ensemble; retain the plotting array convention."
function simulateensembledynamics(equilibrium, φ₀s, ts; trajectories = 1_000, kwargs...)
    values = simulatedynamics(equilibrium, repeat(φ₀s; inner = trajectories), ts; kwargs...)
    return permutedims(reshape(values, length(ts), 10, trajectories, length(φ₀s)), (1, 2, 4, 3))
end
