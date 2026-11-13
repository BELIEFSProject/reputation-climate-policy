struct MarkovDynamics{S, F, T}
    solution::S
    fractions::F
    committed::Bool
    stop::T
end

"Physical observables (φ, τ, τᵉ, a, q, investment, labour, output, emissions) and signal separation χ."
function dynamicvalues(u, t, p::MarkovDynamics; side = t == p.stop ? :left : :right)
    solution = p.solution
    parameters = solution.parameters
    household, firm, signal, grid = parameters.model.household, parameters.model.firm, parameters.signal, parameters.grid
    φ = LE.logistic(u[1])
    a = clamp(u[2], firm.a₀, firm.η)
    τᶜ = parameters.τᶜ(t; side)

    k = min(searchsortedlast(solution.t, t), length(solution.t) - 1)
    if side == :left && k > 1 && t == solution.t[k] && parameters.τᶜ(t; side = :left) != parameters.τᶜ(t)
        k -= 1
    end

    left, right = solution.u[k], solution.u[k + 1]
    w = clamp((t - left.t) / (right.t - left.t), 0, 1)
    jump = parameters.τᶜ(right.t; side = :left) != parameters.τᶜ(right.t)
    wτ = jump && (t < right.t || side == :left) ? 0 : w
    zφ, za = normalise(φ, grid.φ) / grid.h, normalise(a, grid.A) / grid.h
    i, j = min(floor(Int, zφ) + 1, size(grid, 1) - 1), min(floor(Int, za) + 1, size(grid, 2) - 1)
    wφ, wa = zφ - (i - 1), za - (j - 1)
    q, fraction = zero(a), zero(a)

    @inbounds for dj in 0:1, di in 0:1
        weight = (di == 0 ? 1 - wφ : wφ) * (dj == 0 ? 1 - wa : wa)
        q += weight * ((1 - w) * left.q[i + di, j + dj] + w * right.q[i + di, j + dj])
        fraction += weight * ((1 - wτ) * p.fractions[k][i + di, j + dj] + wτ * p.fractions[k + 1][i + di, j + dj])
    end

    τⁿᶜ = a ≥ firm.η ? clamp(parameters.government.τ̄, 0, τᶜ) : τᶜ * clamp(fraction, 0, 1)
    τᵉ = ncexpectedtax(φ, τⁿᶜ, τᶜ)
    τ = p.committed ? τᶜ : τⁿᶜ
    investment = a ≥ firm.η ? 0 : α(q, firm)
    labour = n(τᵉ, a, household, firm)
    χ = signal.ϵ * (τᶜ - τⁿᶜ) / signal.σ

    return φ, τ, τᵉ, a, q, investment, labour, y(labour, firm), e(labour, a, firm), χ
end

function dynamicdrift!(du, u, p::MarkovDynamics, t)
    values = dynamicvalues(u, t, p)
    du[1] = (p.committed ? 1 : -1) * values[10]^2 / 2
    du[2] = values[6]
    du[3] = values[9]
    return nothing
end

function dynamicnoise!(du, u, p::MarkovDynamics, t)
    du[1] = dynamicvalues(u, t, p)[10]
    du[2] = 0
    du[3] = 0
    return nothing
end

"Simulate log odds, abatement and cumulative emissions; the output columns match simulatecommitment."
function simulatedynamics(equilibrium, φ₀s, ts; timestep = 0.01, seed = 2026, committed = false)
    solution = equilibrium.solution
    parameters = solution.parameters
    firm, path = parameters.model.firm, parameters.τᶜ
    if !all(φ -> 0 < φ < 1, φ₀s) || timestep ≤ 0
        throw(ArgumentError("Initial beliefs must lie in (0,1) and the simulation step must be positive."))
    end

    # Interpolating policy fractions preserves exact pooling between announcement knots.
    fractions = [iszero(path(value.t)) ? zero.(value.τ) : value.τ ./ path(value.t) for value in solution.u]
    jumps = unique([path.ts[k] for k in 1:(length(path.ts) - 1) if path.ts[k] == path.ts[k + 1] && path.taxpath[k] != path.taxpath[k + 1] && first(ts) < path.ts[k] ≤ last(ts)])
    stops = sort!(unique!(vcat(first(ts), jumps, last(ts))))
    times = sort!(unique!(vcat(ts, stops)))
    indices = [findall(t -> stops[k] ≤ t ≤ stops[k + 1], times) for k in 1:(length(stops) - 1)]
    problems = [SciMLBase.SDEProblem(dynamicdrift!, dynamicnoise!, [0, firm.a₀, 0], (stops[k], stops[k + 1]), MarkovDynamics(solution, fractions, committed, stops[k + 1])) for k in 1:(length(stops) - 1)]
    capacity = SciMLBase.DiscreteCallback((u, t, integrator) -> u[2] > firm.η, integrator -> (integrator.u[2] = firm.η); save_positions = (false, false))
    paths = zeros(typeof(float(firm.a₀)), length(ts), 10, length(φ₀s))

    for (trajectory, φ₀) in enumerate(φ₀s)
        u = [log(φ₀) - log1p(-φ₀), firm.a₀, 0]
        states = similar(paths, length(times), 3)
        for k in eachindex(problems)
            problem = SciMLBase.remake(problems[k]; u0 = u)
            sol = SDE.solve(problem, SDE.EM(); dt = timestep, adaptive = false, seed = seed + trajectory + length(φ₀s) * (k - 1), saveat = times[indices[k]], save_start = true, save_end = true, save_everystep = false, dense = false, callback = capacity)
            if !SciMLBase.successful_retcode(sol)
                error("Stochastic simulation failed for trajectory $trajectory on $(problem.tspan): $(sol.retcode).")
            end

            for (t, state) in zip(sol.t, sol.u)
                row = searchsortedfirst(times, t)
                states[row, :] .= state
            end
            u = copy(last(sol.u))
        end

        p = last(problems).p
        for (k, t) in enumerate(ts)
            u = view(states, searchsortedfirst(times, t), :)
            side = k < length(ts) && ts[k + 1] == t ? :left : :right
            values = dynamicvalues(u, t, p; side)
            for column in 1:9
                paths[k, column, trajectory] = values[column]
            end
            paths[k, 10, trajectory] = u[3]
        end
    end

    return paths
end
