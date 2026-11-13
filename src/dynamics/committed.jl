"Interpolate the saved valuation at known commitment, φ=1."
function committedvaluation(a, t, solution)
    grid = solution.parameters.grid
    k = clamp(searchsortedlast(solution.t, t), 1, length(solution.t) - 1)
    left, right = solution.u[k], solution.u[k + 1]
    weight = clamp((t - left.t) / (right.t - left.t), 0, 1)
    position = normalise(clamp(a, grid.A.lower, grid.A.upper), grid.A) / grid.h
    j = min(floor(Int, position) + 1, size(grid, 2) - 1)
    fraction = clamp(position - (j - 1), 0, 1)
    lower = (1 - weight) * left.q[end, j] + weight * right.q[end, j]
    upper = (1 - weight) * left.q[end, j + 1] + weight * right.q[end, j + 1]
    return (1 - fraction) * lower + fraction * upper
end

"Calendar-time characteristic for abatement and cumulative emissions, using q_h(t,1,a)."
function committedcharacteristic!(du, u, solution, t; stop = last(solution.t))
    household, firm = solution.parameters.model.household, solution.parameters.model.firm
    a = clamp(u[1], firm.a₀, firm.η)
    q = committedvaluation(a, t, solution)
    τ = solution.parameters.τᶜ(t; side = t == stop ? :left : :right)
    du[1] = a ≥ firm.η ? 0 : α(q, firm)
    du[2] = e(n(τ, a, household, firm), a, firm)
    return nothing
end

"Deterministic known-commitment path; columns match simulatecommitment."
function simulatecharacteristic(solution, ts; timestep = 0.05, tolerance = 1e-8)
    household, firm = solution.parameters.model.household, solution.parameters.model.firm
    path = solution.parameters.τᶜ
    jumps = [path.ts[k] for k in 1:(length(path.ts) - 1) if path.ts[k] == path.ts[k + 1] && path.taxpath[k] != path.taxpath[k + 1] && first(ts) < path.ts[k] < last(ts)]
    stops = sort!(unique!(vcat(first(ts), jumps, last(ts))))
    times = sort!(unique!(vcat(ts, stops)))
    states = Matrix{eltype(first(solution.u).q)}(undef, length(times), 2)
    initialstate = [firm.a₀, zero(firm.a₀)]
    capacity = SciMLBase.DiscreteCallback((u, t, integrator) -> u[1] > firm.η, integrator -> (integrator.u[1] = firm.η); save_positions = (false, false))

    for k in 1:(length(stops) - 1)
        left, right = stops[k], stops[k + 1]
        indices = findall(t -> left ≤ t ≤ right, times)
        drift! = (du, u, p, t) -> committedcharacteristic!(du, u, p, t; stop = right)
        problem = ODE.ODEProblem(drift!, initialstate, (left, right), solution)
        sol = ODE.solve(problem, ODE.Tsit5(); dtmax = timestep, abstol = tolerance, reltol = tolerance, saveat = times[indices], save_start = true, save_end = true, save_everystep = false, dense = false, callback = capacity)
        if !SciMLBase.successful_retcode(sol)
            error("Known-commitment characteristic failed on $(problem.tspan): $(sol.retcode).")
        end
        for (t, state) in zip(sol.t, sol.u)
            states[searchsortedfirst(times, t), :] .= state
        end
        initialstate = copy(last(sol.u))
    end

    values = zeros(eltype(states), length(ts), 10)
    for (k, t) in enumerate(ts)
        row = searchsortedfirst(times, t)
        a = clamp(states[row, 1], firm.a₀, firm.η)
        q = committedvaluation(a, t, solution)
        τ = path(t; side = k < length(ts) && ts[k + 1] == t ? :left : :right)
        labour = n(τ, a, household, firm)
        investment = a ≥ firm.η ? 0 : α(q, firm)
        values[k, :] .= (1, τ, τ, a, q, investment, labour, y(labour, firm), e(labour, a, firm), states[row, 2])
    end
    return values
end
