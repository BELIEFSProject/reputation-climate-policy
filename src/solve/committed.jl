"Numerical controls for the full-information committed benchmark."
Base.@kwdef struct CommittedOptions
    horizon::Float64 = 400.0
    nodes::Int = 81
    tolerance::Float64 = 1e-8
    tailtolerance::Float64 = 1e-5
    gradientstep::Float64 = 1e-3
    iterations::Int = 60
end

struct CommittedState{T} <: StaticArrays.FieldVector{3, T}
    xc::T
    q::T
    vd::T
end

struct CommittedTax{T}
    grid::Vector{T}
    values::Vector{T}
end

function (tax::CommittedTax)(s)
    s <= first(tax.grid) && return first(tax.values)
    s >= last(tax.grid) && return last(tax.values)
    i = clamp(searchsortedlast(tax.grid, s), 1, length(tax.grid) - 1)
    weight = (s - tax.grid[i]) / (tax.grid[i + 1] - tax.grid[i])
    (1 - weight) * tax.values[i] + weight * tax.values[i + 1]
end

struct CommittedTransition{M, S, T}
    model::M
    tax::S
    horizon::T
    xc0::T
    scale::CommittedState{T}
end

struct CommittedPath{P, S, T}
    problem::P
    solution::S
    cost::T
    quadratureerror::T
end

struct CommittedDiagnostics{T}
    market::T
    credit::T
    complementarity::T
    investmentkkt::T
    dirtykkt::T
    interiorprofit::T
    irreversibility::T
    cleanvaluation::T
    dirtyvaluation::T
    boundary::T
    tailinvestment::T
    tailpricedrift::T
    tailprofitdrift::T
    minimumq::T
    minimumvd::T
    minimumxc::T
    schedulekkt::T
end

struct CommittedSolution{P, O, D}
    path::P
    optimisation::O
    diagnostics::D
end

committedstate(state, problem::CommittedTransition) = CommittedState(state[1] * problem.scale.xc, state[2] * problem.scale.q, state[3] * problem.scale.vd)

function committedallocation(path::CommittedPath, s)
    state = committedstate(path.solution(s), path.problem)
    committedallocation(path.problem.model, state.xc, state.vd, state.q, path.problem.tax(s))
end

function committedtrialallocation(state, problem, s)
    # Extend static prices to infeasible Newton iterates only; accepted paths are checked without projection.
    committedallocation(problem.model, max(state.xc, eps(problem.scale.xc)), max(state.vd, zero(state.vd)), state.q, problem.tax(s))
end

function committeddynamics(state, problem, s)
    physical = committedstate(state, problem)
    allocation = committedtrialallocation(physical, problem, s)
    r = problem.model.household.r
    CommittedState(problem.horizon * allocation.investment / problem.scale.xc, problem.horizon * (r * physical.q - allocation.p) / problem.scale.q, problem.horizon * (r * physical.vd - allocation.profit) / problem.scale.vd)
end

function committedboundary(states, problem, grid)
    initial = committedstate(states(first(grid)), problem)
    terminal = committedstate(states(last(grid)), problem)
    allocation = committedtrialallocation(terminal, problem, last(grid))
    r = problem.model.household.r
    CommittedState((initial.xc - problem.xc0) / problem.scale.xc, (terminal.q - allocation.p / r) / problem.scale.q, (terminal.vd - allocation.profit / r) / problem.scale.vd)
end

function committedguess(problem)
    model = problem.model
    terminal = committedstationary(model, last(problem.tax.values))
    finalxc = max(problem.xc0, terminal.xc)
    initial = committedallocation(model, problem.xc0, terminal.vd, model.clean.c, first(problem.tax.values))
    initialq = max(initial.p / model.household.r, model.clean.c)
    initialvd = initial.profit / model.household.r
    # Only the initial guess uses frozen-flow valuations; the BVP determines both initial asset prices.
    guess(parameters, s) = collect(CommittedState(((1 - s) * problem.xc0 + s * finalxc) / problem.scale.xc, ((1 - s) * initialq + s * model.clean.c) / problem.scale.q, ((1 - s) * initialvd + s * terminal.vd) / problem.scale.vd))
    guess
end

function committedintegral(path, flow; tolerance=1e-7)
    problem = path.problem
    integrand(s) = problem.horizon * exp(-problem.model.household.r * problem.horizon * s) * flow(committedallocation(path, s))
    QuadGK.quadgk(integrand, problem.tax.grid...; rtol=tolerance, atol=tolerance)
end

"Competitive transition for an announced schedule; time and private states are normalised separately."
function committedpath(model, xc0, tax::CommittedTax; options=CommittedOptions())
    scale = CommittedState(max(float(xc0), model.dirty.kᵈ), model.clean.c, model.dirty.γᵈ * model.dirty.kᵈ^2 / (2 * model.household.r))
    problem = CommittedTransition(model, tax, options.horizon, float(xc0), scale)
    bvp = SciMLBase.BVProblem(committeddynamics, committedboundary, committedguess(problem), (0.0, 1.0), problem)
    # Market-clearing roots and active constraints require numerical rather than automatic Jacobians.
    jacobian = BoundaryValueDiffEq.BVPJacobianAlgorithm(ADTypes.AutoFiniteDiff())
    solution = BoundaryValueDiffEq.solve(bvp, BoundaryValueDiffEq.MIRK4(; jac_alg=jacobian); dt=1 / (options.nodes - 1), abstol=options.tolerance, reltol=options.tolerance)
    SciMLBase.successful_retcode(solution) || error("Committed private-equilibrium BVP failed: $(solution.retcode).")
    sampled = map(s -> committedstate(solution(s), problem), range(0, 1; length=2 * options.nodes - 1))
    minimum(state -> state.xc, sampled) > zero(xc0) || error("Committed BVP returned non-positive clean capacity.")
    minimum(state -> min(state.q / scale.q, state.vd / scale.vd), sampled) >= -10 * options.tolerance || error("Committed BVP returned negative private valuations.")
    path = CommittedPath(problem, solution, 0.0, 0.0)
    terminal = committedallocation(path, 1.0)
    flowcost, quadratureerror = committedintegral(path, allocation -> allocation.welfare; tolerance=options.tolerance)
    # Freeze the terminal allocation: this tail is stationary only when terminal investment vanishes.
    tailcost = exp(-model.household.r * options.horizon) * terminal.welfare / model.household.r
    CommittedPath(problem, solution, flowcost + tailcost, quadratureerror)
end

"Finite differences of the reduced schedule objective, at a scale above the BVP tolerance."
function committedgradient!(gradient, objective, controls, step)
    trial = copy(controls)
    for i in eachindex(controls)
        lower = max(zero(step), controls[i] - step)
        upper = min(one(step), controls[i] + step)
        trial[i] = upper
        uppercost = objective(trial)
        trial[i] = lower
        lowercost = objective(trial)
        gradient[i] = (uppercost - lowercost) / (upper - lower)
        trial[i] = controls[i]
    end
    gradient
end

function committeddiagnostics(path; nodes=161, tolerance=1e-7, schedulekkt=NaN)
    problem = path.problem
    model = problem.model
    r = model.household.r
    grid = range(0, 1; length=nodes)
    allocations = map(s -> committedallocation(path, s), grid)
    firstallocation = first(allocations)
    terminal = last(allocations)
    market = maximum(a -> abs(a.x - a.xc - a.xd), allocations)
    credit = maximum(a -> max(a.L - a.barL, zero(a.L)), allocations)
    complementarity = maximum(a -> abs(a.μ * (a.barL - a.L)), allocations)
    investmentkkt = maximum(a -> max(max(a.q - model.clean.c - model.clean.ξ * a.investment, 0), abs(a.investment * (model.clean.c + model.clean.ξ * a.investment - a.q))), allocations)
    dirtykkt = maximum(i -> abs(allocations[i].xd - clamp(allocations[i].xd + allocations[i].p - model.dirty.η * problem.tax(grid[i]) - model.dirty.γᵈ * allocations[i].xd, 0, model.dirty.kᵈ)), eachindex(allocations))
    interiorprofit = maximum(a -> 0 < a.xd < model.dirty.kᵈ ? abs(a.profit - model.dirty.γᵈ * a.xd^2 / 2) : zero(a.profit), allocations)
    irreversibility = max(-minimum(diff(map(a -> a.xc, allocations))), 0)
    cleanflow = first(committedintegral(path, a -> a.p; tolerance))
    dirtyflow = first(committedintegral(path, a -> a.profit; tolerance))
    cleanvaluation = abs(firstallocation.q - cleanflow - exp(-r * problem.horizon) * terminal.q)
    dirtyvaluation = abs(firstallocation.vd - dirtyflow - exp(-r * problem.horizon) * terminal.vd)
    boundary = maximum(abs, committedboundary(path.solution, problem, (0.0, 1.0)))
    tailpricedrift = abs(terminal.p - allocations[end - 1].p) * (nodes - 1) / problem.horizon
    tailprofitdrift = abs(terminal.profit - allocations[end - 1].profit) * (nodes - 1) / problem.horizon
    CommittedDiagnostics(promote(market, credit, complementarity, investmentkkt, dirtykkt, interiorprofit, irreversibility, cleanvaluation, dirtyvaluation, boundary, terminal.investment, tailpricedrift, tailprofitdrift, minimum(a -> a.q, allocations), minimum(a -> a.vd, allocations), minimum(a -> a.xc, allocations), schedulekkt)...)
end

"Local optimisation over tax nodes, preserving all private-equilibrium constraint regimes."
function solvecommitted(model, xc0, taxgrid, initialtax; options=CommittedOptions())
    model.firm isa CESFirm && model.firm.ρ < 1 && !iszero(model.firm.ρ) || error("The committed solver requires CES production with ρ < 1 and ρ ≠ 0.")
    xc0 > 0 || error("The tax bound and initial solver require positive inherited clean capacity.")
    first(taxgrid) == 0 && last(taxgrid) == 1 && issorted(taxgrid) && all(>(0), diff(taxgrid)) || error("Tax nodes must increase strictly from zero to one in normalised time.")
    length(taxgrid) == length(initialtax) || error("Provide one initial tax per schedule node.")
    taxbound = committedtaxbound(model, xc0)
    controls = clamp.(initialtax ./ taxbound, 1e-4, 1 - 1e-4)
    objective(controls) = model.household.r * committedpath(model, xc0, CommittedTax(collect(float.(taxgrid)), taxbound .* controls); options).cost
    gradient!(gradient, controls) = committedgradient!(gradient, objective, controls, options.gradientstep)
    optimiser = Optim.Fminbox(Optim.LBFGS())
    settings = Optim.Options(iterations=options.iterations, outer_iterations=10, g_abstol=1e-4, outer_g_abstol=1e-4, f_reltol=1e-7)
    result = Optim.optimize(objective, gradient!, zeros(length(controls)), ones(length(controls)), controls, optimiser, settings)
    optimalcontrols = Optim.minimizer(result)
    path = committedpath(model, xc0, CommittedTax(collect(float.(taxgrid)), taxbound .* optimalcontrols); options)
    gradient = similar(optimalcontrols)
    gradient!(gradient, optimalcontrols)
    schedulekkt = maximum(abs, optimalcontrols .- clamp.(optimalcontrols .- gradient, 0, 1))
    diagnostics = committeddiagnostics(path; nodes=2 * options.nodes - 1, tolerance=options.tolerance, schedulekkt)
    diagnostics.tailinvestment <= options.tailtolerance || error("Terminal investment is $(diagnostics.tailinvestment); extend CommittedOptions.horizon before using the stationary tail.")
    CommittedSolution(path, result, diagnostics)
end
