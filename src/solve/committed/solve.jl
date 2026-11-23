"Numerical controls for the state–costate commitment problem."
Base.@kwdef struct CommittedOptions{T}
    horizon::T = 400.0
    nodes::Int = 81
    tolerance::T = 1e-8
    tailtolerance::T = 1e-5
    initialtax::T = 0.15
end

"Six canonical variables and three constant terminal-constraint multipliers."
struct CommittedState{T} <: StaticArrays.FieldVector{9, T}
    # State
    xᶜ::T
    q::T
    vᵈ::T

    # Costate
    λxᶜ::T
    λq::T
    λvᵈ::T

    # Terminal multipliers
    ψq::T
    ψvᵈ::T
    ψⁱ::T
end

struct CommittedTransition{M, T}
    model::M
    horizon::T
    xᶜ₀::T
    initialtax::T
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
    canonical::T
    hamiltoniankkt::T
    tailhamiltoniankkt::T
    candidatemargin::T
    initialcomplementarity::T
    tailcomplementarity::T
    tailinvestment::T
    taxjump::T
    pricejump::T
    minimumq::T
    minimumvᵈ::T
    minimumxᶜ::T
    singularpoints::Int
    investmentboundarypoints::Int
end

struct CommittedSolution{P, D}
    path::P
    diagnostics::D
end

committedstate(state, problem::CommittedTransition) = CommittedState((state .* problem.scale)...)
committedcomplementarity(a, b) = hypot(a, b) - a - b

function committedpolicy(state::CommittedState, problem::CommittedTransition; tail=false, trial=false)
    @unpack model, scale = problem

    # Newton iterates may leave the economic domain; accepted paths use unprojected states.
    xᶜ = trial ? max(state.xᶜ, eps(scale.xᶜ)) : state.xᶜ
    vᵈ = trial ? max(state.vᵈ, zero(state.vᵈ)) : state.vᵈ

    tail && return committedpolicy(model, xᶜ, vᵈ, min(state.q, model.clean.c), state.ψq, state.ψvᵈ)
    committedpolicy(model, xᶜ, vᵈ, state.q, state.λq, state.λvᵈ)
end

function committedpolicy(path::CommittedPath, s; tail=false)
    state = committedstate(path.solution(s), path.problem)

    committedpolicy(state, path.problem; tail)
end

committedallocation(path::CommittedPath, s) = committedpolicy(path, s).allocation
committedtail(path::CommittedPath) = committedpolicy(path, 1.0; tail=true)

function committeddynamics(state, problem, s)
    @unpack model, horizon, scale = problem

    physical = committedstate(state, problem)
    policy = committedpolicy(physical, problem; trial=true)
    allocation = policy.allocation
    derivatives = policy.derivatives
    r = model.household.r

    ẋᶜ = allocation.investment
    q̇ = r * physical.q - allocation.p
    v̇ᵈ = r * physical.vᵈ - allocation.profit

    # At q=c use the inactive limiting derivative; persistent boundary arcs require separate matching.
    investmentderivative = physical.q > model.clean.c ? inv(model.clean.ξ) : zero(physical.q)

    λxᶜ = r * physical.λxᶜ - derivatives.welfare.xᶜ + physical.λq * derivatives.p.xᶜ + physical.λvᵈ * derivatives.profit.xᶜ
    λq = -(model.clean.c + model.clean.ξ * allocation.investment + physical.λxᶜ) * investmentderivative
    λvᵈ = -derivatives.welfare.vᵈ + physical.λq * derivatives.p.vᵈ + physical.λvᵈ * derivatives.profit.vᵈ

    # The endpoint multipliers are unknown constants, not additional economic states.
    derivative = CommittedState{eltype(physical)}(ẋᶜ, q̇, v̇ᵈ, λxᶜ, λq, λvᵈ, 0, 0, 0)

    CommittedState((horizon .* derivative ./ scale)...)
end

function committedboundary(states, problem, grid)
    @unpack model, scale = problem

    initial = committedstate(states(first(grid)), problem)
    terminal = committedstate(states(last(grid)), problem)

    tail = committedpolicy(terminal, problem; tail=true, trial=true)
    allocation = tail.allocation
    derivatives = tail.derivatives
    r = model.household.r

    terminalλxᶜ = (derivatives.welfare.xᶜ - terminal.ψq * derivatives.p.xᶜ - terminal.ψvᵈ * derivatives.profit.xᶜ) / r
    terminalλvᵈ = terminal.ψvᵈ + (derivatives.welfare.vᵈ - terminal.ψq * derivatives.p.vᵈ - terminal.ψvᵈ * derivatives.profit.vᵈ) / r

    [((initial.xᶜ - problem.xᶜ₀) / scale.xᶜ), initial.λq / scale.λq, committedcomplementarity(initial.λvᵈ / scale.λvᵈ, initial.vᵈ / scale.vᵈ), (terminal.q - allocation.p / r) / scale.q, (terminal.vᵈ - allocation.profit / r) / scale.vᵈ, (terminal.λxᶜ - terminalλxᶜ) / scale.λxᶜ, (terminal.λq - terminal.ψq - terminal.ψⁱ) / scale.λq, (terminal.λvᵈ - terminalλvᵈ) / scale.λvᵈ, committedcomplementarity(terminal.ψⁱ / scale.ψⁱ, (model.clean.c - terminal.q) / scale.q)]
end

function committedguess(problem::CommittedTransition, reference)
    @unpack model, scale = problem

    x̄ᶜ = max(problem.xᶜ₀, reference.xᶜ)
    
    terminal = committedmarketallocation(model, x̄ᶜ, reference.vᵈ, model.clean.c, problem.initialtax)
    initial = committedmarketallocation(model, problem.xᶜ₀, reference.vᵈ, model.clean.c, problem.initialtax)
    derivatives = committedstaticderivatives(model, terminal, problem.initialtax)
    
    r = model.household.r

    q₀ = max(initial.p / r, model.clean.c)
    vᵈ₀ = initial.profit / r

    terminalλxᶜ = derivatives.welfare.xᶜ / r
    terminalλvᵈ = derivatives.welfare.vᵈ / r

    # Frozen-flow values initialise the guess only; initial asset values remain endogenous.
    guess = @closure (parameters, s) -> begin
        xᶜ = (1 - s) * problem.xᶜ₀ + s * x̄ᶜ
        q = (1 - s) * q₀ + s * terminal.p / r
        vᵈ = (1 - s) * vᵈ₀ + s * reference.vᵈ


        [xᶜ, q, vᵈ, terminalλxᶜ, 0, s * terminalλvᵈ, 0, 0, 0] ./ scale
    end
    
    return guess
end

function committedguess(problem::CommittedTransition, previous::CommittedPath)
    guess(parameters, s) = previous.solution(s) .* previous.problem.scale ./ problem.scale

    guess
end

function committedintegral(path, flow; tolerance=1e-8)
    problem = path.problem

    integrand(s) = problem.horizon * exp(-problem.model.household.r * problem.horizon * s) * flow(committedallocation(path, s))

    QuadGK.quadgk(integrand, path.solution.t...; rtol=tolerance, atol=tolerance)
end

function committeddiagnostics(path; nodes=161, tolerance=1e-8)
    @unpack model, horizon, scale = path.problem

    grid = sort!(unique(vcat(range(0, 1; length=nodes), path.solution.t)))
    states = map(s -> committedstate(path.solution(s), path.problem), grid)
    policies = map(state -> committedpolicy(state, path.problem), states)
    allocations = map(policy -> policy.allocation, policies)

    initial = first(states)
    terminal = last(states)
    tail = committedtail(path)
    tailallocation = tail.allocation
    r = model.household.r

    market = maximum(a -> abs(a.x - a.xᶜ - a.xᵈ), allocations)
    credit = maximum(a -> max(a.L - a.L̄, zero(a.L)), allocations)
    complementarity = maximum(a -> abs(a.μ * (a.L̄ - a.L)), allocations)

    investmentkkt = maximum(a -> max(max(a.q - model.clean.c - model.clean.ξ * a.investment, 0), abs(a.investment * (model.clean.c + model.clean.ξ * a.investment - a.q))), allocations)
    dirtykkt = maximum(i -> abs(allocations[i].xᵈ - clamp(allocations[i].xᵈ + allocations[i].p - model.dirty.η * policies[i].tax - model.dirty.γᵈ * allocations[i].xᵈ, 0, model.dirty.kᵈ)), eachindex(allocations))
    interiorprofit = maximum(a -> 0 < a.xᵈ < model.dirty.kᵈ ? abs(a.profit - model.dirty.γᵈ * a.xᵈ^2 / 2) : zero(a.profit), allocations)
    irreversibility = max(-minimum(diff(map(state -> state.xᶜ, states))), 0)

    cleanflow = first(committedintegral(path, a -> a.p; tolerance))
    dirtyflow = first(committedintegral(path, a -> a.profit; tolerance))
    cleanvaluation = abs(initial.q - cleanflow - exp(-r * horizon) * tailallocation.p / r)
    dirtyvaluation = abs(initial.vᵈ - dirtyflow - exp(-r * horizon) * tailallocation.profit / r)

    boundary = maximum(abs, committedboundary(path.solution, path.problem, (0.0, 1.0)))
    midpoints = (path.solution.t[1:end-1] + path.solution.t[2:end]) / 2
    canonical = maximum(s -> maximum(abs, path.solution(s, Val{1}) - committeddynamics(path.solution(s), path.problem, s)) / horizon, midpoints)

    hamiltoniankkt = maximum(policy -> policy.foc, policies)
    tailhamiltoniankkt = tail.foc
    candidatemargin = minimum(policy -> policy.gap, policies)

    initialcomplementarity = abs(committedcomplementarity(initial.λvᵈ / scale.λvᵈ, initial.vᵈ / scale.vᵈ))
    tailcomplementarity = abs(committedcomplementarity(terminal.ψⁱ / scale.ψⁱ, (model.clean.c - terminal.q) / scale.q))
    tailinvestment = cleaninvestment(terminal.q, model.clean)

    values = promote(market, credit, complementarity, investmentkkt, dirtykkt, interiorprofit, irreversibility, cleanvaluation, dirtyvaluation, boundary, canonical, hamiltoniankkt, tailhamiltoniankkt, candidatemargin, initialcomplementarity, tailcomplementarity, tailinvestment, tail.tax - last(policies).tax, tailallocation.p - last(allocations).p, minimum(state -> state.q, states), minimum(state -> state.vᵈ, states), minimum(state -> state.xᶜ, states))
    investmentboundarypoints = count(state -> abs(state.q - model.clean.c) <= 64 * eps(model.clean.c), states)

    CommittedDiagnostics(values..., count(policy -> policy.singular, policies), investmentboundarypoints)
end

"Collocate the canonical system with a pointwise Hamiltonian policy and optimal stationary continuation."
function solvecommitted(model, xᶜ₀; options=CommittedOptions(), guess=nothing)
    @unpack household, firm, clean, dirty = model

    if !(firm isa CESFirm && firm.ρ < 1) || firm.ρ ≈ 0
        error("The committed canonical solver requires CES production with ρ < 1 and ρ ≠ 0.")
    end

    reference = committedstationary(model, options.initialtax)
    
    xᶜscale = max(xᶜ₀, dirty.kᵈ)
    qscale = clean.c
    vᵈscale = dirty.γᵈ * dirty.kᵈ^2 / 2household.r
    costscale = reference.y / household.r

    scale = CommittedState(xᶜscale, qscale, vᵈscale, costscale / xᶜscale, costscale / qscale, costscale / vᵈscale, costscale / qscale, costscale / vᵈscale, costscale / qscale)
    
    problem = CommittedTransition(model, options.horizon, xᶜ₀, options.initialtax, scale)

    initialguess = if isnothing(guess)
        committedguess(problem, reference)
    elseif guess isa CommittedPath
        committedguess(problem, guess)
    else
        guess
    end

    bvp = SciMLBase.BVProblem(committeddynamics, committedboundary, initialguess, (0.0, 1.0), problem)

    # Analytical static derivatives enter the costates; finite differences form the collocation Jacobian.
    jacobian = BoundaryValueDiffEq.BVPJacobianAlgorithm(ADTypes.AutoFiniteDiff())

    solution = BoundaryValueDiffEq.solve(bvp, BoundaryValueDiffEq.MIRK4(; jac_alg=jacobian); dt=1 / (options.nodes - 1), abstol=options.tolerance, reltol=options.tolerance)
    SciMLBase.successful_retcode(solution) || error("Committed state–costate BVP failed: $(solution.retcode).")

    sampled = map(s -> committedstate(solution(s), problem), range(0, 1; length=2 * options.nodes - 1))
    minimum(state -> state.xᶜ, sampled) > zero(xᶜ₀) || error("Committed canonical solution returned non-positive clean capacity.")
    minimum(state -> min(state.q / scale.q, state.vᵈ / scale.vᵈ), sampled) >= -10 * options.tolerance || error("Committed canonical solution returned negative private valuations.")

    path = CommittedPath(problem, solution, 0.0, 0.0)
    tail = committedtail(path).allocation
    flowcost, quadratureerror = committedintegral(path, a -> a.welfare; tolerance=options.tolerance)

    tailcost = exp(-household.r * options.horizon) * tail.welfare / household.r
    path = CommittedPath(problem, solution, flowcost + tailcost, quadratureerror)

    diagnostics = committeddiagnostics(path; nodes=2 * options.nodes - 1, tolerance=options.tolerance)
    diagnostics.tailinvestment <= options.tailtolerance || error("Terminal investment is $(diagnostics.tailinvestment); the stationary continuation is infeasible.")
    diagnostics.boundary <= 100 * options.tolerance || error("Committed endpoint residual is $(diagnostics.boundary); inspect the canonical boundary conditions.")

    CommittedSolution(path, diagnostics)
end
