"Numerical controls for the state–costate commitment problem."
Base.@kwdef struct CommittedOptions{T}
    horizon::T = 400.0
    nodes::Int = 81
    tolerance::T = 1e-8
    tailtolerance::T = 1e-5
    initialtax::T = 0.15
    verbose::Bool = true
    solverlogging::Bool = false
    loginterval::T = 5.0
end

"Thread-safe counters for throttled collocation progress; never used to select the policy."
struct CommittedProgress
    dynamics::Base.Threads.Atomic{Int}
    boundaries::Base.Threads.Atomic{Int}
    active::Base.Threads.Atomic{Bool}
    started::UInt64
    lastlog::Base.Threads.Atomic{UInt64}
    interval::UInt64
end

function CommittedProgress(started, interval)
    CommittedProgress(Base.Threads.Atomic{Int}(0), Base.Threads.Atomic{Int}(0), Base.Threads.Atomic{Bool}(true), started, Base.Threads.Atomic{UInt64}(started), UInt64(round(1e9 * interval)))
end

committedprogress!(::Nothing, phase) = nothing
function committedprogress!(progress::CommittedProgress, phase)
    progress.active[] || return nothing
    counter = phase === :dynamics ? progress.dynamics : progress.boundaries
    evaluations = Base.Threads.atomic_add!(counter, 1) + 1
    evaluations % 64 == 0 || return nothing

    now = time_ns()
    previous = progress.lastlog[]
    if now - previous >= progress.interval && Base.Threads.atomic_cas!(progress.lastlog, previous, now) == previous
        @info "Committed BVP evaluations" elapsed=(now - progress.started) / 1e9 dynamics=progress.dynamics[] boundaries=progress.boundaries[]
    end

    return nothing
end

"Three private states and their current-value costates."
struct CommittedState{T} <: StaticArrays.FieldVector{6, T}
    # State
    xᶜ::T
    q::T
    vᵈ::T

    # Costate
    λxᶜ::T
    λq::T
    λvᵈ::T
end

StaticArrays.similar_type(::Type{<:CommittedState}, ::Type{T}, ::StaticArrays.Size{(6,)}) where {T} = CommittedState{T}

struct CommittedTransition{M, T, G, P}
    model::M
    horizon::T
    xᶜ₀::T
    initialtax::T
    scale::CommittedState{T}
    policyguess::G
    progress::P
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
    boundarygap::T
    tailboundarygap::T
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

function committedstate(state::AbstractVector, problem::CommittedTransition)
    scale = problem.scale
    T = promote_type(eltype(state), eltype(scale))

    CommittedState{T}(state[1] * scale.xᶜ, state[2] * scale.q, state[3] * scale.vᵈ, state[4] * scale.λxᶜ, state[5] * scale.λq, state[6] * scale.λvᵈ)
end

committedcomplementarity(a, b) = hypot(a, b) - a - b

function committedpolicy(state::CommittedState, problem::CommittedTransition, s)
    committedpolicy(problem.model, max(state.xᶜ, eps(problem.scale.xᶜ)), max(state.vᵈ, zero(state.vᵈ)), state.q, state.λq, state.λvᵈ; zguess=problem.policyguess(s))
end

function committedpolicy(path::CommittedPath, s)
    state = committedstate(path.solution(s), path.problem)

    committedpolicy(state, path.problem, s)
end

committedallocation(path::CommittedPath, s) = committedpolicy(path, s).allocation

function committedtail(path::CommittedPath)
    problem = path.problem
    state = committedstate(last(path.solution.u), problem)
    allocation = committedtail(problem.model, state.xᶜ, state.vᵈ, state.q)
    geometry = committedgeometry(problem.model, allocation.z, allocation.vᵈ)
    ψq, ψvᵈ, _ = committedtailmultipliers(problem.model, allocation, state.λxᶜ, state.λq, geometry)
    τ = (allocation.p - problem.model.dirty.γᵈ * allocation.xᵈ) / problem.model.dirty.η
    derivatives, foc, creditkink, dirtykink, singular = committedpolicyderivatives(problem.model, allocation, τ, ψq, ψvᵈ, geometry)
    hamiltonian = committedflow(problem.model, allocation) - ψq * allocation.p - ψvᵈ * allocation.profit
    T = promote_type(typeof(τ), typeof(hamiltonian), typeof(foc))

    CommittedPolicy{typeof(allocation), typeof(derivatives), T}(allocation, τ, derivatives, hamiltonian, foc, creditkink, dirtykink, singular)
end

"Freeze the previous policy branch as a deterministic, interpolated root reference."
function committedpolicyguess(path::CommittedPath)
    grid = path.solution.t
    ratios = map(s -> committedpolicy(path, s).allocation.z, grid)

    guess = @closure s -> begin
        i = clamp(searchsortedlast(grid, s), firstindex(grid), lastindex(grid) - 1)
        weight = clamp((s - grid[i]) / (grid[i + 1] - grid[i]), zero(s), one(s))
        (1 - weight) * ratios[i] + weight * ratios[i + 1]
    end

    guess
end

function committeddynamics!(du, u, problem::CommittedTransition, s)
    @unpack model, horizon, scale = problem
    committedprogress!(problem.progress, :dynamics)

    xᶜ = u[1] * scale.xᶜ
    q = u[2] * scale.q
    vᵈ = u[3] * scale.vᵈ
    λxᶜ = u[4] * scale.λxᶜ
    λq = u[5] * scale.λq
    λvᵈ = u[6] * scale.λvᵈ

    # Project only the private allocation at trial iterates, not the canonical variables.
    policy = committedpolicy(model, max(xᶜ, eps(scale.xᶜ)), max(vᵈ, zero(vᵈ)), q, λq, λvᵈ; zguess=problem.policyguess(s))
    allocation = policy.allocation
    derivatives = policy.derivatives
    r = model.household.r

    ẋᶜ = allocation.investment
    q̇ = r * q - allocation.p
    v̇ᵈ = r * vᵈ - allocation.profit

    # At q=c use the inactive limiting derivative; persistent boundary arcs require separate matching.
    investmentderivative = ifelse(q > model.clean.c, inv(model.clean.ξ), zero(q))

    λ̇xᶜ = r * λxᶜ - derivatives.welfare.xᶜ + λq * derivatives.p.xᶜ + λvᵈ * derivatives.profit.xᶜ
    λ̇q = -(model.clean.c + model.clean.ξ * allocation.investment + λxᶜ) * investmentderivative
    λ̇vᵈ = -derivatives.welfare.vᵈ + λq * derivatives.p.vᵈ + λvᵈ * derivatives.profit.vᵈ

    du[1] = horizon * ẋᶜ / scale.xᶜ
    du[2] = horizon * q̇ / scale.q
    du[3] = horizon * v̇ᵈ / scale.vᵈ
    du[4] = horizon * λ̇xᶜ / scale.λxᶜ
    du[5] = horizon * λ̇q / scale.λq
    du[6] = horizon * λ̇vᵈ / scale.λvᵈ

    return du
end

function committedboundary!(residual, states, problem::CommittedTransition, grid)
    # The boundary conditions use endpoint mesh states without interpolating the path.
    committedboundary!(residual, first(states), last(states), problem)
end

function committedboundary!(residual, initial, terminal, problem::CommittedTransition)
    committedinitialboundary!(view(residual, 1:3), initial, problem)
    committedterminalboundary!(view(residual, 4:6), terminal, problem)

    return nothing
end

function committedinitialboundary!(residual, initial, problem::CommittedTransition)
    residual[1] = (initial[1] * problem.scale.xᶜ - problem.xᶜ₀) / problem.scale.xᶜ
    residual[2] = initial[5]
    residual[3] = committedcomplementarity(initial[6], initial[3])

    return nothing
end

function committedterminalboundary!(residual, terminal, problem::CommittedTransition)
    @unpack model, scale = problem
    committedprogress!(problem.progress, :boundary)

    xᶜ = terminal[1] * scale.xᶜ
    q = terminal[2] * scale.q
    vᵈ = terminal[3] * scale.vᵈ
    λxᶜ = terminal[4] * scale.λxᶜ
    λq = terminal[5] * scale.λq
    λvᵈ = terminal[6] * scale.λvᵈ
    allocation = committedtail(model, xᶜ, vᵈ, q)
    ∂q, ∂g = committedtailderivatives(model, allocation)

    # Cross-multiplication avoids dividing by ∂q.vᵈ at a fold of stationary tails.
    residual[1] = (q - allocation.p / model.household.r) / scale.q
    residual[2] = (∂q.xᶜ * scale.xᶜ / scale.q) * (λvᵈ - ∂g.vᵈ) / scale.λvᵈ - (∂q.vᵈ * scale.vᵈ / scale.q) * (λxᶜ - ∂g.xᶜ) / scale.λxᶜ
    residual[3] = committedcomplementarity((∂g.xᶜ - λxᶜ - ∂q.xᶜ * λq) / scale.λxᶜ, (model.clean.c - q) / scale.q)

    return nothing
end

function committedguess(problem::CommittedTransition{M, T}, reference) where {M, T}
    @unpack model, scale = problem

    x̄ᶜ = max(problem.xᶜ₀, reference.xᶜ)
    
    terminal = committedmarketallocation(model, x̄ᶜ, reference.vᵈ, model.clean.c, problem.initialtax)
    initial = committedmarketallocation(model, problem.xᶜ₀, reference.vᵈ, model.clean.c, problem.initialtax)
    tail = committedtail(model, x̄ᶜ, reference.vᵈ, model.clean.c)
    _, ∂g = committedtailderivatives(model, tail)
    
    r = model.household.r

    q₀ = max(initial.p / r, model.clean.c)
    vᵈ₀ = initial.profit / r

    terminalλxᶜ = ∂g.xᶜ
    terminalλvᵈ = ∂g.vᵈ
    # Avoid starting exactly at the non-differentiable investment threshold.
    terminalq = terminal.p / r + model.clean.c * cbrt(eps(one(T)))

    # Frozen-flow values initialise the guess only; initial asset values remain endogenous.
    guess = @closure (parameters, s) -> begin
        xᶜ = (1 - s) * problem.xᶜ₀ + s * x̄ᶜ
        q = (1 - s) * q₀ + s * terminalq
        vᵈ = (1 - s) * vᵈ₀ + s * reference.vᵈ


        T[xᶜ / scale.xᶜ, q / scale.q, vᵈ / scale.vᵈ, terminalλxᶜ / scale.λxᶜ, zero(T), s * terminalλvᵈ / scale.λvᵈ]
    end
    
    return guess
end

function committedguess(problem::CommittedTransition{M, T}, previous::CommittedPath) where {M, T}
    guess = @closure (parameters, s) -> begin
        state = previous.solution(s)
        scaled = Vector{T}(undef, length(problem.scale))

        @inbounds for i in eachindex(scaled, state, problem.scale)
            scaled[i] = state[i] * previous.problem.scale[i] / problem.scale[i]
        end

        scaled
    end

    guess
end

function committedintegral(path, flow; tolerance=1e-8)
    problem = path.problem

    integrand(s) = problem.horizon * exp(-problem.model.household.r * problem.horizon * s) * flow(committedallocation(path, s))

    QuadGK.quadgk(integrand, path.solution.t; rtol=tolerance, atol=tolerance)
end

function committeddiagnostics(path; nodes=161, tolerance=1e-8, verbose=false)
    @unpack model, horizon, scale = path.problem

    started = time_ns()
    if verbose
        @info "Sampling committed policies for diagnostics" nodes=nodes
    end
    grid = sort!(unique(vcat(range(zero(horizon), one(horizon); length=nodes), path.solution.t)))
    states = map(s -> committedstate(path.solution(s), path.problem), grid)
    policies = map((state, s) -> committedpolicy(state, path.problem, s), states, grid)
    allocations = map(policy -> policy.allocation, policies)

    initial = first(states)
    terminal = last(states)
    tail = committedtail(path)
    tailallocation = tail.allocation
    ψq, ψvᵈ, ψⁱ = committedtailmultipliers(model, tailallocation, terminal.λxᶜ, terminal.λq)
    r = model.household.r

    market = maximum(a -> abs(a.x - a.xᶜ - a.xᵈ), allocations)
    credit = maximum(a -> max(a.L - a.L̄, zero(a.L)), allocations)
    complementarity = maximum(a -> abs(a.μ * (a.L̄ - a.L)), allocations)

    investmentkkt = maximum(a -> max(max(a.q - model.clean.c - model.clean.ξ * a.investment, 0), abs(a.investment * (model.clean.c + model.clean.ξ * a.investment - a.q))), allocations)
    dirtykkt = maximum(i -> abs(allocations[i].xᵈ - clamp(allocations[i].xᵈ + allocations[i].p - model.dirty.η * policies[i].tax - model.dirty.γᵈ * allocations[i].xᵈ, 0, model.dirty.kᵈ)), eachindex(allocations))
    interiorprofit = maximum(a -> 0 < a.xᵈ < model.dirty.kᵈ ? abs(a.profit - model.dirty.γᵈ * a.xᵈ^2 / 2) : zero(a.profit), allocations)
    irreversibility = max(-minimum(diff(map(state -> state.xᶜ, states))), 0)

    if verbose
        @info "Checking private present values" elapsed=(time_ns() - started) / 1e9 samples=length(grid)
    end
    cleanflow = first(committedintegral(path, a -> a.p; tolerance))
    dirtyflow = first(committedintegral(path, a -> a.profit; tolerance))
    cleanvaluation = abs(initial.q - cleanflow - exp(-r * horizon) * tailallocation.p / r)
    dirtyvaluation = abs(initial.vᵈ - dirtyflow - exp(-r * horizon) * tailallocation.profit / r)

    if verbose
        @info "Checking canonical and boundary residuals" elapsed=(time_ns() - started) / 1e9
    end
    residual = zeros(eltype(scale), length(scale))
    committedboundary!(residual, first(path.solution.u), last(path.solution.u), path.problem)
    boundary = maximum(abs, residual)
    midpoints = (path.solution.t[1:end-1] + path.solution.t[2:end]) / 2
    canonical = zero(horizon)

    for s in midpoints
        state = path.solution(s)
        derivative = path.solution(s, Val{1})
        committeddynamics!(residual, state, path.problem, s)

        @inbounds for i in eachindex(residual, derivative)
            canonical = max(canonical, abs(derivative[i] - residual[i]) / horizon)
        end
    end

    hamiltoniankkt = maximum(policy -> policy.foc, policies)
    tailhamiltoniankkt = tail.foc
    if verbose
        @info "Comparing policy boundary Hamiltonians" elapsed=(time_ns() - started) / 1e9
    end
    boundarygap = maximum(i -> committedboundarygap(model, policies[i], states[i].λq, states[i].λvᵈ), eachindex(policies))
    tailboundarygap = committedboundarygap(model, tail, ψq, ψvᵈ)

    initialcomplementarity = abs(committedcomplementarity(initial.λvᵈ / scale.λvᵈ, initial.vᵈ / scale.vᵈ))
    tailcomplementarity = abs(committedcomplementarity(ψⁱ / scale.λq, (model.clean.c - terminal.q) / scale.q))
    tailinvestment = cleaninvestment(terminal.q, model.clean)

    investmentboundarypoints = count(state -> abs(state.q - model.clean.c) <= 64 * eps(model.clean.c), states)

    CommittedDiagnostics{eltype(scale)}(market, credit, complementarity, investmentkkt, dirtykkt, interiorprofit, irreversibility, cleanvaluation, dirtyvaluation, boundary, canonical, hamiltoniankkt, tailhamiltoniankkt, boundarygap, tailboundarygap, initialcomplementarity, tailcomplementarity, tailinvestment, tail.tax - last(policies).tax, tailallocation.p - last(allocations).p, minimum(state -> state.q, states), minimum(state -> state.vᵈ, states), minimum(state -> state.xᶜ, states), count(policy -> policy.singular, policies), investmentboundarypoints)
end

"Collocate the canonical system on one stationary policy branch and check optimality ex post."
function solvecommitted(model, xᶜ₀::T; options=CommittedOptions(), guess=nothing) where T
    @unpack household, firm, clean, dirty = model
    started = time_ns()

    xᶜ₀ >= zero(T) || error("The committed solver requires non-negative initial clean capacity.")

    if !(firm isa CESFirm && firm.ρ < 1) || firm.ρ ≈ 0
        error("The committed canonical solver requires CES production with ρ < 1 and ρ ≠ 0.")
    end

    if options.verbose
        @info "Preparing committed BVP" horizon=options.horizon nodes=options.nodes tolerance=options.tolerance
    end
    reference = committedstationary(model, options.initialtax)
    
    xᶜscale = max(xᶜ₀, dirty.kᵈ)
    qscale = clean.c
    vᵈscale = dirty.γᵈ * dirty.kᵈ^2 / 2household.r
    costscale = reference.y / household.r

    scale = CommittedState(xᶜscale, qscale, vᵈscale, costscale / xᶜscale, costscale / qscale, costscale / vᵈscale)
    
    policyguess = guess isa CommittedPath ? committedpolicyguess(guess) : (@closure s -> reference.z)
    progress = options.verbose && options.loginterval > 0 ? CommittedProgress(time_ns(), options.loginterval) : nothing
    problem = CommittedTransition(model, options.horizon, xᶜ₀, options.initialtax, scale, policyguess, progress)

    initialguess = if isnothing(guess)
        committedguess(problem, reference)
    elseif guess isa CommittedPath
        committedguess(problem, guess)
    else
        guess
    end

    bvpfunction = SciMLBase.BVPFunction{true}(committeddynamics!, (committedinitialboundary!, committedterminalboundary!); bcresid_prototype=(zeros(T, 3), zeros(T, 3)), twopoint=Val(true))
    bvp = SciMLBase.TwoPointBVProblem(bvpfunction, initialguess, (zero(T), one(T)), problem)

    # Colour the known collocation sparsity rather than perturbing every mesh unknown separately.
    jacobian = BVD.BVPJacobianAlgorithm(ADTypes.AutoSparse(ADTypes.AutoFiniteDiff()))
    # Damp nonlinear steps rather than starting with unrestricted Gauss–Newton updates.
    nonlinear = NonlinearSolve.LevenbergMarquardt(; disable_geodesic=Val(true))
    verbosity = if options.solverlogging
        BVD.BVPVerbosity(SciMLLogging.All())
    elseif options.verbose
        BVD.BVPVerbosity(SciMLLogging.Minimal())
    else
        BVD.BVPVerbosity(SciMLLogging.None())
    end

    collocationstarted = time_ns()
    if options.verbose
        @info "Solving committed BVP with sparse finite differences" preparation=(collocationstarted - started) / 1e9
    end
    solution = try
        BVD.solve(bvp, BVD.MIRK4(; jac_alg=jacobian, nlsolve=nonlinear); dt=inv(options.nodes - 1), abstol=options.tolerance, reltol=options.tolerance, verbose=verbosity)
    finally
        isnothing(progress) || (progress.active[] = false)
    end
    if options.verbose
        @info "Committed collocation finished" elapsed=(time_ns() - collocationstarted) / 1e9 retcode=solution.retcode meshpoints=length(solution.t)
        if !isnothing(progress)
            @info "Committed collocation work" dynamics=progress.dynamics[] boundaries=progress.boundaries[]
        end
    end
    if !SciMLBase.successful_retcode(solution)
        error("Committed state–costate BVP failed: $(solution.retcode).")
    end

    sampled = map(s -> committedstate(solution(s), problem), range(zero(T), one(T); length=2 * options.nodes - 1))
    minimum(state -> state.xᶜ, sampled) >= -10 * options.tolerance * scale.xᶜ || error("Committed canonical solution returned negative clean capacity.")
    minimum(state -> min(state.q / scale.q, state.vᵈ / scale.vᵈ), sampled) >= -10 * options.tolerance || error("Committed canonical solution returned negative private valuations.")

    path = CommittedPath(problem, solution, zero(T), zero(T))
    terminal = committedstate(last(solution.u), problem)
    terminal.vᵈ > zero(T) || error("The six-dimensional terminal reduction requires positive dirty value; a shutdown tail needs separate corner conditions.")
    tail = committedtail(path).allocation
    tailoutput = sqrt(2household.r * terminal.vᵈ / dirty.γᵈ)
    tailtax = (tail.p - dirty.γᵈ * tailoutput) / dirty.η
    zero(T) < tailoutput < dirty.kᵈ && tailtax > zero(T) || error("The six-dimensional terminal reduction requires interior dirty output and a positive tail tax; recovered output=$(tailoutput), tax=$(tailtax).")
    quadraturestarted = time_ns()
    if options.verbose
        @info "Integrating committed welfare"
    end
    flowcost, quadratureerror = committedintegral(path, a -> a.welfare; tolerance=options.tolerance)

    tailcost = exp(-household.r * options.horizon) * tail.welfare / household.r
    path = CommittedPath(problem, solution, flowcost + tailcost, quadratureerror)

    if options.verbose
        @info "Committed welfare integral finished" elapsed=(time_ns() - quadraturestarted) / 1e9 cost=path.cost quadratureerror=quadratureerror
    end
    diagnosticsstarted = time_ns()
    diagnostics = committeddiagnostics(path; nodes=2options.nodes - 1, tolerance=options.tolerance, verbose=options.verbose)
    if options.verbose
        @info "Committed diagnostics finished" elapsed=(time_ns() - diagnosticsstarted) / 1e9 boundary=diagnostics.boundary canonical=diagnostics.canonical boundarygap=diagnostics.boundarygap tailboundarygap=diagnostics.tailboundarygap
    end

    if !(diagnostics.tailinvestment ≤ options.tailtolerance)
        error("Terminal investment is $(diagnostics.tailinvestment); the stationary continuation is infeasible.")
    end

    if !(diagnostics.boundary ≤ 100options.tolerance)
        error("Committed endpoint residual is $(diagnostics.boundary); inspect the canonical boundary conditions.")
    end

    if max(diagnostics.boundarygap, diagnostics.tailboundarygap) > options.tolerance
        @warn "A policy boundary or nearby ratio improves the Hamiltonian; the selected branch is not verified optimal." boundarygap=diagnostics.boundarygap tailboundarygap=diagnostics.tailboundarygap
    end

    if options.verbose
        @info "Committed solve finished" elapsed=(time_ns() - started) / 1e9
    end

    CommittedSolution(path, diagnostics)
end
