## Normalised state space
struct NonCommittedMarkovGrid{T}
    h::T
    φ::AffineGridMap{T}
    A::AffineGridMap{T}
end

function NonCommittedMarkovGrid(model::AbstractModel, h; alower = model.firm.a₀)
    isfinite(h) && h > 0 || throw(ArgumentError("The grid step must be finite and positive."))
    N = round(Int, inv(h))
    N ≥ 1 && isapprox(N * h, one(h)) || throw(ArgumentError("The grid step must divide [0, 1]."))
    isfinite(alower) && isfinite(model.firm.η) && 0 ≤ alower < model.firm.η || throw(ArgumentError("The abatement grid must lie in [0,η] and have finite positive width."))
    T = promote_type(typeof(float(h)), typeof(alower), typeof(model.firm.η))
    return NonCommittedMarkovGrid(one(T) / N, AffineGridMap(zero(T), one(T)), AffineGridMap(T(alower), T(model.firm.η)))
end

Base.size(grid::NonCommittedMarkovGrid) = (round(Int, inv(grid.h)) + 1, round(Int, inv(grid.h)) + 1)
Base.size(grid::NonCommittedMarkovGrid, d::Integer) = size(grid)[d]
Base.CartesianIndices(grid::NonCommittedMarkovGrid) = CartesianIndices(size(grid))
unitcube(grid::NonCommittedMarkovGrid) = (range(0, 1; length = size(grid, 1)), range(0, 1; length = size(grid, 2)))
physicalstate(grid::NonCommittedMarkovGrid, z) = (grid.φ(z[1]), grid.A(z[2]))
physicalstate(grid::NonCommittedMarkovGrid, i::Integer, j::Integer) = (i == size(grid, 1) ? grid.φ.upper : grid.φ((i - 1) * grid.h), j == size(grid, 2) ? grid.A.upper : grid.A((j - 1) * grid.h))
normalisedstate(grid::NonCommittedMarkovGrid, x) = (normalise(x[1], grid.φ), normalise(x[2], grid.A))

struct NonCommittedMarkovParameters{M, G, S, C, D, T}
    model::M
    government::G
    signal::S
    τᶜ::C
    grid::D
    Δtmax::T
    tolerance::T
    policytolerance::T
    relaxation::T
    maxiters::Int
end

function NonCommittedMarkovParameters(model, government, signal, τᶜ, grid, Δtmax; tolerance = 1e-8, policytolerance = 1e-7, relaxation = 0.5, maxiters = 1000)
    holding, valuetolerance, taxtolerance, damping = promote(float(Δtmax), tolerance, policytolerance, relaxation)
    return NonCommittedMarkovParameters(model, government, signal, τᶜ, grid, holding, valuetolerance, taxtolerance, damping, maxiters)
end

struct NonCommittedMarkovValue{T, G, M <: AbstractMatrix}
    t::T
    grid::G
    q::M
    uᵍ::M
    u::M
    τ::M
end

Base.copy(V::NonCommittedMarkovValue) = NonCommittedMarkovValue(V.t, V.grid, copy(V.q), copy(V.uᵍ), copy(V.u), copy(V.τ))

struct NonCommittedMarkovDiagnostics{T}
    iterations::Int
    valuegap::T
    bellmangap::T
    policygap::T
    policyimprovement::T
end

struct NonCommittedMarkovSolution{T, V, P, D}
    t::Vector{T}
    u::Vector{V}
    parameters::P
    diagnostics::Vector{D}
end

function (solution::NonCommittedMarkovSolution)(t)
    first(solution.t) ≤ t ≤ last(solution.t) || throw(DomainError(t, "Calendar time is outside the solution."))
    k = min(searchsortedlast(solution.t, t), length(solution.t) - 1)
    weight = (t - solution.t[k]) / (solution.t[k + 1] - solution.t[k])
    left, right = solution.u[k], solution.u[k + 1]
    q = (1 - weight) .* left.q .+ weight .* right.q
    uᵍ = (1 - weight) .* left.uᵍ .+ weight .* right.uᵍ
    u = (1 - weight) .* left.u .+ weight .* right.u
    path = solution.parameters.τᶜ
    # The last pre-jump slab uses its frozen left policy, without mixing in the new regime.
    τ = t < right.t && path(right.t; side = :left) != path(right.t) ? copy(left.τ) : (1 - weight) .* left.τ .+ weight .* right.τ
    τ = clamp.(τ, zero(t), path(t))
    return NonCommittedMarkovValue(t, left.grid, q, uᵍ, u, τ)
end

## Kushner--Dupuis chain
struct NonCommittedTransition{T}
    Δt::T
    pᵩ⁺::T
    pᵩ⁻::T
    pₐ⁺::T
    p₀::T
end

noncommittedexpectedtax(φ, τ, τᶜ) = φ * τᶜ + (1 - φ) * τ

function noncommittedbeliefmoments(τ, τᶜ, φ, signal::Signal)
    χ = (signal.ϵ / signal.σ) * (τᶜ - τ)
    return -φ^2 * (1 - φ) * χ^2, φ * (1 - φ) * χ
end

function noncommittedtransition(φ, a, q, τ, τᶜ, parameters::NonCommittedMarkovParameters, Δt; firmmeasure = false)
    grid, firm, signal = parameters.grid, parameters.model.firm, parameters.signal
    Δφ, Δa = grid.h * span(grid.φ), grid.h * span(grid.A)
    μᵩ, σᵩ = noncommittedbeliefmoments(τ, τᶜ, φ, signal)
    B = φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * (τᶜ - τ)
    τᵉ = noncommittedexpectedtax(φ, τ, τᶜ)
    λₐ = a ≥ firm.η ? zero(q) : α(q, firm) / Δa
    variance = σᵩ^2 / (2Δφ^2)
    # Freeze a holding time valid for every hidden deviation in [0, τᶜ].
    bound = λₐ + 2variance + B * max(τᵉ, τᶜ - τᵉ) / Δφ
    holding = min(parameters.Δtmax, Δt)
    bound > 0 && (holding = min(holding, inv(bound)))
    μ = firmmeasure ? zero(μᵩ) : μᵩ
    pᵩ⁺ = holding * (variance + max(μ, zero(μ)) / Δφ)
    pᵩ⁻ = holding * (variance + max(-μ, zero(μ)) / Δφ)
    pₐ⁺ = holding * λₐ
    p₀ = max(one(holding) - pᵩ⁺ - pᵩ⁻ - pₐ⁺, zero(holding))
    return NonCommittedTransition(promote(holding, pᵩ⁺, pᵩ⁻, pₐ⁺, p₀)...)
end

function noncommittedcontinuation(x, i, j, transition::NonCommittedTransition)
    continuation = transition.p₀ * x[i, j]
    transition.pᵩ⁺ > 0 && (continuation += transition.pᵩ⁺ * x[i + 1, j])
    transition.pᵩ⁻ > 0 && (continuation += transition.pᵩ⁻ * x[i - 1, j])
    transition.pₐ⁺ > 0 && (continuation += transition.pₐ⁺ * x[i, j + 1])
    return continuation
end

function noncommittedweights(transition, r, Δt)
    holding = transition.Δt
    discount = exp(-r * holding)
    # Integrate a frozen flow exactly, preserving constant-flow NPV = flow/r.
    flowweight = -expm1(-r * holding) / r
    timeweight = isfinite(Δt) ? holding / Δt : zero(holding)
    return discount, flowweight, timeweight
end

## One joint solve across beliefs at each abatement layer
struct NonCommittedColumn{V, W, P, T}
    value::V
    future::W
    parameters::P
    j::Int
    τᶜ::T
    Δt::T
    qscale::T
    valuescale::T
    taxscale::T
    κ₀::T
    κ₁::T
    κ₂::T
end

function noncommittedcolumncontinuation(x, component, i, column, transition, timeweight)
    V, V′, j = column.value, column.future, column.j
    scale = component == 1 ? column.qscale : column.valuescale
    currentfield = component == 1 ? V.q : V.uᵍ
    current = transition.p₀ * scale * x[3i - 3 + component]
    transition.pᵩ⁺ > 0 && (current += transition.pᵩ⁺ * scale * x[3i + component])
    transition.pᵩ⁻ > 0 && (current += transition.pᵩ⁻ * scale * x[3i - 6 + component])
    transition.pₐ⁺ > 0 && (current += transition.pₐ⁺ * currentfield[i, j + 1])
    isnothing(V′) && return current
    futurefield = component == 1 ? V′.q : V′.uᵍ
    return (1 - timeweight) * current + timeweight * noncommittedcontinuation(futurefield, i, j, transition)
end

function noncommittedimprovement(x, i, column, transition)
    grid, signal = column.parameters.grid, column.parameters.signal
    τᶜ, j = column.τᶜ, column.j
    τ = clamp(column.taxscale * x[3i], zero(τᶜ), τᶜ)
    (iszero(τᶜ) || iszero(column.κ₂)) && return zero(τ), zero(τ)
    φ = physicalstate(grid, i, j)[1]
    τᵉ = noncommittedexpectedtax(φ, τ, τᶜ)
    B = φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * (τᶜ - τ)
    discount, flowweight, timeweight = noncommittedweights(transition, column.parameters.model.household.r, column.Δt)
    center = (1 - timeweight) * column.valuescale * x[3i - 1]
    !isnothing(column.future) && (center += timeweight * column.future.uᵍ[i, j])
    backward, forward = zero(center), zero(center)
    if !iszero(B)
        below = (1 - timeweight) * column.valuescale * x[3i - 4]
        above = (1 - timeweight) * column.valuescale * x[3i + 2]
        if !isnothing(column.future)
            below += timeweight * column.future.uᵍ[i - 1, j]
            above += timeweight * column.future.uᵍ[i + 1, j]
        end
        backward = (center - below) / (grid.h * span(grid.φ))
        forward = (above - center) / (grid.h * span(grid.φ))
    end
    coefficient = discount * transition.Δt * B / flowweight
    # A deviation changes the drift sign at the frozen expected tax.
    belowtax = clamp(-(column.κ₁ + coefficient * backward) / column.κ₂, zero(τ), τᵉ)
    abovetax = clamp(-(column.κ₁ + coefficient * forward) / column.κ₂, τᵉ, τᶜ)
    objective(τ̂) = column.κ₁ * τ̂ + column.κ₂ * τ̂^2 / 2 + coefficient * (τ̂ - τᵉ) * (τ̂ ≤ τᵉ ? backward : forward)
    belowvalue, abovevalue = objective(belowtax), objective(abovetax)
    candidate = belowvalue ≤ abovevalue ? belowtax : abovetax
    improvement = max(objective(τ) - min(belowvalue, abovevalue), zero(τ))
    return candidate, improvement
end

function noncommittedcolumnresidual!(residual, x, column)
    parameters, j = column.parameters, column.j
    model, grid = parameters.model, parameters.grid
    household, firm = model.household, model.firm
    a = physicalstate(grid, 1, j)[2]
    @inbounds for i in 1:size(grid, 1)
        φ = physicalstate(grid, i, j)[1]
        q, uᵍ = column.qscale * x[3i - 2], column.valuescale * x[3i - 1]
        rawtax = column.taxscale * x[3i]
        τ = clamp(rawtax, zero(rawtax), column.τᶜ)
        τᵉ = noncommittedexpectedtax(φ, τ, column.τᶜ)
        investment = a ≥ firm.η ? zero(q) : α(q, firm)
        firmtransition = noncommittedtransition(φ, a, q, τ, column.τᶜ, parameters, column.Δt; firmmeasure = true)
        transition = noncommittedtransition(φ, a, q, τ, column.τᶜ, parameters, column.Δt)
        discount, flowweight, timeweight = noncommittedweights(transition, household.r, column.Δt)
        futureq = noncommittedcolumncontinuation(x, 1, i, column, firmtransition, timeweight)
        targetq = discount * futureq + flowweight * y(n(τᵉ, a, household, firm), firm) * τᵉ
        # One-sided marginal value at the irreversible capacity constraint.
        a ≥ firm.η && (targetq = min(c(firm), targetq))
        residual[3i - 2] = (q - targetq) / (flowweight * household.r * column.qscale)
        if a ≥ firm.η
            residual[3i - 1] = (uᵍ - column.κ₀ / household.r) / column.valuescale
        else
            futurevalue = noncommittedcolumncontinuation(x, 2, i, column, transition, timeweight)
            flow = k(investment, firm) + column.κ₀ + column.κ₁ * τ + column.κ₂ * τ^2 / 2
            residual[3i - 1] = (uᵍ - discount * futurevalue - flowweight * flow) / (flowweight * household.r * column.valuescale)
        end
        candidate, _ = noncommittedimprovement(x, i, column, transition)
        residual[3i] = (rawtax - candidate) / column.taxscale
    end
    return residual
end

function noncommittedsocialcolumn!(lower, diagonal, upper, rhs, V, V′, j, τᶜ, parameters, Δt)
    model, grid = parameters.model, parameters.grid
    household, firm = model.household, model.firm
    a = physicalstate(grid, 1, j)[2]
    fill!(lower, 0)
    fill!(upper, 0)
    @inbounds for i in axes(V.u, 1)
        φ = physicalstate(grid, i, j)[1]
        q, τ = V.q[i, j], V.τ[i, j]
        τᵉ = noncommittedexpectedtax(φ, τ, τᶜ)
        investment = a ≥ firm.η ? zero(q) : α(q, firm)
        flow = simplifiedcost(τᵉ, a, investment, household, firm, model.climate)
        if a ≥ firm.η
            diagonal[i], rhs[i] = one(q), flow / household.r
            continue
        end
        transition = noncommittedtransition(φ, a, q, τ, τᶜ, parameters, Δt)
        discount, flowweight, timeweight = noncommittedweights(transition, household.r, Δt)
        current = discount * (1 - timeweight)
        diagonal[i] = 1 - current * transition.p₀
        i > 1 && (lower[i - 1] = -current * transition.pᵩ⁻)
        i < size(grid, 1) && (upper[i] = -current * transition.pᵩ⁺)
        rhs[i] = flowweight * flow
        j < size(grid, 2) && (rhs[i] += current * transition.pₐ⁺ * V.u[i, j + 1])
        !isnothing(V′) && (rhs[i] += discount * timeweight * noncommittedcontinuation(V′.u, i, j, transition))
    end
    return nothing
end

function noncommittedstatictax(φ, a, τᶜ, parameters)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    curvature = κ₂(a, household, firm, climate)
    iszero(curvature) && return zero(τᶜ)
    return clamp(-κ₁(a, household, firm, climate) / curvature, zero(τᶜ), τᶜ)
end

function noncommittedinitial(t, τᶜ, parameters)
    model, grid = parameters.model, parameters.grid
    household, firm = model.household, model.firm
    q, uᵍ, u, τ = (zeros(typeof(grid.h), size(grid)) for _ in 1:4)
    @inbounds for j in axes(q, 2), i in axes(q, 1)
        φ, a = physicalstate(grid, i, j)
        τ[i, j] = noncommittedstatictax(φ, a, τᶜ, parameters)
        τᵉ = noncommittedexpectedtax(φ, τ[i, j], τᶜ)
        q[i, j] = y(n(τᵉ, a, household, firm), firm) * τᵉ / household.r
        a ≥ firm.η && (q[i, j] = min(c(firm), q[i, j]))
        investment = a ≥ firm.η ? zero(t) : α(q[i, j], firm)
        uᵍ[i, j] = simplifiedcost(τ[i, j], a, investment, household, firm, model.climate) / household.r
        u[i, j] = simplifiedcost(τᵉ, a, investment, household, firm, model.climate) / household.r
    end
    return NonCommittedMarkovValue(t, grid, q, uᵍ, u, τ)
end

function noncommittedstep!(V, V′, τᶜ, parameters, Δt; verbose = 0)
    model, grid = parameters.model, parameters.grid
    household, firm, climate = model.household, model.firm, model.climate
    Nᵩ, Nₐ = size(grid)
    T = eltype(V.q)
    qscale = max(one(T), c(firm), firm.A * τᶜ / household.r)
    valuescale = max(one(T), firm.A / household.r)
    taxscale = max(τᶜ, sqrt(eps(T)))
    x, residual = zeros(T, 3Nᵩ), zeros(T, 3Nᵩ)
    lower, diagonal, upper, rhs = zeros(T, Nᵩ - 1), zeros(T, Nᵩ), zeros(T, Nᵩ - 1), zeros(T, Nᵩ)
    prototype = BM.BandedMatrix{T}(undef, (3Nᵩ, 3Nᵩ), (5, 5))
    fill!(BM.bandeddata(prototype), one(T))
    algorithm = NLS.NewtonRaphson(; autodiff = ADTypes.AutoForwardDiff(), linesearch = NLS.BackTracking(), linsolve = LS.LUFactorization())
    iterations, valuegap, bellmangap, policygap, improvement = 0, zero(T), zero(T), zero(T), zero(T)
    # Irreversibility makes the system triangular in a; interleave q,uᵍ,τ within each φ node.
    for j in Nₐ:-1:1
        a = physicalstate(grid, 1, j)[2]
        coefficients = promote(τᶜ, Δt, qscale, valuescale, taxscale, κ₀(a, household, firm, climate), κ₁(a, household, firm, climate), κ₂(a, household, firm, climate))
        column = NonCommittedColumn(V, V′, parameters, j, coefficients...)
        @inbounds for i in 1:Nᵩ
            x[3i - 2] = V.q[i, j] / qscale
            x[3i - 1] = V.uᵍ[i, j] / valuescale
            x[3i] = V.τ[i, j] / taxscale
        end
        nonlinearfunction = SciMLBase.NonlinearFunction{true}(noncommittedcolumnresidual!; jac_prototype = prototype)
        problem = SciMLBase.NonlinearProblem(nonlinearfunction, copy(x), column)
        solved = NLS.solve(problem, algorithm; abstol = min(parameters.tolerance, parameters.policytolerance) / 4, reltol = zero(T), maxiters = parameters.maxiters)
        SciMLBase.successful_retcode(solved) || error("Coupled policy/valuation solve failed at t=$(V.t), a=$a: $(solved.retcode); residual=$(maximum(abs, solved.resid)).")
        copyto!(x, solved.u)
        iterations += solved.stats.nsteps
        @inbounds for i in 1:Nᵩ
            V.q[i, j] = max(qscale * x[3i - 2], zero(T))
            V.uᵍ[i, j] = valuescale * x[3i - 1]
            V.τ[i, j] = clamp(taxscale * x[3i], zero(T), τᶜ)
            x[3i - 2], x[3i] = V.q[i, j] / qscale, V.τ[i, j] / taxscale
        end
        noncommittedcolumnresidual!(residual, x, column)
        @inbounds for i in 1:Nᵩ
            valuegap = max(valuegap, abs(residual[3i - 2]))
            bellmangap = max(bellmangap, abs(residual[3i - 1]))
            policygap = max(policygap, abs(residual[3i]))
            φ = physicalstate(grid, i, j)[1]
            transition = noncommittedtransition(φ, a, V.q[i, j], V.τ[i, j], τᶜ, parameters, Δt)
            _, gain = noncommittedimprovement(x, i, column, transition)
            improvement = max(improvement, gain)
        end
        noncommittedsocialcolumn!(lower, diagonal, upper, rhs, V, V′, j, τᶜ, parameters, Δt)
        LA.ldiv!(LA.lu!(LA.Tridiagonal(lower, diagonal, upper)), rhs)
        copyto!(view(V.u, :, j), rhs)
        verbose > 1 && j % 10 == 0 && @info "Solved coupled abatement layer" time = V.t abatement = a valuegap bellmangap policygap
    end
    max(valuegap, bellmangap) ≤ parameters.tolerance && policygap ≤ parameters.policytolerance || error("Coupled residuals exceed tolerance at t=$(V.t): pricing=$valuegap, Bellman=$bellmangap, policy=$policygap.")
    return NonCommittedMarkovDiagnostics(iterations, valuegap, bellmangap, policygap, improvement)
end

"""
Coupled expected firm valuation and hidden government policy on a normalised (φ,a) grid.

The reduced government objective uᵍ uses the explicit κ coefficients at actual τ; household welfare u uses expected τ. The old Government fields are retained for compatibility and do not enter either objective. Installation has constant price c(firm). Firm valuations anticipate future learning and policy, replacing the two prescribed forecast curves. At a=η a one-sided marginal-value obstacle enforces q≤c and investment is zero. A stationary constant-announcement tail supplies terminal values, followed by backward calendar-time solves. Known announcement jumps are included in the calendar mesh; stored policy at a jump uses the new regime. Irreversibility reduces each joint Newton problem to one belief column; no q grid or outer forecast-path iteration is used. Running flows are integrated exactly over each exponentially discounted holding interval.
"""
function solvenoncommitted(τᶜ::CommittedTaxPath, model::LinearLabourModel, government::Government, signal::Signal; h = 1 / 40, timestep = 1.0, maxholdingtime = 1.0, horizon = last(τᶜ.ts), maxiters = 1000, tolerance = 1e-8, policytolerance = 1e-7, relaxation = 0.5, verbose = 1)
    all(x -> isfinite(x) && x > 0, (timestep, maxholdingtime, horizon, tolerance, policytolerance)) && maxiters > 0 || throw(ArgumentError("Time steps, horizon, tolerances and iteration limit must be finite and positive."))
    all(x -> isfinite(x) && x > 0, (signal.σ, model.household.r, model.household.ν, model.firm.A, model.firm.ξ)) && all(x -> isfinite(x) && x ≥ 0, (signal.ϵ, model.climate.γ, c(model.firm))) || throw(ArgumentError("Model parameters must be finite: discount, labour scale, productivity, noise and adjustment cost positive; price, damages and signal sensitivity nonnegative."))
    all(τ -> isfinite(τ) && 0 ≤ τ ≤ inv(model.firm.η), (τᶜ.taxpath..., τᶜ.tailtax)) || throw(ArgumentError("The announced taxes must lie in [0,1/η], preserving positive employment."))
    horizon ≥ last(τᶜ.ts) && issorted(τᶜ.ts) || throw(ArgumentError("The horizon must include the ordered announcement path."))
    grid = NonCommittedMarkovGrid(model, h)
    # Retained legacy field; globalization is now Newton backtracking, not policy damping.
    parameters = NonCommittedMarkovParameters(model, government, signal, τᶜ, grid, maxholdingtime; tolerance, policytolerance, relaxation, maxiters)
    jumps = [τᶜ.ts[k] for k in 1:(length(τᶜ.ts) - 1) if τᶜ.ts[k] == τᶜ.ts[k + 1] && τᶜ.taxpath[k] != τᶜ.taxpath[k + 1] && 0 < τᶜ.ts[k] < horizon]
    times = sort!(unique!(vcat(collect(range(zero(horizon), horizon; length = ceil(Int, horizon / timestep) + 1)), jumps)))
    terminal = noncommittedinitial(last(times), τᶜ.tailtax, parameters)
    verbose > 0 && @info "Solving stationary coupled tail" tax = τᶜ.tailtax gridsize = size(grid)
    taildiagnostics = noncommittedstep!(terminal, nothing, τᶜ.tailtax, parameters, Inf; verbose)
    values = Vector{typeof(terminal)}(undef, length(times))
    diagnostics = Vector{typeof(taildiagnostics)}(undef, length(times))
    values[end], diagnostics[end] = terminal, taildiagnostics
    for k in (length(times) - 1):-1:1
        future = values[k + 1]
        mandate = τᶜ(times[k])
        current = NonCommittedMarkovValue(times[k], grid, copy(future.q), copy(future.uᵍ), copy(future.u), clamp.(copy(future.τ), zero(mandate), mandate))
        diagnostics[k] = noncommittedstep!(current, future, mandate, parameters, times[k + 1] - times[k]; verbose)
        values[k] = current
        if verbose > 0 && (k == 1 || k % 10 == 0)
            @info "Solved coupled time slice" time = times[k] iterations = diagnostics[k].iterations pricing = diagnostics[k].valuegap bellman = diagnostics[k].bellmangap policy = diagnostics[k].policygap
        end
    end
    return NonCommittedMarkovSolution(times, values, parameters, diagnostics)
end
