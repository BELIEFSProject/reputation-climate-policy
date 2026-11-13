struct NCDAEParameters{M, G, S, C, D, T}
    model::M
    government::G
    signal::S
    τᶜ::C
    grid::D
    horizon::T
    timestep::T
    qscale::T
    valuescale::T
    taxscale::T
    tolerance::T
    policytolerance::T
    maxiters::Int
end

function NCDAEParameters(τᶜ::CommittedTaxPath, model::AbstractModel, government::Government, signal::Signal; h = 1 / 40, timestep = 1.0, horizon = last(τᶜ.ts), tolerance = 1e-8, policytolerance = 1e-7, maxiters = 1000)
    household, firm = model.household, model.firm
    if household isa Household && household.ϕ != 2
        throw(ArgumentError("The non-committed solver supports inverse Frisch elasticities 1 and 2."))
    end
    if !isfinite(government.κᵇ) || government.κᵇ ≤ 0 || !isfinite(government.τ̄) || government.τ̄ < 0
        throw(ArgumentError("Behavioural curvature must be positive and the preferred tax must be non-negative."))
    end
    if !all(x -> isfinite(x) && x > 0, (horizon, timestep, tolerance, policytolerance, household.r, household.ν, firm.A, firm.ξ, signal.σ)) || maxiters < 1
        throw(ArgumentError("The horizon, tolerances, discount, labour scale, productivity, adjustment cost and noise must be positive."))
    end
    if horizon < last(τᶜ.ts) || !issorted(τᶜ.ts) || !all(τ -> isfinite(τ) && 0 ≤ τ ≤ inv(firm.η), (τᶜ.taxpath..., τᶜ.tailtax)) || signal.ϵ < 0 || model.climate.γ < 0
        throw(ArgumentError("The horizon must include the ordered announcement; taxes, signal sensitivity and damages must be admissible."))
    end

    grid = NonCommittedMarkovGrid(model, h)
    output = household isa LinearHousehold ? firm.A^2 / household.ν : y(n(0, firm.a₀, household, firm), firm)
    maxtax = max(maximum(τᶜ.taxpath), τᶜ.tailtax)
    qscale = max(1, c(firm), output * maxtax / household.r)
    valuescale = max(1, output / household.r)
    taxscale = max(maxtax, sqrt(eps(typeof(grid.h))))
    scales = promote(float(horizon), timestep, qscale, valuescale, taxscale, tolerance, policytolerance)
    return NCDAEParameters(model, government, signal, τᶜ, grid, scales..., maxiters)
end

function ncdaemandata(parameters, s, interval)
    if !isnothing(interval)
        if s == parameters.horizon - first(interval)
            return parameters.τᶜ(first(interval); side = :left)
        elseif s == parameters.horizon - last(interval)
            return parameters.τᶜ(last(interval))
        end
    end
    return parameters.τᶜ(parameters.horizon - s)
end

function ncdaegenerator(x, component, i, j, λₐ, λ⁺, λ⁻)
    center = x[component, i, j]
    value = zero(center)
    if i > 1
        value += λ⁻ * (x[component, i - 1, j] - center)
    end
    if i < size(x, 2)
        value += λ⁺ * (x[component, i + 1, j] - center)
    end
    if j < size(x, 3)
        value += λₐ * (x[component, i, j + 1] - center)
    end
    return value
end

"Hidden best response, or largest consistent tax for projection, with conjectures held fixed."
function ncdaetax(x, τ, i, j, τᶜ, parameters; equilibrium = false)
    grid, government = parameters.grid, parameters.government
    φ, a = physicalstate(grid, i, j)
    if iszero(τᶜ)
        return zero(τ), zero(τ)
    end

    coefficient = φ * (1 - φ) * (parameters.signal.ϵ / parameters.signal.σ)^2
    B = coefficient * (τᶜ - τ)
    slope = zero(τ)
    if 1 < i < size(grid, 1) && a < parameters.model.firm.η
        Δφ = grid.h * span(grid.φ)
        slope = parameters.valuescale * (x[2, i + 1, j] - x[2, i - 1, j]) / (2Δφ)
    end

    linear, curvature = -government.κᵇ * government.τ̄, government.κᵇ
    candidate = clamp(-(linear + B * slope) / curvature, 0, τᶜ)
    if equilibrium
        uppergradient = linear + curvature * τᶜ
        if uppergradient ≤ 0
            candidate = τᶜ
        else
            lowergradient = linear + coefficient * τᶜ * slope
            gain = max(-lowergradient, 0)
            candidate = τᶜ * gain / (uppergradient + gain)
        end
    end
    improvement = max((τ - candidate) * (linear + B * slope + curvature * (τ + candidate) / 2), 0)
    return candidate, improvement
end

"Value rates in remaining calendar time s=T-t; tax is an algebraic residual."
function ncdaefield!(field, state, parameters, s; interval = nothing)
    model, grid = parameters.model, parameters.grid
    household, firm, climate = model.household, model.firm, model.climate
    x = reshape(state, 4, size(grid)...)
    f = reshape(field, size(x))
    τᶜ = ncdaemandata(parameters, s, interval)
    Δφ, Δa = grid.h * span(grid.φ), grid.h * span(grid.A)

    @inbounds Threads.@threads :dynamic for idx in CartesianIndices(grid)
        i, j = idx.I
        φ, a = physicalstate(grid, i, j)
        q = parameters.qscale * x[1, i, j]
        τ = clamp(parameters.taxscale * x[4, i, j], 0, τᶜ)
        τᵉ = ncexpectedtax(φ, τ, τᶜ)
        investment = j == size(grid, 2) ? 0 : α(q, firm)
        μᵩ, σᵩ = ncbeliefmoments(τ, τᶜ, φ, parameters.signal)
        B = φ * (1 - φ) * (parameters.signal.ϵ / parameters.signal.σ)^2 * (τᶜ - τ)
        # Freeze enough diffusion to keep both belief rates non-negative under every hidden deviation.
        variance = max(σᵩ^2 / (2Δφ^2), B * max(τᵉ, τᶜ - τᵉ) / (2Δφ))
        λₐ = investment / Δa
        λ⁺, λ⁻ = max(variance + μᵩ / (2Δφ), 0), max(variance - μᵩ / (2Δφ), 0)

        # Firm valuation uses the public measure, under which beliefs are martingales.
        f[1, i, j] = ncdaegenerator(x, 1, i, j, λₐ, variance, variance) + y(n(τᵉ, a, household, firm), firm) * τᵉ / parameters.qscale - household.r * x[1, i, j]
        if j == size(grid, 2)
            f[2, i, j], f[3, i, j] = 0, 0
        else
            f[2, i, j] = ncdaegenerator(x, 2, i, j, λₐ, λ⁺, λ⁻) + simplifiedcost(τᵉ, a, investment, household, firm, climate; relative = true) / parameters.valuescale - household.r * x[2, i, j]
            f[3, i, j] = ncdaegenerator(x, 3, i, j, λₐ, λ⁺, λ⁻) + simplifiedcost(τᵉ, a, investment, household, firm, climate; relative = true) / parameters.valuescale - household.r * x[3, i, j]
        end
        # The nonlinear tax is already an algebraic state: enforce its exact best-response fixed point.
        candidate, _ = ncdaetax(x, τ, i, j, τᶜ, parameters; equilibrium = true)
        f[4, i, j] = x[4, i, j] - candidate / parameters.taxscale
    end
    return nothing
end

function ncdaeresidual!(residual, derivative, state, parameters, s; interval = nothing)
    ncdaefield!(residual, state, parameters, s; interval)
    x = reshape(state, 4, size(parameters.grid)...)
    dx, r = reshape(derivative, size(x)), reshape(residual, size(x))
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate

    @inbounds for j in axes(x, 3), i in axes(x, 2)
        r[1, i, j] = (dx[1, i, j] - r[1, i, j]) / household.r
        if j == size(x, 3)
            # Upper obstacle: max(r*(q-c), q_s-fq)=0, in normalised units.
            obstacle = x[1, i, j] - c(firm) / parameters.qscale
            if FD.value(obstacle) ≥ FD.value(r[1, i, j])
                r[1, i, j] = obstacle
            end
            r[2, i, j], r[3, i, j] = x[2, i, j], x[3, i, j]
        else
            r[2, i, j] = (dx[2, i, j] - r[2, i, j]) / household.r
            r[3, i, j] = (dx[3, i, j] - r[3, i, j]) / household.r
        end
    end
    return nothing
end

"Conservative stencil, including couplings that vanish at a mimicking initial guess."
function ncdaejacobian(parameters)
    nφ, na = size(parameters.grid)
    rows, columns = Int[], Int[]
    for j in 1:na, i in 1:nφ
        node = i + (j - 1) * nφ
        neighbours = [node]
        if i > 1
            push!(neighbours, node - 1)
        end
        if i < nφ
            push!(neighbours, node + 1)
        end
        if j < na
            push!(neighbours, node + nφ)
        end
        for neighbour in neighbours, row in 1:4, column in 1:4
            push!(rows, 4node - 4 + row)
            push!(columns, 4neighbour - 4 + column)
        end
    end
    return SP.sparse(rows, columns, ones(typeof(parameters.qscale), length(rows)), 4nφ * na, 4nφ * na)
end

"Prepare the sparse combined Jacobian ∂ₓR + γ∂ₓ̇R for implicit time steps."
function ncdaejacobian(residual!, derivative, state, parameters, s, autodiff)
    buffer = similar(state)
    combined! = (residual, z, dx, x, parameters, γ, s) -> residual!(residual, dx .+ γ .* (z .- x), z, parameters, s)
    preparation = DI.prepare_jacobian(combined!, buffer, autodiff, state, DI.Constant(derivative), DI.Constant(state), DI.Constant(parameters), DI.Constant(one(s)), DI.Constant(s))
    jacobian! = function (jacobian, dx, x, parameters, γ, s)
        DI.jacobian!(combined!, buffer, jacobian, preparation, autodiff, x, DI.Constant(dx), DI.Constant(x), DI.Constant(parameters), DI.Constant(γ), DI.Constant(s))
        return nothing
    end
    return jacobian!
end

function ncstate(value::NCMarkovValue{T}, parameters) where T
    x = zeros(T, 4, size(parameters.grid)...)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    baseline = κ₀(firm.η, household, firm, climate) / household.r
    x[1, :, :] .= value.q ./ parameters.qscale
    x[2, :, :] .= (value.uᵍ .- baseline) ./ parameters.valuescale
    x[3, :, :] .= (value.u .- baseline) ./ parameters.valuescale
    x[4, :, :] .= value.τ ./ parameters.taxscale
    return vec(x)
end

function ncdaevalue(state, parameters, s; calendar = parameters.horizon - s)
    x = reshape(state, 4, size(parameters.grid)...)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    baseline = κ₀(firm.η, household, firm, climate) / household.r
    @views value = NCMarkovValue(calendar, parameters.grid, parameters.qscale .* x[1, :, :], baseline .+ parameters.valuescale .* x[2, :, :], baseline .+ parameters.valuescale .* x[3, :, :], clamp.(parameters.taxscale .* x[4, :, :], 0, parameters.τᶜ(calendar)))

    return value
end

function ncdaeprojecttax!(state, parameters, s; interval = nothing)
    x = reshape(state, 4, size(parameters.grid)...)
    mandate = ncdaemandata(parameters, s, interval)

    @inbounds for j in axes(x, 3), i in axes(x, 2)
        candidate, _ = ncdaetax(x, parameters.taxscale * x[4, i, j], i, j, mandate, parameters; equilibrium = true)
        x[4, i, j] = candidate / parameters.taxscale
    end
end

function ncdaederivative(state, parameters, s; interval = nothing)
    derivative = similar(state)
    ncdaefield!(derivative, state, parameters, s; interval)
    x = reshape(state, 4, size(parameters.grid)...)
    dx = reshape(derivative, size(x))
    dx[4, :, :] .= 0
    @inbounds for i in axes(x, 2)
        if x[1, i, end] ≥ c(parameters.model.firm) / parameters.qscale
            dx[1, i, end] = min(dx[1, i, end], 0)
        end
    end

    return derivative
end

function ncdaediagnostics(state, derivative, parameters, s; interval = nothing, iterations = 0)
    residual = similar(state)
    ncdaeresidual!(residual, derivative, state, parameters, s; interval)
    r = reshape(residual, 4, size(parameters.grid)...)
    x = reshape(state, size(r))
    mandate = ncdaemandata(parameters, s, interval)
    improvement = zero(parameters.taxscale)
    @inbounds for j in axes(x, 3), i in axes(x, 2)
        τ = clamp(parameters.taxscale * x[4, i, j], 0, mandate)
        _, gain = ncdaetax(x, τ, i, j, mandate, parameters)
        improvement = max(improvement, gain)
    end
    return NonCommittedMarkovDiagnostics(iterations, maximum(abs, view(r, 1, :, :)), max(maximum(abs, view(r, 2, :, :)), maximum(abs, view(r, 3, :, :))), maximum(abs, view(r, 4, :, :)), improvement)
end

"Solve the Markov equilibrium by fully implicit time steps, retaining tax and capacity as algebraic constraints."
function solvencdae(τᶜ::CommittedTaxPath{TI, T, TT}, model::AbstractModel, government::Government, signal::Signal; h = 1 / 40, timestep = 1.0, savestep = timestep, horizon = last(τᶜ.ts), tolerance = 1e-8, policytolerance = 1e-7, maxiters = 1000, verbose = 1) where {TI, T, TT}
    parameters = NCDAEParameters(τᶜ, model, government, signal; h, timestep, horizon, tolerance, policytolerance, maxiters)
    internalnorm = x -> maximum(abs, x)
    prototype = ncdaejacobian(parameters)
    autodiff = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector = ADTypes.KnownJacobianSparsityDetector(prototype), coloring_algorithm = SMC.GreedyColoringAlgorithm())
    state = ncstate(ncinitial(parameters.horizon, τᶜ.tailtax, parameters), parameters)

    ncdaeprojecttax!(state, parameters, 0)

    stationaryderivative = zeros(TT, length(state))
    stationary! = (residual, x, parameters) -> ncdaeresidual!(residual, stationaryderivative, x, parameters, 0)

    tailresidual! = (residual, dx, x, parameters, s) -> ncdaeresidual!(residual, dx, x, parameters, s)
    tailjacobian! = ncdaejacobian(tailresidual!, stationaryderivative, state, parameters, zero(horizon), autodiff)
    stationaryjacobian! = (jacobian, x, parameters) -> tailjacobian!(jacobian, stationaryderivative, x, parameters, zero(horizon), zero(horizon))
    stationaryfunction = SciMLBase.NonlinearFunction{true}(stationary!; jac = stationaryjacobian!, jac_prototype = copy(prototype))
    # Explicit directional AD avoids the line search's dense analytic-Jacobian work array.
    stationaryalgorithm = NLS.NewtonRaphson(; autodiff = ADTypes.AutoForwardDiff(), jvp_autodiff = ADTypes.AutoForwardDiff(), linesearch = NLS.BackTracking(), linsolve = LS.LUFactorization())

    if verbose > 0
        @info "Solving continuous-time stationary tail" tax = τᶜ.tailtax gridsize = size(parameters.grid)
    end

    # Start from a well-conditioned preference and continue to the requested behavioural curvature.
    tailiterations = 0
    initialcurvature = ∂ₜₜsimplifiedcost(0, model.firm.a₀, model.household, model.firm, model.climate)
    stages = ceil(Int, log2(max(1, initialcurvature / government.κᵇ)))
    for stage in (4stages:-1:0) ./ 4
        tailgovernment = Government(; κᵇ = government.κᵇ * exp2(stage), τ̄ = government.τ̄)
        tailparameters = NCDAEParameters(τᶜ, model, tailgovernment, signal; h, timestep, horizon, tolerance, policytolerance, maxiters)
        ncdaeprojecttax!(state, tailparameters, 0)
        stationaryproblem = SciMLBase.NonlinearProblem(stationaryfunction, state, tailparameters)
        stationary = NLS.solve(stationaryproblem, stationaryalgorithm; abstol = min(tolerance, policytolerance), reltol = 0, internalnorm, maxiters)
        tailiterations += stationary.stats.nsteps
        if !SciMLBase.successful_retcode(stationary)
            retryproblem = SciMLBase.NonlinearProblem(stationaryfunction, copy(stationary.u), tailparameters)
            retryalgorithm = NLS.TrustRegion(; autodiff = ADTypes.AutoForwardDiff(), linsolve = LS.LUFactorization(), jacobian_reuse = false)
            stationary = NLS.solve(retryproblem, retryalgorithm; abstol = min(tolerance, policytolerance), reltol = 0, internalnorm, maxiters)
            tailiterations += stationary.stats.nsteps
        end
        if !SciMLBase.successful_retcode(stationary)
            error("Continuous-time non-committed tail failed at curvature $(tailgovernment.κᵇ): $(stationary.retcode); residual=$(maximum(abs, stationary.resid)).")
        end
        state = copy(stationary.u)
        if verbose > 1
            @info "Solved stationary continuation stage" curvature = tailgovernment.κᵇ iterations = stationary.stats.nsteps residual = maximum(abs, stationary.resid)
        end
    end

    terminal = ncdaevalue(state, parameters, 0; calendar = parameters.horizon)

    values = [terminal]
    diagnostics = [ncdaediagnostics(state, stationaryderivative, parameters, 0; iterations = tailiterations)]

    jumps = TT[τᶜ.ts[k] for k in 1:(length(τᶜ.ts) - 1) if τᶜ.ts[k] == τᶜ.ts[k + 1] && τᶜ.taxpath[k] != τᶜ.taxpath[k + 1] && 0 < τᶜ.ts[k] < parameters.horizon]
    times = sort!(unique!(vcat(collect(range(0, parameters.horizon; length = ceil(Int, parameters.horizon / savestep) + 1)), jumps)))
    boundaries = reverse(sort!(unique!(vcat(0, parameters.horizon, jumps))))

    for k in 1:(length(boundaries) - 1)
        interval = (boundaries[k], boundaries[k + 1])
        left, right = parameters.horizon .- interval

        ncdaeprojecttax!(state, parameters, left; interval)

        calendars = reverse([t for t in times if last(interval) ≤ t < first(interval)])
        savetimes = parameters.horizon .- calendars
        stops = [parameters.horizon - t for t in τᶜ.ts if interval[2] < t < interval[1]]

        mesh = sort!(unique!(vcat(collect(range(left, right; length = ceil(Int, (right - left) / timestep) + 1)), stops, savetimes)))

        timeprecision = 32eps(typeof(parameters.horizon)) * max(1, parameters.horizon)
        mesh = mesh[vcat(true, diff(mesh) .> timeprecision)]
        mesh[end] = right
        previous = copy(state)
        stepsize, time = Ref(one(left)), Ref(left)
        daeresidual! = (residual, dx, x, parameters, s) -> ncdaeresidual!(residual, dx, x, parameters, s; interval)
        combinedjacobian! = ncdaejacobian(daeresidual!, zeros(TT, length(state)), state, parameters, left, autodiff)
        residual! = (residual, x, parameters) -> daeresidual!(residual, (x .- previous) ./ stepsize[], x, parameters, time[])
        jacobian! = (jacobian, x, parameters) -> combinedjacobian!(jacobian, (x .- previous) ./ stepsize[], x, parameters, inv(stepsize[]), time[])
        functionvalue = SciMLBase.NonlinearFunction{true}(residual!; jac = jacobian!, jac_prototype = copy(prototype))
        problem = SciMLBase.NonlinearProblem(functionvalue, copy(state), parameters)
        cache = NLS.init(problem, stationaryalgorithm; abstol = min(tolerance, policytolerance) / 10, reltol = 0, internalnorm, maxiters)
        saveindex, iterations = 1, 0

        for step in eachindex(mesh)
            if step > 1
                copyto!(previous, state)
                stepsize[], time[] = mesh[step] - mesh[step - 1], mesh[step]
                SciMLBase.reinit!(cache, state; p = parameters)
                solved = NLS.solve!(cache)
                iterations += solved.stats.nsteps
                if !SciMLBase.successful_retcode(solved)
                    # Policy and capacity switches can stall the Newton line search near a root.
                    if verbose > 0
                        @info "Retrying implicit step with a trust region" time = parameters.horizon-time[] residual = maximum(abs, solved.resid)
                    end
                    retryproblem = SciMLBase.NonlinearProblem(functionvalue, copy(solved.u), parameters)
                    retryalgorithm = NLS.TrustRegion(; autodiff = ADTypes.AutoForwardDiff(), linsolve = LS.LUFactorization(), jacobian_reuse = false)
                    solved = NLS.solve(retryproblem, retryalgorithm; abstol = min(tolerance, policytolerance) / 10, reltol = 0, internalnorm, maxiters)
                    iterations += solved.stats.nsteps
                end
                if !SciMLBase.successful_retcode(solved)
                    error("Non-committed implicit step failed at t=$(parameters.horizon-time[]): $(solved.retcode); residual=$(maximum(abs, solved.resid)).")
                end
                copyto!(state, solved.u)
                derivative = (state .- previous) ./ stepsize[]
            else
                derivative = ncdaederivative(state, parameters, left; interval)
            end
            while saveindex ≤ length(savetimes) && isapprox(time[], savetimes[saveindex]; rtol = 0, atol = timeprecision)
                push!(values, ncdaevalue(state, parameters, time[]; calendar = calendars[saveindex]))
                push!(diagnostics, ncdaediagnostics(state, derivative, parameters, time[]; interval, iterations))
                if verbose > 0 && (saveindex % 10 == 0 || saveindex == length(savetimes))
                    @info "Solved non-committed time slice" time = calendars[saveindex] iterations residual = max(diagnostics[end].valuegap, diagnostics[end].bellmangap, diagnostics[end].policygap)
                end
                saveindex += 1
            end
        end
        if verbose > 0
            @info "Solved non-committed implicit interval" interval steps = length(mesh) - 1 iterations
        end
    end

    return NonCommittedMarkovSolution(times, reverse!(values), parameters, reverse!(diagnostics))
end
