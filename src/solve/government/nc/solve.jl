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
