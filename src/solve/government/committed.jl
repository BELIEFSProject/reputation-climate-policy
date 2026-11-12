struct CommittedBVPSolution{M, I, V, T, R}
    model::M
    initialtax::I
    t::V
    a::V
    q::V
    tax::V
    investment::V
    λₐ::V
    λq::V
    welfarecost::T
    stoppingtime::T
    boundarygap::T
    retcode::R
end

"Lowest stationary price supporting marginal abatement at zero emissions."
function committedsupportingtax(model::AbstractModel)
    model.household.r * c(model.firm) / y(n(0, model.firm.η, model.household, model.firm), model.firm)
end

"Square-root labour supply: solve the stationary condition in sqrt(1-(η-a)τ)."
function committedtax(a, λq, model::Model)
    @unpack household, firm, climate = model
    if household.ϕ != 2
        throw(ArgumentError("The committed solver supports inverse Frisch elasticities 1 and 2."))
    end
    output = y(n(0, firm.η, household, firm), firm)
    if a ≥ firm.η
        return climate.γ * output
    end

    gap = firm.η - a
    upper = min(inv(firm.η), inv(gap))
    quadratic, linear, constant = gap + 3λq, 2climate.γ * gap^2 * output, gap + λq
    discriminant = linear^2 + 4quadratic * constant
    roots = if iszero(quadratic)
        iszero(linear) ? (-1, -1) : (constant / linear, -1)
    elseif discriminant > 0
        ((sqrt(discriminant) - linear) / (2quadratic), -(sqrt(discriminant) + linear) / (2quadratic))
    else
        (-1, -1)
    end

    # Subtract the zero-tax Hamiltonian analytically to retain accuracy near capacity.
    objective = function (τ)
        fraction = sqrt(max(1 - gap * τ, 0))
        return gap^2 * τ * (τ * (fraction + 2) / (3 * (1 + fraction)^2) - climate.γ * output) - λq * τ * fraction
    end
    tax, value = zero(a + λq), zero(a + λq)
    if objective(upper) < value
        tax, value = upper, objective(upper)
    end
    for fraction in roots
        if 0 ≤ fraction ≤ 1
            candidate = iszero(quadratic) ? (1 - fraction^2) / gap : (2λq + linear * fraction) / (gap * quadratic)
            if 0 ≤ candidate ≤ upper && objective(candidate) < value
                tax, value = candidate, objective(candidate)
            end
        end
    end
    return tax
end

"Bounded Hamiltonian minimisation, with the pre-contact price convention at full abatement."
function committedtax(a, λq, model::LinearLabourModel)
    @unpack household, firm, climate = model
    output = firm.A^2 / household.ν

    if a >= firm.η
        return climate.γ * output
    end

    linear = κ₁(a, household, firm, climate) - output * λq
    curvature = κ₂(a, household, firm, climate) + 2output * (firm.η - a) * λq
    upper = inv(firm.η)

    if curvature > 0
        return clamp(-linear / curvature, 0, upper)
    elseif linear * upper + curvature * upper^2 / 2 < 0
        return upper
    end

    return 0
end

"Regular field on reversed time s∈[0,1]; BVProblem supplies (x[3]-x[4])/s."
function committeddrift!(dx, x, model::AbstractModel, s)
    @unpack household, firm, climate = model

    ascale = firm.η - firm.a₀
    output = y(n(0, firm.η, household, firm), firm)
    tscale = sqrt(2firm.ξ * ascale / (climate.γ * output^2))
    qscale = firm.ξ * ascale / tscale
    λscale = climate.γ * output * ascale^2
    duration = tscale * exp(x[5])

    a = firm.η - ascale * x[1]
    q = c(firm) + qscale * x[2]
    λq = s * λscale * x[4]
    τ = committedtax(a, λq, model)

    source = y(n(τ, a, household, firm), firm) * τ
    sourcea = firm.A * τ * ∂ₐn(τ, a, household, firm)
    costa = ∂ₐsimplifiedcost(τ, a, 0, household, firm, climate)

    dx[1] = duration * qscale * x[2] / (firm.ξ * ascale)
    dx[2] = duration * (source - household.r * q) / qscale
    dx[3] = -duration * household.r * x[3] + duration^2 * (costa + source - λq * sourcea) / (firm.ξ * λscale)
    dx[4] = 0
    dx[5] = 0
    return nothing
end

function committedboundary!(residual, states, _, _, initialshadow)
    terminal, initial = first(states), last(states)

    residual[1] = terminal[1]                 # a=η.
    residual[2] = terminal[2]                 # q=c.
    residual[3] = terminal[4] - terminal[3]   # Regularity: S*x(0)=0.
    residual[4] = initial[1] - 1              # a=a₀.
    residual[5] = initial[4] - initialshadow  # Initial policy, or λq=0 for a fresh plan.
    return nothing
end

function initialcommittedguess(_, s, initialshadow)
    shadow = -s^3 * (1 - s)^2 + initialshadow * s^5
    shadowslope = -3s^2 * (1 - s)^2 + 2s^3 * (1 - s) + 5initialshadow * s^4
    # Ordinary vectors avoid the MVector cache-copy bug during sparsity tracing.
    return [s^2, 2s, shadow + s * shadowslope, shadow, 0]
end

function initialshadowcost(τ₀, model)
    @unpack household, firm, climate = model
    output = y(n(0, firm.η, household, firm), firm)
    gap = firm.η - firm.a₀
    λscale = climate.γ * output * gap^2

    # Invert the initial Hamiltonian FOC at a=a₀.
    ϕ = inversefrisch(household)
    initialoutput = y(n(τ₀, firm.a₀, household, firm), firm)
    λinitial = gap^2 * (τ₀ - 2climate.γ * initialoutput) / (ϕ - (1 + ϕ) * gap * τ₀)

    return λinitial / λscale
end

function committedproblem(model::AbstractModel; initialtax = nothing)
    if inversefrisch(model.household) ∉ (1, 2)
        throw(ArgumentError("The committed solver supports inverse Frisch elasticities 1 and 2."))
        # FIXME: Make a specialised solver for a model type with φ = 2, similar to the linear one.
    end

    initialshadow = isnothing(initialtax) ? 0 : initialshadowcost(initialtax, model)

    boundary! = @closure (residual, states, model, mesh) -> committedboundary!(residual, states, model, mesh, initialshadow)
    guess = @closure (model, s) -> initialcommittedguess(model, s, initialshadow)

    singular_term = zeros(5, 5)
    singular_term[4, 3] = 1
    singular_term[4, 4] = -1

    return BVP.BVProblem(committeddrift!, boundary!, guess, (0.0, 1.0), model; singular_term, nlls = Val(false))
end

"Hermite interpolation with the complete singular field, used for welfare quadrature."
function committedinterpolate(solution, s)
    j = clamp(searchsortedlast(solution.t, s), 1, length(solution.t) - 1)
    left, right = solution.u[j], solution.u[j + 1]
    sl, sr = solution.t[j], solution.t[j + 1]
    width = sr - sl
    θ = (s - sl) / width
    dl, dr = similar(left), similar(right)

    committeddrift!(dl, left, solution.prob.p, sl)
    committeddrift!(dr, right, solution.prob.p, sr)

    if sl > 0
        dl[4] += (left[3] - left[4]) / sl
    end
    dr[4] += (right[3] - right[4]) / sr

    return (2θ^3 - 3θ^2 + 1) .* left .+ width * (θ^3 - 2θ^2 + θ) .* dl .+ (-2θ^3 + 3θ^2) .* right .+ width * (θ^3 - θ^2) .* dr
end

"Solve the full-interval committed BVP; initialtax anchors an inherited plan instead of imposing λq(0)=0."
function solvecommitted(model::AbstractModel; initialtax = nothing, intervals = 100, verbose = 1)
    @unpack household, firm, climate = model
    problem = committedproblem(model; initialtax)

    if verbose > 0
        @info "Solving committed BVP" intervals initialtax
    end

    solution = BVP.solve(problem, BVP.MIRK4(); dt = inv(intervals), adaptive = false, verbose = verbose > 1)
    if !SciMLBase.successful_retcode(solution)
        @warn "Committed BVP failed with return code $(solution.retcode)."
    end

    ascale = firm.η - firm.a₀
    output = y(n(0, firm.η, household, firm), firm)
    tscale = sqrt(2firm.ξ * ascale / (climate.γ * output^2))
    qscale = firm.ξ * ascale / tscale
    λscale = climate.γ * output * ascale^2
    duration = tscale * exp(first(solution.u)[5])

    s = reverse(solution.t)
    states = reverse(solution.u)
    calendar = duration .* (1 .- s)
    a = [firm.η - ascale * x[1] for x in states]
    q = [c(firm) + qscale * x[2] for x in states]
    λq = [time * λscale * x[4] for (time, x) in zip(s, states)]
    λₐ = [firm.ξ * λscale * x[3] / duration - value for (x, value) in zip(states, q)]
    investment = (q .- c(firm)) ./ firm.ξ
    tax = similar(a)

    map!((state, shadow) -> committedtax(state, shadow, model), tax, a, λq)
    tax[end] = climate.γ * output # The terminal convention must not depend on roundoff in a=η.

    residual = similar(first(solution.u))
    problem.f.bc(residual, solution.u, model, solution.t)
    boundarygap = maximum(abs, residual)
    baseline = κ₀(firm.η, household, firm, climate)

    cost = @closure (s) -> begin
        x = committedinterpolate(solution, s)
        state = firm.η - ascale * x[1]
        shadow = s * λscale * x[4]
        τ = committedtax(state, shadow, model)
        return exp(-household.r * duration * (1 - s)) * (simplifiedcost(τ, state, qscale * x[2] / firm.ξ, household, firm, climate) - baseline)
    end

    # Baseline/r includes the stationary full-abatement tail.
    wtail, _ = QGK.quadgk(cost, solution.t...)
    welfarecost = baseline / household.r + duration * wtail
    if verbose > 0
        @info "Solved committed BVP" duration boundarygap retcode = solution.retcode
    end

    return CommittedBVPSolution(model, initialtax, calendar, a, q, tax, investment, λₐ, λq, welfarecost, duration, boundarygap, solution.retcode)
end

function CommittedTaxPath(solution::CommittedBVPSolution)
    support = committedsupportingtax(solution.model)
    return CommittedTaxPath([solution.tax; support], [solution.t; solution.stoppingtime], support)
end
