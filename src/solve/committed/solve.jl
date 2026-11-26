## Normalised states and solver time

include("equilibrium.jl")
include("dynamics.jl")

struct ScaledCommittedParameters{P <: CommittedParameters, T}
    parameters::P
    horizon::T
    scale::SA.SVector{8, T}
end

function dynamics(u::SA.StaticVector{8}, p::ScaledCommittedParameters, s)
    p.horizon .* dynamics(u .* p.scale, p.parameters, p.horizon * s) ./ p.scale
end

function dynamics!(du::AbstractVector, u::AbstractVector, p::Union{CommittedParameters, ScaledCommittedParameters}, t)
    copyto!(du, dynamics(SA.SVector{8}(u), p, t))
    nothing
end

function initialboundary(u::SA.StaticVector{8}, p::ScaledCommittedParameters)
    SA.SVector(u[1] - p.parameters.xᶜ₀ / p.scale[1], u[6], committedcomplementarity(u[7], u[3]), u[8])
end

function initialboundary!(residual::AbstractVector, u::AbstractVector, p::Union{CommittedParameters, ScaledCommittedParameters})
    copyto!(residual, initialboundary(SA.SVector{8}(u), p))
    nothing
end

function terminalboundary(u::SA.StaticVector{8}, p::ScaledCommittedParameters)
    model = p.parameters.model
    scale = p.scale
    physical = u .* scale
    a, taxmultiplier, investmentmultiplier = committedterminal(physical, p.parameters)
    taxscale = model.household.r * scale[2] / model.dirty.η
    costscale = scale[1] * scale[5]
    taxcondition = committedcomplementarity(taxmultiplier * taxscale / costscale, a.τ / taxscale)
    investmentcondition = committedcomplementarity(investmentmultiplier / scale[6], (model.clean.c - physical[2]) / scale[2])

    SA.SVector((physical[3] - a.profit / model.household.r) / scale[3], (physical[4] - physical[2] * physical[1]) / scale[4], taxcondition, investmentcondition)
end

function terminalboundary!(residual::AbstractVector, u::AbstractVector, p::Union{CommittedParameters, ScaledCommittedParameters})
    copyto!(residual, terminalboundary(SA.SVector{8}(u), p))
    nothing
end

"Return (solution, scale); physical states are solution(s) .* scale, with s=t/horizon."
function solvecommitted(model, xᶜ₀; horizon=400.0, nodes=81, tolerance=1e-8, initialtax=0.03, adaptive=false, maxiters=200, verbose=true, guess=nothing)
    @unpack household, firm, dirty, clean = model

    if !(firm isa CESFirm && firm.ρ < 1) || firm.ρ ≈ 0
        error("The committed solver requires CES production with ρ < 1 and ρ ≠ 0.")
    end
    if !(horizon > 0 && nodes >= 3 && tolerance > 0 && dirty.ζ > 0 && xᶜ₀ > 0)
        error("Require a positive horizon, tolerance, compliance cost and initial capacity, with at least three mesh points.")
    end

    started = time_ns()
    r = household.r
    reference = committedstationary(model, initialtax)
    xscale, qscale = max(xᶜ₀, dirty.kᵈ), clean.c
    vscale = dirty.γᵈ * dirty.kᵈ^2 / (2r)
    costscale = reference.y / r
    vcscale = qscale * xscale
    # Order: xᶜ, q, vᵈ, vᶜ, λxᶜ, λq, λvᵈ, λvᶜ.
    scale = SA.SVector(xscale, qscale, vscale, vcscale, costscale / xscale, costscale / qscale, costscale / vscale, costscale / vcscale)
    initialcapacity, referencez = promote(xᶜ₀, reference.z)
    parameters = CommittedParameters(model, initialcapacity, referencez)
    scaledparameters = ScaledCommittedParameters(parameters, convert(eltype(scale), horizon), scale)

    initialguess = if isnothing(guess)
        initialresidual = @closure z -> begin
            g = committedgeometry(model, z, reference.vᵈ, reference.vᶜ)
            g.x - xᶜ₀ - dirtyoutput(g.p, initialtax, dirty)
        end
        initialz = committedroot(initialresidual, (:initial, xᶜ₀, reference.vᵈ))
        initial = committedallocationat(model, xᶜ₀, reference.vᵈ, reference.vᶜ, clean.c, initialz, initialtax)
        terminalx = max(xᶜ₀, reference.xᶜ)
        tail, ∂f, ∂g, ∂τ = committedtail(model, terminalx, reference.vᵈ, clean.c)
        terminalq = tail.p / r + clean.c * cbrt(eps(one(r)))
        @closure (p, s) -> begin
            xᶜ = (1 - s) * xᶜ₀ + s * terminalx
            q = (1 - s) * max(initial.p / r, clean.c) + s * terminalq
            collect(SA.SVector(xᶜ, q, (1 - s) * initial.profit / r + s * reference.vᵈ, q * xᶜ, ∂g[1], s * ∂g[3], s * ∂g[2], zero(r)) ./ scale)
        end
    else
        previous, previousscale = guess
        if length(previousscale) != 8
            error("The warm start must use the eight-variable green-equity system.")
        end
        @closure (p, s) -> collect(previous(s) .* previousscale ./ scale)
    end

    bvpfunction = SciMLBase.BVPFunction{true}(dynamics!, (initialboundary!, terminalboundary!); bcresid_prototype=(zeros(eltype(scale), 4), zeros(eltype(scale), 4)), twopoint=Val(true))
    bvp = SciMLBase.TwoPointBVProblem(bvpfunction, initialguess, (zero(r), one(r)), scaledparameters)
    jacobian = BVD.BVPJacobianAlgorithm(ADTypes.AutoSparse(ADTypes.AutoFiniteDiff()))
    nonlinear = NonlinearSolve.LevenbergMarquardt(; disable_geodesic=Val(true))
    if verbose
        @info "Solving committed BVP" horizon nodes tolerance adaptive maxiters
    end
    solution = BVD.solve(bvp, BVD.MIRK4(; jac_alg=jacobian, nlsolve=nonlinear); dt=inv(nodes - 1), adaptive, abstol=tolerance, reltol=tolerance, nlsolve_kwargs=(; abstol=tolerance, reltol=tolerance, maxiters, show_trace=Val(verbose)), verbose)
    gap = maximum(abs, solution.resid)
    if verbose
        @info "Committed BVP finished" retcode=solution.retcode residual=gap meshpoints=length(solution.t) elapsed=(time_ns() - started) / 1e9
    end
    if !SciMLBase.successful_retcode(solution)
        error("Committed BVP failed: $(solution.retcode), residual=$(gap).")
    end
    if !(gap <= tolerance)
        error("Committed collocation residual is $(gap), above tolerance=$(tolerance).")
    end
    if !adaptive
        # This MIRK version prepares dense-output stages only during adaptive error estimation.
        BVD.BoundaryValueDiffEqMIRK.interp_setup!(solution.interp.cache)
    end

    states = map(u -> u .* scale, solution.u)
    if !(minimum(u -> min(u[1] / scale[1], u[2] / scale[2], u[3] / scale[3], u[4] / scale[4]), states) >= -tolerance)
        error("Committed solution has negative private states.")
    end
    terminal = last(states)
    if abs(terminal[4] - terminal[2] * terminal[1]) > tolerance * scale[4]
        error("Terminal green equity is inconsistent with its stationary cash flows.")
    end
    if !(terminal[3] > 0)
        error("The interior tail requires positive dirty value; a shutdown tail needs separate corner conditions.")
    end
    tail, ∂f, ∂g, ∂τ = committedtail(model, terminal[1], terminal[3], terminal[2])
    if abs(∂τ[1] * scale[1] * dirty.η / (r * scale[2])) <= sqrt(eps(one(r)))
        error("The stationary endpoint chart is singular; change the terminal parametrisation at this fold.")
    end
    if !(0 < tail.xᵈ < dirty.kᵈ)
        error("The stationary reduction requires interior dirty output; recovered output=$(tail.xᵈ).")
    end
    if !(tail.e > 0)
        error("The stationary tail requires partial compliance; emissions=$(tail.e). A zero-emission tail needs its own endpoint KKT conditions.")
    end
    taxtolerance = tolerance * r * scale[2] / dirty.η
    if !(tail.τ >= -taxtolerance)
        error("The tail violates the non-negative tax bound: tax=$(tail.τ), feasibility tolerance=$(taxtolerance).")
    end
    if tail.τ < 0
        @warn "Tail tax is slightly negative within the requested feasibility tolerance; reduce tolerance to resolve the zero-tax boundary more precisely." tax=tail.τ feasibilitytolerance=taxtolerance
    end
    if !(terminal[2] <= clean.c + tolerance * scale[2])
        error("Terminal investment remains positive.")
    end

    marketgap, policygap = zero(r), zero(r)
    for u in states
        a, ∂w, ∂p, ∂π = committedpolicy(model, u[1], u[3], u[4], u[2], u[6], u[7], u[8]; zguess=reference.z)
        marketgap = max(marketgap, abs(a.x - a.xᶜ - a.xᵈ))
        slope = ∂w[4] - (u[6] + u[1] * u[8]) * ∂p[4] - u[7] * ∂π[4]
        violation = a.τ <= 64eps(one(r)) ? max(-slope, zero(r)) : 0 < a.xᵈ < dirty.kᵈ ? abs(slope) : zero(r)
        policygap = max(policygap, violation)
    end
    if verbose
        @info "Committed equilibrium checks" market=marketgap policy=policygap tailtax=tail.τ tailabatement=tail.h tailemissions=tail.e greenvalue=terminal[4]
    end
    if policygap > tolerance
        @warn "The selected policy branch has a nonzero Hamiltonian KKT residual." residual=policygap
    end
    solution, scale
end
