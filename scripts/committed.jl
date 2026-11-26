## Setup
import Revise: includet
import Dates
import Roots
import Printf
import StaticArrays as SA
import JLD2
import QuadGK

import ADTypes
import SciMLBase
import BoundaryValueDiffEq as BVD
import NonlinearSolve

import FastClosures: @closure
import UnPack: @unpack

includet("../src/primitives/calibration.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/households.jl")
includet("../src/agents/energy.jl")
includet("../src/agents/banking.jl")
includet("../src/agents/government.jl")
includet("../src/solve/equilibrium.jl")
includet("../src/solve/model.jl")
includet("../src/solve/committed/solve.jl")
includet("../src/solve/committed/diagnostics.jl")
includet("../src/utils/saving.jl")

calibration = Calibration()
model = GeneralEquilibriumModel(calibration)
xᶜ₀ = 0.4

## Solve
# Full-information commitment (φ=1); the private departure cost is zero.
solution, scale = solvecommitted(model, xᶜ₀; horizon=400.0, nodes=101, tolerance=1e-5, adaptive=false, maxiters=200, verbose=true)

## Diagnostics
# Solver time is s=t/horizon. Physical variables follow the order below.
xᶜ, q, vᵈ, vᶜ, λxᶜ, λq, λvᵈ, λvᶜ = solution(0.0) .* scale
initial, ∂w, ∂p, ∂π = committedpolicy(model, xᶜ, vᵈ, vᶜ, q, λq, λvᵈ, λvᶜ)
Printf.@printf("Initial: xc=%.6f q=%.6f vd=%.6f vc=%.6f tax=%.6f h=%.6f emissions=%.6f investment=%.6f\n", xᶜ, q, vᵈ, vᶜ, initial.τ, initial.h, initial.e, initial.investment)
Printf.@printf("Initial credit: loans=%.6f limit=%.6f multiplier=%.6f\n", initial.L, initial.L̄, initial.μ)

xᶜ, q, vᵈ, vᶜ, λxᶜ, λq, λvᵈ, λvᶜ = solution(1.0) .* scale
tail, ∂f, ∂g, ∂τ = committedtail(model, xᶜ, vᵈ, q)
Printf.@printf("Terminal: xc=%.6f q=%.6f vd=%.6f vc=%.6f tax=%.6f xd=%.6f h=%.6f emissions=%.6f\n", xᶜ, q, vᵈ, vᶜ, tail.τ, tail.xᵈ, tail.h, tail.e)
Printf.@printf("Collocation: retcode=%s residual=%.3e nodes=%d\n", string(solution.retcode), maximum(abs, solution.resid), length(solution.t))
valuations, valuationresiduals = committedvaluation(solution, scale, model)
Printf.@printf("Valuations: q=%.3e vd=%.3e vc=%.3e welfare=%.8f\n", valuationresiduals[1], valuationresiduals[2], valuationresiduals[3], valuations[4])
if maximum(abs, valuationresiduals) > 1e-3
    error("Discounted cash-flow errors exceed 1e-3 in scaled units; refine the solution before saving.")
end

## Save
filename = savecommitted(committedfilename(), solution, scale, model; valuations, valuationresiduals)
@info "Saved committed solution" filename
