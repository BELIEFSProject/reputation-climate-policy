## Setup
import Roots
import Printf
import StaticArrays

import ADTypes
import SciMLBase
import SciMLLogging
import BoundaryValueDiffEq as BVD
import NonlinearSolve
import QuadGK
import Roots

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

includet("../src/solve/committed/allocation.jl")
includet("../src/solve/committed/policy.jl")
includet("../src/solve/committed/tail.jl")
includet("../src/solve/committed/solve.jl")

calibration = Calibration()
model = GeneralEquilibriumModel(calibration)
xᶜ₀ = 0.
# Set solverlogging=true for detailed nonlinear iterations; loginterval controls progress frequency.
options = CommittedOptions(; verbose=true, solverlogging=false, loginterval=5.0, nodes = 20, tolerance = 1e-3)

## Solve
# Known committed type (φ₀ = 1); policy is selected from the state–costate Hamiltonian.
committed = solvecommitted(model, xᶜ₀; options)
path = committed.path
diagnostics = committed.diagnostics

## Diagnostics
initialstate = committedstate(path.solution(0.0), path.problem)
terminalstate = committedstate(path.solution(1.0), path.problem)
initial = committedallocation(path, 0.0)
tailpolicy = committedtail(path)
terminal = tailpolicy.allocation
Printf.@printf("Committed canonical candidate: retcode=%s, discounted welfare cost=%.8f\n", string(path.solution.retcode), path.cost)
Printf.@printf("Initial allocation: xc=%.6f q=%.6f vd=%.6f p=%.6f xd=%.6f investment=%.6f\n", initial.xᶜ, initialstate.q, initialstate.vᵈ, initial.p, initial.xᵈ, initial.investment)
Printf.@printf("Initial costates: lambda_xc=%.6f lambda_q=%.3e lambda_vd=%.3e\n", initialstate.λxᶜ, initialstate.λq, initialstate.λvᵈ)
Printf.@printf("Stationary tail: tax=%.6f xc=%.6f q=%.6f vd=%.6f p=%.6f xd=%.6f investment=%.3e\n", tailpolicy.tax, terminalstate.xᶜ, terminalstate.q, terminalstate.vᵈ, terminal.p, terminal.xᵈ, diagnostics.tailinvestment)
Printf.@printf("Hamiltonian checks: policy KKT=%.3e tail KKT=%.3e boundary gap=%.3e tail boundary gap=%.3e singular points=%d investment-boundary points=%d\n", diagnostics.hamiltoniankkt, diagnostics.tailhamiltoniankkt, diagnostics.boundarygap, diagnostics.tailboundarygap, diagnostics.singularpoints, diagnostics.investmentboundarypoints)
Printf.@printf("Scaled canonical defect per year=%.3e\n", diagnostics.canonical)
Printf.@printf("Endpoint checks: boundary=%.3e initial complementarity=%.3e terminal complementarity=%.3e tax jump=%.3e price jump=%.3e\n", diagnostics.boundary, diagnostics.initialcomplementarity, diagnostics.tailcomplementarity, diagnostics.taxjump, diagnostics.pricejump)
Printf.@printf("Residuals: market=%.3e credit=%.3e credit complementarity=%.3e investment KKT=%.3e dirty KKT=%.3e irreversibility=%.3e\n", diagnostics.market, diagnostics.credit, diagnostics.complementarity, diagnostics.investmentkkt, diagnostics.dirtykkt, diagnostics.irreversibility)
Printf.@printf("Valuation checks: clean PV=%.3e dirty PV=%.3e interior profit identity=%.3e minimum q=%.3e minimum vd=%.3e quadrature error=%.3e\n", diagnostics.cleanvaluation, diagnostics.dirtyvaluation, diagnostics.interiorprofit, diagnostics.minimumq, diagnostics.minimumvᵈ, path.quadratureerror)
