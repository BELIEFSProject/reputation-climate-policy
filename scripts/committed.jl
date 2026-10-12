## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings
using Plots

import JLD2

import Base.Threads
import FastClosures: @closure
import UnPack: @unpack, @pack!

# Linear algebra
import LinearAlgebra as LA
import SparseArrays
import StaticArrays as SA
import StaticArraysCore

# Interpolation and integration
import SciMLBase
import SpecialFunctions
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqCore as ODECore
import OrdinaryDiffEqRosenbrock as ODERosenbrock
import BoundaryValueDiffEq as BVP
import QuadGK
import SpecialFunctions as SF

# Optimization
import Optimization, Optim
import FiniteDiff
import Roots
import BandedMatrices: BandError

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/dynamics/state.jl")
includet("../src/dynamics/firm.jl")
includet("../src/dynamics/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")
includet("../src/utils/integrals.jl")

includet("../src/solve/model.jl")
includet("../src/solve/government/committed.jl")

includet("plotting/utils.jl")

const SIMPATH = joinpath("data", "solutions")
ispath(SIMPATH) || mkpath(SIMPATH);

## Defaults
household, firm, signal, climate = initmodels()
model = ModelParameters(household, firm, climate)

filename = joinpath(SIMPATH, solutionfilename(household, firm, climate))

if isfile(filename)
    throw("Committed solution in $filename already saved! Breaking to avoid overwriting.")
end

## Solver
x₀ = initialguess(SA.MVector, model)
odefn = ODE.ODEFunction{true}(driftcommitted!)
odeproblem = ODE.ODEProblem(ODE.ODEFunction{true}(driftcommitted!), x₀, (0., 1.), model)
lb, ub = bvpbounds(model)
bcresid_prototype = (initialcondition(x₀, model), terminalcondition(x₀, model))

bvpfunction = SciMLBase.BVPFunction(driftcommitted!, (initialcondition!, terminalcondition!); bcresid_prototype, twopoint = Val(true))
horizons = (5., 10., 20., 50., 100.)
solutionguess = ODE.solve(odeproblem, ODE.Rosenbrock23(); tspan = (0., horizons[1]))

for T in horizons
    bvproblem = BVP.TwoPointBVProblem(bvpfunction, solutionguess, (0., T), model; lb, ub)
    candidate = BVP.solve(bvproblem, BVP.MIRK4(); dt = T * 1e-2)
    SciMLBase.successful_retcode(candidate) || error("Committed BVP failed at T = $T with return code $(candidate.retcode).")

    if !isnothing(bvpsol)
        commondomain = range(0., bvpsol.t[end], length = 101)
        horizonerror = maximum(LA.norm(candidate(t) - bvpsol(t), Inf) for t in commondomain)
        @printf "Horizon %5.1f: maximum change on [0, %5.1f] = %.3e\n" T bvpsol.t[end] horizonerror
    end

    bvpsol = candidate
end

optimalpath = bvpsol
T = last(horizons)
taxpath = map(Base.Fix2(committedtax, model), optimalpath.u)
ts = optimalpath.t

initialresidual = initialcondition(first(optimalpath.u), model)
terminalresidual = terminalcondition(last(optimalpath.u), model)
boundaryerror = max(LA.norm(initialresidual, Inf), LA.norm(terminalresidual, Inf))
@printf "Maximum boundary residual: %.3e\n" boundaryerror

## Illustrate solution
if isinteractive()
    Plots.default(linewidth = 5)
    abatementpath = getindex.(optimalpath.u, 1)
    marginalvaluepath = getindex.(optimalpath.u, 2)

    taxfig = Plots.plot(ts, taxpath ./ taxfactor; c = :black, xlabel = "Year", ylabel = L"Policy [USD / tCO$_2$]", xlims = (0, T), margins = 6Plots.mm, label = L"\tau^c")

    labourpath = map((τ, a) -> n(τ, a, household, firm), taxpath, abatementpath)
    outputpath = firm.A .* labourpath

    marginalvaluefig = Plots.plot(ts, marginalvaluepath; c = :darkred, ylabel = L"Marginal value $q$", xlims = (0, T), margins = 6Plots.mm, label = L"q")

    abatementfig = Plots.plot(ts, abatementpath; c = :darkgreen, ylabel = L"Emissions-intensity reduction", xlims = (0, T), margins = 6Plots.mm, label = L"a")

    outputfig = Plots.plot(ts, outputpath; c = :darkblue, xlabel = "Year", ylabel = "Output [trillion USD / year]", xlims = (0, T), margins = 6Plots.mm, label = L"y_t")

    solutionfig = Plots.plot(taxfig, marginalvaluefig, abatementfig, outputfig; layout = (2, 2), size = (1000., 800.))

end

## Save solution
## Save 
committedtaxpath = CommittedTaxPath(taxpath, ts)

JLD2.jldopen(filename, "w") do file
    @pack! file = committedtaxpath, model
end

@printf "Saved committed model outcome in %s\n" filename
