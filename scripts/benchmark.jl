## Setup
import Roots
import Printf
import StaticArrays as SA

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
includet("../src/solve/committed/equilibrium.jl")
includet("../src/solve/committed/dynamics.jl")

calibration = Calibration()
model = GeneralEquilibriumModel(calibration)
xᶜ₀ = 0.4

