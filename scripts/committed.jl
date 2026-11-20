## Setup
import Roots
import Printf
import StaticArrays

import ADTypes
import SciMLBase
import BoundaryValueDiffEq
import QuadGK
import Optim

include("../src/primitives/calibration.jl")
include("../src/primitives/signal.jl")
include("../src/primitives/climate.jl")
include("../src/agents/firm.jl")
include("../src/agents/households.jl")
include("../src/agents/energy.jl")
include("../src/agents/banking.jl")
include("../src/agents/government.jl")
include("../src/solve/equilibrium.jl")
include("../src/solve/model.jl")
include("../src/solve/committedallocation.jl")
include("../src/solve/committed.jl")

calibration = Calibration()
model = GeneralEquilibriumModel(calibration)
xc0 = 0.40
options = CommittedOptions()

## Solve
# Known committed type (φ₀ = 1); concentrate policy nodes near the start of the transition.
taxgrid = collect(range(0, 1; length=5)) .^ 2
initialtax = fill(0.15, length(taxgrid))
committed = solvecommitted(model, xc0, taxgrid, initialtax; options)
path = committed.path
diagnostics = committed.diagnostics

## Diagnostics
initial = committedallocation(path, 0.0)
terminal = committedallocation(path, 1.0)
Printf.@printf("Committed schedule candidate: converged=%s, discounted welfare cost=%.8f, schedule KKT=%.3e\n", string(Optim.converged(committed.optimisation)), path.cost, diagnostics.schedulekkt)
Printf.@printf("Initial allocation: xc=%.6f q=%.6f vd=%.6f p=%.6f xd=%.6f investment=%.6f\n", initial.xc, initial.q, initial.vd, initial.p, initial.xd, initial.investment)
Printf.@printf("Terminal allocation: xc=%.6f q=%.6f vd=%.6f p=%.6f xd=%.6f investment=%.3e\n", terminal.xc, terminal.q, terminal.vd, terminal.p, terminal.xd, diagnostics.tailinvestment)
Printf.@printf("Terminal drift checks: price=%.3e profit=%.3e; quadrature error=%.3e\n", diagnostics.tailpricedrift, diagnostics.tailprofitdrift, path.quadratureerror)
Printf.@printf("Residuals: market=%.3e credit=%.3e complementarity=%.3e investment KKT=%.3e irreversibility=%.3e boundary=%.3e clean PV=%.3e dirty PV=%.3e\n", diagnostics.market, diagnostics.credit, diagnostics.complementarity, diagnostics.investmentkkt, diagnostics.irreversibility, diagnostics.boundary, diagnostics.cleanvaluation, diagnostics.dirtyvaluation)
Printf.@printf("Dirty-sector checks: utilisation KKT=%.3e interior profit identity=%.3e; minimum q=%.3e minimum vd=%.3e\n", diagnostics.dirtykkt, diagnostics.interiorprofit, diagnostics.minimumq, diagnostics.minimumvd)
display(hcat(options.horizon .* path.problem.tax.grid, path.problem.tax.values))
