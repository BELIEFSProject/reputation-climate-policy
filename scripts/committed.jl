## Setup
import Roots
import Optim
import Printf

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

struct CommittedAllocation{T}
    z::T
    n::T
    ω::T
    p::T
    x::T
    xd::T
    xc::T
    y::T
    L::T
    barL::T
    μ::T
    vd::T
    q::T
    xc_dot::T
    welfare::T
end

function committedallocation(model, xc, vd, tax)
    firm = model.firm
    firm isa CESFirm || error("The initial committed solver supports CESFirm only.")
    household = model.household
    dirty = model.dirty
    clean = model.clean
    barL = lendinglimit(vd, model.bank)
    n(z) = labour(z, barL, household, firm, model.bank.δL, household.r)
    residual(z) = z * n(z) - xc - dirtysupply(z, n(z), tax, household, firm, dirty)
    z = Roots.find_zero(residual, (0.05, 5.0), Roots.Brent())
    labourlevel = n(z)
    wagelevel = wage(z, labourlevel, household)
    price = energyprice(z, labourlevel, household, firm)
    dirtylevel = dirtyoutput(price, tax, dirty)
    demand = energydemand(z, labourlevel)
    output = y(labourlevel, z, firm)
    liquiditylevel = liquidity(z, labourlevel, household, firm, model.bank.δL)
    shadow = max(shadowcost(z, labourlevel, household, firm, model.bank.δL, household.r), zero(tax))
    marginalvalue = price / household.r
    investment = cleaninvestment(marginalvalue, clean)
    welfare = welfarecost(labourlevel, dirtylevel, investment, xc, output, household, dirty, clean, model.υ, model.climate.γ)
    CommittedAllocation(z, labourlevel, wagelevel, price, demand, dirtylevel, xc, output, liquiditylevel, barL, shadow, vd, marginalvalue, investment, welfare)
end

function stationarydirtyvalue(model, xc, tax)
    residual(vd) = begin
        allocation = committedallocation(model, xc, vd, tax)
        vd - dirtyvalue(allocation.p, tax, allocation.xd, model.household.r, model.dirty)
    end
    vd = Roots.find_zero(residual, (0.0, 5.0), Roots.Brent())
    allocation = committedallocation(model, xc, vd, tax)
    allocation
end

calibration = Calibration()
model = GeneralEquilibriumModel(calibration)
xc0 = 0.40

reference = stationarydirtyvalue(model, xc0, 0.15)
Printf.@printf("Reference allocation: z=%.6f n=%.6f p=%.6f xd=%.6f L=%.6f barL=%.6f mu=%.6f vd=%.6f q=%.6f investment=%.6f welfare=%.6f\n", reference.z, reference.n, reference.p, reference.xd, reference.L, reference.barL, reference.μ, reference.vd, reference.q, reference.xc_dot, reference.welfare)

objective(tax) = begin
    allocation = stationarydirtyvalue(model, xc0, tax[1])
    allocation.welfare / model.household.r
end

result = Optim.optimize(objective, [0.0], [1.0 / model.dirty.η], [0.15], Optim.Fminbox(Optim.BFGS()))
optimalallocation = stationarydirtyvalue(model, xc0, Optim.minimizer(result)[1])
optimaltax = Optim.minimizer(result)[1]

if !Optim.converged(result)
    error("Committed constant-policy optimisation did not converge.")
end

Printf.@printf("Committed constant-policy tax=%.6f z=%.6f n=%.6f p=%.6f xd=%.6f vd=%.6f q=%.6f investment=%.6f discounted-cost=%.6f\n", optimaltax, optimalallocation.z, optimalallocation.n, optimalallocation.p, optimalallocation.xd, optimalallocation.vd, optimalallocation.q, optimalallocation.xc_dot, Optim.minimum(result))
