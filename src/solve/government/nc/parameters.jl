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
    grid = NonCommittedMarkovGrid(model, h)
    output = household isa LinearHousehold ? firm.A^2 / household.ν : y(n(0, firm.a₀, household, firm), firm)
    maxtax = max(maximum(τᶜ.taxpath), τᶜ.tailtax)
    qscale = max(1, c(firm), output * maxtax / household.r)
    valuescale = max(1, output / household.r)
    taxscale = max(maxtax, sqrt(eps(typeof(grid.h))))
    scales = promote(float(horizon), timestep, qscale, valuescale, taxscale, tolerance, policytolerance)
    return NCDAEParameters(model, government, signal, τᶜ, grid, scales..., maxiters)
end

