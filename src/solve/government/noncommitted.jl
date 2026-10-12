## State space
struct NonCommittedGrid{Tᵠ, Tₘ, Tₐ}
    φgrid::Tᵠ
    mgrid::Tₘ
    agrid::Tₐ
end

function Base.size(grid::NonCommittedGrid)
    (length(grid.φgrid), length(grid.mgrid), length(grid.agrid))
end
function Base.length(grid::NonCommittedGrid)
    prod(size(grid))
end
function eltypes(grid::NonCommittedGrid)
    promote_type(eltype(grid.φgrid), eltype(grid.mgrid), eltype(grid.agrid))
end
function Base.CartesianIndices(grid::NonCommittedGrid)
    CartesianIndices((eachindex(grid.φgrid), eachindex(grid.mgrid), eachindex(grid.agrid)))
end

abstract type NonCommittedTaxMethod end

"Tax condition from an instantaneous hidden-action deviation"
struct OneShotTax <: NonCommittedTaxMethod end

"Tax condition obtained by controlling the complete posterior generator"
struct FullGeneratorTax{T} <: NonCommittedTaxMethod
    upperbound::T
end

struct NonCommittedScalingParameters{T}
    q::T
    W::T
    function NonCommittedScalingParameters(firm::Firm{T}, government::Government{T}) where T
        q = firm.r * c(firm.e₀, firm)
        W = government.y₀
    
        return new{T}(q, W)
    end
end

struct NonCommittedParameters{TC, T, H, F, G, S, C, TG, TS, TM <: NonCommittedTaxMethod}
    τᶜ::TC
    horizon::T
    household::H
    firm::F
    government::G
    signal::S
    climate::C
    grid::TG
    scaling::TS
    taxmethod::TM
end

function NonCommittedParameters(τᶜ, horizon, grid::NonCommittedGrid, household::Household, firm::Firm, government::Government, signal::Signal, climate::Climate, taxmethod::NonCommittedTaxMethod)
    scaling = NonCommittedScalingParameters(firm, government)
    return NonCommittedParameters(τᶜ, horizon, household, firm, government, signal, climate, grid, scaling, taxmethod)
end

function noncommittedviews(x, grid::NonCommittedGrid)
    n = length(grid)
    q = reshape(view(x, 1:n), size(grid))
    W = reshape(view(x, (n + 1):(2n)), size(grid))

    return q, W
end

## Policies
function noncommittedtaxcoefficient(∂ᵩW, φ, signal::Signal, ::OneShotTax)
    -φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * ∂ᵩW
end
function noncommittedtaxcoefficient(∂ᵩW, _, φ, signal::Signal, taxtype::OneShotTax)
    noncommittedtaxcoefficient(∂ᵩW, φ, signal, taxtype)
end
function noncommittedtaxcoefficient(∂ᵩW, ∂ᵩᵩW, φ, signal::Signal, ::FullGeneratorTax)
    φ^2 * (1 - φ) * (signal.ϵ / signal.σ)^2 * (-2∂ᵩW + (1 - φ) * ∂ᵩᵩW)
end

"Minimum-tax solution of the hidden-action condition"
function noncommittedtax(∂ᵩW, ∂ᵩᵩW, φ, τᶜ, signal::Signal, government::Government, taxmethod::OneShotTax)
    taxcoefficient = noncommittedtaxcoefficient(∂ᵩW, ∂ᵩᵩW, φ, signal, taxmethod)

    if taxcoefficient ≤ 0 || τᶜ ≤ 0
        return zero(taxcoefficient + τᶜ)
    end

    return taxcoefficient * τᶜ / (household.r * government.δ + taxcoefficient)
end

function fullgeneratorhamiltonian(τ, taxcoefficient, τᶜ, government::Government)
    household.r * government.δ * τ^2 / 2 +
        taxcoefficient * (τᶜ - τ)^2 / 2
end

"Global solution of the complete-generator tax problem"
function noncommittedtax(∂ᵩW, ∂ᵩᵩW, φ, τᶜ, signal::Signal, government::Government, taxmethod::FullGeneratorTax)
    taxcoefficient = noncommittedtaxcoefficient(∂ᵩW, ∂ᵩᵩW, φ, signal, taxmethod)
    
    curvature = household.r * government.δ + taxcoefficient
    zerotax = zero(taxcoefficient + τᶜ + taxmethod.upperbound)
    upperbound = zerotax + taxmethod.upperbound

    if curvature > 0
        τ = taxcoefficient * τᶜ / curvature

        return clamp(τ, zerotax, upperbound)
    end

    zerovalue = fullgeneratorhamiltonian(zerotax, taxcoefficient, τᶜ, government)
    uppervalue = fullgeneratorhamiltonian(upperbound, taxcoefficient, τᶜ, government)

    return uppervalue < zerovalue ? upperbound : zerotax
end

function noncommittedtax(∂ᵩW, φ, τᶜ, signal::Signal, government::Government)
    noncommittedtax(∂ᵩW, zero(∂ᵩW), φ, τᶜ, signal, government, OneShotTax())
end

function noncommittedtaxresidual(τ, ∂ᵩW, ∂ᵩᵩW, φ, τᶜ, signal::Signal, government::Government, taxmethod::NonCommittedTaxMethod)
    taxcoefficient = noncommittedtaxcoefficient(∂ᵩW, ∂ᵩᵩW, φ, signal, taxmethod)

    return household.r * government.δ * τ - taxcoefficient * (τᶜ - τ)
end


function noncommittedtaxviolation(τ, residual, ::OneShotTax)
    iszero(τ) ? max(-residual, zero(residual)) : abs(residual)
end

function noncommittedtaxviolation(τ, residual, method::FullGeneratorTax)
    if iszero(method.upperbound)
        return zero(residual)
    elseif iszero(τ)
        return max(-residual, zero(residual))
    elseif iszero(method.upperbound - τ)
        return max(residual, zero(residual))
    end

    return abs(residual)
end

function noncommittedtaxcomplementarity(τ, residual, ::OneShotTax)
    abs(τ * residual)
end

function noncommittedtaxcomplementarity(τ, residual, method::FullGeneratorTax)
    if iszero(method.upperbound)
        return zero(residual)
    end

    min(abs(τ * residual), abs((method.upperbound - τ) * residual))
end

function noncommittedtaxgap(τ, _, _, ::Government, ::OneShotTax)
    zero(τ)
end

function noncommittedtaxgap(τ, taxcoefficient, τᶜ, government::Government, method::FullGeneratorTax)
    zerotax = zero(τ + taxcoefficient + τᶜ + method.upperbound)
    upperbound = zerotax + method.upperbound

    chosenvalue = fullgeneratorhamiltonian(τ, taxcoefficient, τᶜ, government)
    minimumvalue = min(
        fullgeneratorhamiltonian(zerotax, taxcoefficient, τᶜ, government),
        fullgeneratorhamiltonian(upperbound, taxcoefficient, τᶜ, government)
    )

    curvature = household.r * government.δ + taxcoefficient

    if curvature > 0
        stationarytax = clamp(taxcoefficient * τᶜ / curvature, zerotax, upperbound)
        minimumvalue = min(
            minimumvalue,
            fullgeneratorhamiltonian(stationarytax, taxcoefficient, τᶜ, government)
        )
    end

    return max(chosenvalue - minimumvalue, zero(chosenvalue))
end

function noncommittedtaxresidual(τ, ∂ᵩW, φ, τᶜ, signal::Signal, government::Government)
    noncommittedtaxresidual(τ, ∂ᵩW, zero(∂ᵩW), φ, τᶜ, signal, government, OneShotTax())
end

function firminvestmentgap(q, a, u, firm::Firm)
    q - firm.r * (c(a, firm) + firm.ξ * u)
end

## Finite differences
@inline function forwardmderivative(x, i, j, k, grid::NonCommittedGrid)
    # The upper boundary should be placed beyond the states reachable before T.
    if j == length(grid.mgrid)
        return zero(eltype(x))
    end

    return (x[i, j + 1, k] - x[i, j, k]) / (grid.mgrid[j + 1] - grid.mgrid[j])
end

@inline function forwardaderivative(x, i, j, k, grid::NonCommittedGrid)
    if k == length(grid.agrid)
        return zero(eltype(x))
    end

    return (x[i, j, k + 1] - x[i, j, k]) / (grid.agrid[k + 1] - grid.agrid[k])
end

@inline function backwardφderivative(x, i, j, k, grid::NonCommittedGrid)
    if i == 1
        return zero(eltype(x))
    end

    return (x[i, j, k] - x[i - 1, j, k]) / (grid.φgrid[i] - grid.φgrid[i - 1])
end

@inline function centralφsecondderivative(x, i, j, k, grid::NonCommittedGrid)
    if i == 1 || i == length(grid.φgrid)
        return zero(eltype(x))
    end

    leftstep = grid.φgrid[i] - grid.φgrid[i - 1]
    rightstep = grid.φgrid[i + 1] - grid.φgrid[i]
    leftderivative = (x[i, j, k] - x[i - 1, j, k]) / leftstep
    rightderivative = (x[i + 1, j, k] - x[i, j, k]) / rightstep

    return 2 * (rightderivative - leftderivative) / (leftstep + rightstep)
end

## Terminal values
"Terminal state after the committed tax tail has become negligible"
function noncommittedterminalstate(parameters::NonCommittedParameters)
    @unpack firm, government, climate, grid, scaling = parameters

    x = zeros(eltype(parameters.τᶜ), 2length(grid))
    _, W = noncommittedviews(x, grid)

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid), i in eachindex(grid.φgrid)
        a = grid.agrid[k]
        m = grid.mgrid[j]
        W[i, j, k] = V₃damages(a, m, firm, government, climate) / scaling.W
    end

    return x
end

## Coupled firm-government system
function noncommittedcalendar(s, parameters::NonCommittedParameters)
    (1 - s) * parameters.horizon
end

function noncommittedreversetime(t, parameters::NonCommittedParameters)
    noncommittedreversetime(t, parameters.horizon)
end
function noncommittedreversetime(t, horizon)
    1 - t / horizon
end

"Coupled firm and government system in reverse time"
function noncommittedreversedrift!(dx::TX, x, parameters::NonCommittedParameters, s) where {T, TX <: AbstractVecOrMat{T}}
    @unpack τᶜ, horizon, firm, government, signal, climate, grid, scaling, taxmethod = parameters

    qnormalised, Wnormalised = noncommittedviews(x, grid)
    dqnormalised, dWnormalised = noncommittedviews(dx, grid)

    calendartime = noncommittedcalendar(s, parameters)
    τᶜₜ  = τᶜ(calendartime)

    @inbounds Threads.@threads for idx in CartesianIndices(grid)
        i, j, k = idx.I

        φ = grid.φgrid[i]
        m = grid.mgrid[j]
        a = grid.agrid[k]

        if iszero(φ)
            dqnormalised[idx] = zero(T)
            dWnormalised[idx] = zero(T)
            continue
        end

        q = scaling.q * qnormalised[idx]
        W = scaling.W * Wnormalised[idx]

        ∂ₘq = scaling.q * forwardmderivative(qnormalised, i, j, k, grid)
        ∂ₐq = scaling.q * forwardaderivative(qnormalised, i, j, k, grid)
        ∂ᵩᵩq = scaling.q * centralφsecondderivative(qnormalised, i, j, k, grid)

        ∂ₘW = scaling.W * forwardmderivative(Wnormalised, i, j, k, grid)
        ∂ₐW = scaling.W * forwardaderivative(Wnormalised, i, j, k, grid)
        ∂ᵩW = scaling.W * backwardφderivative(Wnormalised, i, j, k, grid)
        ∂ᵩᵩW = scaling.W * centralφsecondderivative(Wnormalised, i, j, k, grid)

        u = investmentpolicy(q, a, firm)
        τ = noncommittedtax(∂ᵩW, ∂ᵩᵩW, φ, τᶜₜ , signal, government, taxmethod)
        τᵉ = expectedtax(φ, τ, τᶜₜ )

        signaltonoise = χ(τ, τᶜₜ , signal)
        bᵩ = beliefdrift(signaltonoise, φ)
        σᵩ = beliefdiffusion(signaltonoise, φ)

        dq = firmmarginalvaluedrift(
            q, a, u, τᵉ, ∂ₘq, ∂ₐq, ∂ᵩᵩq, σᵩ, firm
        )
        dW = governmentvaluedrift(
            W, a, m, u, τ, ∂ₘW, ∂ₐW, ∂ᵩW, ∂ᵩᵩW, bᵩ, σᵩ,
            firm, government, climate
        )

        dqnormalised[idx] = horizon * dq / scaling.q
        dWnormalised[idx] = iszero(cumulativeemissionsdrift(a, firm)) ? zero(T) : horizon * dW / scaling.W
    end

    return dx
end

function noncommittedjacobianprototype(grid::NonCommittedGrid)
    nᵠ, nₘ, nₐ = size(grid)
    n = length(grid)
    I = Int[]
    J = Int[]

    sizehint!(I, 14n)
    sizehint!(J, 14n)

    @inbounds for k in 1:nₐ, j in 1:nₘ, i in 1:nᵠ
        node = i + (j - 1) * nᵠ + (k - 1) * nᵠ * nₘ
        statecolumns = Int[node]
        taxcolumns = Int[node]

        i > 1 && push!(statecolumns, node - 1)
        i < nᵠ && push!(statecolumns, node + 1)
        j < nₘ && push!(statecolumns, node + nᵠ)
        k < nₐ && push!(statecolumns, node + nᵠ * nₘ)

        i > 1 && push!(taxcolumns, node - 1)
        i < nᵠ && push!(taxcolumns, node + 1)

        for column in statecolumns
            push!(I, node)
            push!(J, column)
            push!(I, n + node)
            push!(J, n + column)
        end

        for column in taxcolumns
            push!(I, node)
            push!(J, n + column)
        end

        push!(I, n + node)
        push!(J, node)
    end

    T = eltypes(grid)

    return SparseArrays.sparse(I, J, ones(T, length(I)), 2n, 2n)
end

function noncommittedproblem(parameters::NonCommittedParameters)
    x₀ = noncommittedterminalstate(parameters)
    jacobian = noncommittedjacobianprototype(parameters.grid)
    drift = SciMLBase.ODEFunction(noncommittedreversedrift!; jac_prototype = jacobian)

    return SciMLBase.ODEProblem(drift, x₀, (0., 1.), parameters)
end

## Diagnostics
function noncommittedpolicies(x, parameters::NonCommittedParameters, s)
    @unpack τᶜ, firm, government, signal, grid, scaling, taxmethod = parameters
    qnormalised, Wnormalised = noncommittedviews(x, grid)

    investment = similar(qnormalised)
    tax = similar(qnormalised)
    expectedtax = similar(qnormalised)
    taxcoefficient = similar(qnormalised)
    taxcurvature = similar(qnormalised)
    beliefvalue = similar(qnormalised)

    calendartime = noncommittedcalendar(s, parameters)
    τᶜₜ = τᶜ(calendartime)

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid), i in eachindex(grid.φgrid)
        φ = grid.φgrid[i]
        a = grid.agrid[k]

        q = scaling.q * qnormalised[i, j, k]
        
        ∂ᵩW = scaling.W * backwardφderivative(Wnormalised, i, j, k, grid)
        ∂ᵩᵩW = scaling.W * centralφsecondderivative(Wnormalised, i, j, k, grid)

        beliefvalue[i, j, k] = -φ * (1 - φ) * ∂ᵩW
        taxcoefficient[i, j, k] = noncommittedtaxcoefficient(∂ᵩW, ∂ᵩᵩW, φ, signal, taxmethod)
        taxcurvature[i, j, k] = household.r * government.δ + taxcoefficient[i, j, k]
        investment[i, j, k] = investmentpolicy(q, a, firm)
        tax[i, j, k] = noncommittedtax(∂ᵩW, ∂ᵩᵩW, φ, τᶜₜ, signal, government, taxmethod)
        expectedtax[i, j, k] = expectedtax(φ, tax[i, j, k], τᶜₜ)
    end

    return (; investment, tax, expectedtax, taxcoefficient, taxcurvature, beliefvalue)
end

function noncommittedpoliciesattime(solution, parameters::NonCommittedParameters, t)
    s = noncommittedreversetime(t, parameters)

    return noncommittedpolicies(solution(s), parameters, s)
end

function noncommittedvalues(x, parameters::NonCommittedParameters)
    qnormalised, Wnormalised = noncommittedviews(x, parameters.grid)

    return (
        q = parameters.scaling.q .* qnormalised,
        W = parameters.scaling.W .* Wnormalised,
    )
end

function noncommittedtime(solution, parameters::NonCommittedParameters)
    map(Base.Fix2(noncommittedcalendar, parameters), solution.t)
end

function noncommittedresiduals(x, parameters::NonCommittedParameters, s)
    @unpack τᶜ, firm, government, signal, grid, scaling, taxmethod = parameters
    qnormalised, Wnormalised = noncommittedviews(x, grid)
    policies = noncommittedpolicies(x, parameters, s)

    investmentgap = similar(qnormalised)
    taxresidual = similar(qnormalised)
    τᶜₜ = τᶜ(noncommittedcalendar(s, parameters))

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        φ = grid.φgrid[i]
        a = grid.agrid[k]
        q = scaling.q * qnormalised[i, j, k]
        ∂ᵩW = scaling.W * backwardφderivative(Wnormalised, i, j, k, grid)
        ∂ᵩᵩW = scaling.W * centralφsecondderivative(Wnormalised, i, j, k, grid)

        investmentgap[i, j, k] = firminvestmentgap(q, a, policies.investment[i, j, k], firm)
        taxresidual[i, j, k] = noncommittedtaxresidual(policies.tax[i, j, k], ∂ᵩW, ∂ᵩᵩW,φ, τᶜₜ, signal, government, taxmethod)
    end

    return (; investmentgap, taxresidual)
end

function noncommittedkktdiagnostics(x::TX, parameters::NonCommittedParameters, s) where {T, TX <: AbstractVecOrMat{T}}
    @unpack firm, grid, taxmethod = parameters
    policies = noncommittedpolicies(x, parameters, s)
    residuals = noncommittedresiduals(x, parameters, s)

    firmviolation = zero(T)
    taxviolation = zero(T)
    complementarity = zero(T)
    taxhamiltoniangap = zero(T)
    committedtax = parameters.τᶜ(noncommittedcalendar(s, parameters))

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        a = grid.agrid[k]
        u = policies.investment[i, j, k]
        τ = policies.tax[i, j, k]
        investmentgap = residuals.investmentgap[i, j, k]
        taxresidual = residuals.taxresidual[i, j, k]
        taxcoefficient = policies.taxcoefficient[i, j, k]

        if !iszero(cumulativeemissionsdrift(a, firm))
            firmviolation = max(firmviolation, investmentgap)
            complementarity = max(
                complementarity, abs(u * investmentgap)
            )
        end

        taxviolation = max(taxviolation, noncommittedtaxviolation(τ, taxresidual, taxmethod))
        complementarity = max(complementarity, noncommittedtaxcomplementarity(τ, taxresidual, taxmethod))
        taxhamiltoniangap = max(
            taxhamiltoniangap,
            noncommittedtaxgap(τ, taxcoefficient, committedtax, parameters.government, taxmethod)
        )
    end

    return (;
        firmviolation, taxviolation, complementarity, taxhamiltoniangap
    )
end
