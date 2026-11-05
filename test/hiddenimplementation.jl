using Test

import Roots

include("../src/primitives/constants.jl")
include("../src/primitives/signal.jl")
include("../src/primitives/climate.jl")
include("../src/agents/households.jl")
include("../src/agents/firm.jl")
include("../src/agents/government.jl")
include("../src/solve/model.jl")
include("../src/dynamics/government.jl")
include("../src/solve/government/noncommittedmarkov.jl")

testhousehold = Household(; ν = 1.0, ϕ = 1.0, r = 0.1)
testfirm = Firm(; e₀ = 1.0, a₀ = 0.1, A = 1.0, η = 1.0, κ = 1.0, ξ = 2.0)
testclimate = Climate(; γ = 0.02)
testmodel = Model(testhousehold, testfirm, testclimate)
testgovernment = Government(; κₚ = 0.5, b = 0.3)

@testset "Explicit welfare coefficients and constant installation price" begin
    @test c(testfirm) == c(testfirm.a₀, testfirm)
    @test k(0.2, testfirm) ≈ c(testfirm) * 0.2 + testfirm.ξ * 0.2^2 / 2
    @test α(0.5, testfirm) ≈ (0.5 - c(testfirm)) / testfirm.ξ
    @test α(0.5, testfirm) != α(0.5, 0.7, testfirm)
    for a in (testfirm.a₀, 0.55, testfirm.η), τ in (0.0, 0.2, 0.8)
        labour = n(τ, a, testhousehold, testfirm)
        resourcecost = k(0.2, testfirm) + v(labour, testhousehold) - y(labour, testfirm) * (1 - d(e(labour, a, testfirm), testclimate))
        @test simplifiedcost(τ, a, 0.2, testhousehold, testfirm, testclimate) ≈ resourcecost
    end
    @test κ₁(testfirm.η, testhousehold, testfirm, testclimate) == 0
    @test κ₂(testfirm.η, testhousehold, testfirm, testclimate) == 0
end

@testset "Public and government Markov moments and frozen holding time" begin
    signal = Signal(; ϵ = 1.0, σ = 0.5)
    grid = NonCommittedMarkovGrid(testmodel, 0.25)
    path = CommittedTaxPath([0.6, 0.6], [0.0, 1.0])
    parameters = NonCommittedMarkovParameters(testmodel, testgovernment, signal, path, grid, 2.0)
    @test physicalstate(grid, 1, 1) == (0.0, testfirm.a₀)
    @test physicalstate(grid, size(grid, 1), size(grid, 2)) == (1.0, testfirm.η)
    φ, a = physicalstate(grid, 3, 3)
    q, τ, τᶜ = 0.7, 0.2, 0.6
    transition = noncommittedtransition(φ, a, q, τ, τᶜ, parameters, 0.7)
    publictransition = noncommittedtransition(φ, a, q, τ, τᶜ, parameters, 0.7; firmmeasure = true)
    μᵩ, σᵩ = noncommittedbeliefmoments(τ, τᶜ, φ, signal)
    Δφ, Δa = grid.h * span(grid.φ), grid.h * span(grid.A)
    @test all(p -> p ≥ 0, (transition.pᵩ⁺, transition.pᵩ⁻, transition.pₐ⁺, transition.p₀))
    @test transition.pᵩ⁺ + transition.pᵩ⁻ + transition.pₐ⁺ + transition.p₀ ≈ 1
    @test (transition.pᵩ⁺ - transition.pᵩ⁻) * Δφ / transition.Δt ≈ μᵩ
    @test transition.pₐ⁺ * Δa / transition.Δt ≈ α(q, testfirm)
    @test publictransition.Δt == transition.Δt
    @test publictransition.pᵩ⁺ == publictransition.pᵩ⁻
    @test 2publictransition.pᵩ⁺ * Δφ^2 / publictransition.Δt ≈ σᵩ^2
    @test noncommittedcontinuation(fill(7.0, size(grid)), 3, 3, transition) ≈ 7
    B = φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * (τᶜ - τ)
    τᵉ = noncommittedexpectedtax(φ, τ, τᶜ)
    for τ̂ in range(0, τᶜ; length = 101)
        exitprobability = transition.Δt * (σᵩ^2 / Δφ^2 + abs(B * (τ̂ - τᵉ)) / Δφ + α(q, testfirm) / Δa)
        @test exitprobability ≤ 1 + 16eps(Float64)
    end
    for prior in (0.0, 1.0)
        boundary = noncommittedtransition(prior, a, q, τ, τᶜ, parameters, 0.7)
        @test boundary.pᵩ⁺ == boundary.pᵩ⁻ == 0
    end
    @test noncommittedtransition(φ, testfirm.η, q, τ, τᶜ, parameters, 0.7).pₐ⁺ == 0
end

@testset "Hidden policy minimises both frozen drift branches" begin
    signal = Signal(; ϵ = 1.0, σ = 0.5)
    grid = NonCommittedMarkovGrid(testmodel, 0.25)
    path = CommittedTaxPath([0.6, 0.6], [0.0, 1.0])
    parameters = NonCommittedMarkovParameters(testmodel, testgovernment, signal, path, grid, 0.2)
    i, j, Δt, τᶜ = 3, 3, 0.7, 0.6
    φ, a = physicalstate(grid, i, j)
    Δφ = grid.h * span(grid.φ)
    taxgrid = range(0, τᶜ; length = 12001)
    for (backward, forward) in ((-0.25, 0.25), (-2.0, -1.5), (1.0, -1.0))
        q, uᵍ, τ = fill(0.7, size(grid)), zeros(size(grid)), fill(0.2, size(grid))
        for column in axes(uᵍ, 2), row in axes(uᵍ, 1)
            distance = physicalstate(grid, row, column)[1] - φ
            uᵍ[row, column] = distance * (distance ≤ 0 ? backward : forward)
        end
        V = NonCommittedMarkovValue(0.0, grid, q, uᵍ, copy(uᵍ), τ)
        V′ = NonCommittedMarkovValue(Δt, grid, copy(q), 1.5 .* uᵍ .+ 0.1, copy(uᵍ), copy(τ))
        column = NonCommittedColumn(V, V′, parameters, j, τᶜ, Δt, 1.0, 1.0, 1.0, κ₀(a, testhousehold, testfirm, testclimate), κ₁(a, testhousehold, testfirm, testclimate), κ₂(a, testhousehold, testfirm, testclimate))
        x = collect(Iterators.flatten((q[row, j], uᵍ[row, j], τ[row, j]) for row in axes(q, 1)))
        transition = noncommittedtransition(φ, a, q[i, j], τ[i, j], τᶜ, parameters, Δt)
        candidate, improvement = noncommittedimprovement(x, i, column, transition)
        discount, flowweight, weight = noncommittedweights(transition, testhousehold.r, Δt)
        continuation = (1 - weight) .* uᵍ .+ weight .* V′.uᵍ
        τᵉ = noncommittedexpectedtax(φ, τ[i, j], τᶜ)
        B = φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * (τᶜ - τ[i, j])
        _, σᵩ = noncommittedbeliefmoments(τ[i, j], τᶜ, φ, signal)
        bellman = τ̂ -> begin
            hiddenμ = B * (τ̂ - τᵉ)
            pplus = transition.Δt * (σᵩ^2 / (2Δφ^2) + max(hiddenμ, 0) / Δφ)
            pminus = transition.Δt * (σᵩ^2 / (2Δφ^2) + max(-hiddenμ, 0) / Δφ)
            pstay = 1 - pplus - pminus - transition.pₐ⁺
            expectation = pstay * continuation[i, j] + pplus * continuation[i + 1, j] + pminus * continuation[i - 1, j] + transition.pₐ⁺ * continuation[i, j + 1]
            return discount * expectation + flowweight * simplifiedcost(τ̂, a, α(q[i, j], testfirm), testhousehold, testfirm, testclimate)
        end
        objectives = map(bellman, taxgrid)
        @test abs(candidate - taxgrid[argmin(objectives)]) ≤ 1.01step(taxgrid)
        @test bellman(candidate) ≤ minimum(objectives) + 1e-12
        @test improvement ≈ (bellman(τ[i, j]) - bellman(candidate)) / flowweight atol = 1e-10
    end
end

@testset "Zero announcement gives exact non-annualised welfare" begin
    path = CommittedTaxPath([0.0, 0.0], [0.0, 1.0])
    tolerance = 1e-10
    solution = solvenoncommitted(path, testmodel, testgovernment, Signal(); h = 0.5, timestep = 0.5, maxholdingtime = 0.25, maxiters = 20, tolerance, policytolerance = tolerance, verbose = 0)
    @test solution.t == [0.0, 0.5, 1.0]
    for V in solution.u
        @test maximum(abs, V.q) ≤ tolerance
        @test all(iszero, V.τ)
        @test V.uᵍ ≈ V.u
        for j in axes(V.u, 2), i in axes(V.u, 1)
            _, a = physicalstate(V.grid, i, j)
            @test V.u[i, j] ≈ κ₀(a, testhousehold, testfirm, testclimate) / testhousehold.r rtol = 1e-10
        end
    end
    @test solution(0.25).u ≈ first(solution.u).u
end

@testset "Uninformative signal isolates static policy and forward valuation" begin
    path = CommittedTaxPath([0.0, 0.4], [0.0, 1.0], 0.4)
    tolerance = 1e-8
    solution = solvenoncommitted(path, testmodel, testgovernment, Signal(; ϵ = 0.0, σ = 0.5); h = 0.5, timestep = 0.5, maxholdingtime = 0.2, maxiters = 100, tolerance, policytolerance = tolerance, verbose = 0)
    @test all(d -> max(d.valuegap, d.bellmangap, d.policygap) ≤ tolerance, solution.diagnostics)
    for V in solution.u
        @test all(isfinite, V.q) && all(isfinite, V.uᵍ) && all(isfinite, V.u)
        @test minimum(V.q) ≥ 0
        @test maximum(V.q[:, end]) ≤ c(testfirm) + tolerance
        for j in axes(V.τ, 2), i in axes(V.τ, 1)
            φ, a = physicalstate(V.grid, i, j)
            @test 0 ≤ V.τ[i, j] ≤ path(V.t)
            @test V.τ[i, j] ≈ noncommittedstatictax(φ, a, path(V.t), solution.parameters) atol = tolerance
        end
    end
    @test all(iszero, first(solution.u).τ)
    @test maximum(first(solution.u).q) > 0 # Future taxation gives investment value even at zero current tax.
    @test first(solution.u).q != last(solution.u).q
end

@testset "Supporting-price tail has the correct public cap valuation" begin
    output = y(n(0.0, testfirm.η, testhousehold, testfirm), testfirm)
    support = testhousehold.r * c(testfirm) / output
    path = CommittedTaxPath([support, support], [0.0, 1.0], support)
    solution = solvenoncommitted(path, testmodel, testgovernment, Signal(); h = 0.5, timestep = 0.5, maxholdingtime = 0.25, maxiters = 30, tolerance = 1e-10, policytolerance = 1e-10, verbose = 0)
    for V in solution.u
        φ = collect(first(unitcube(V.grid)))
        @test V.q[:, end] ≈ φ .* c(testfirm) rtol = 1e-9 atol = 1e-10
        @test all(iszero, V.τ[:, end])
        @test all(τ -> iszero(τ * e(n(τ, testfirm.η, testhousehold, testfirm), testfirm.η, testfirm)), (zero(support), support))
    end
end

@testset "Calendar jumps use one-sided policy interpolation" begin
    grid = NonCommittedMarkovGrid(testmodel, 0.5)
    path = CommittedTaxPath([0.6, 0.6, 0.1, 0.1], [0.0, 0.4, 0.4, 1.0], 0.1)
    parameters = NonCommittedMarkovParameters(testmodel, testgovernment, Signal(), path, grid, 0.25)
    values = [NonCommittedMarkovValue(t, grid, fill(q, size(grid)), fill(q + 1, size(grid)), fill(q + 2, size(grid)), fill(τ, size(grid))) for (t, q, τ) in ((0.0, 1.0, 0.3), (0.4, 2.0, 0.05), (1.0, 3.0, 0.04))]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), 3)
    solution = NonCommittedMarkovSolution([0.0, 0.4, 1.0], values, parameters, diagnostics)
    @test all(==(0.3), solution(0.3).τ)
    @test all(==(0.05), solution(0.4).τ)
    @test solution(0.3).q ≈ fill(1.75, size(grid))
    @test solution(0.3).u ≈ fill(3.75, size(grid))
    @test path(0.4; side = :left) == 0.6
    @test path(0.4) == 0.1

    support = testhousehold.r * c(testfirm) / y(n(0.0, testfirm.η, testhousehold, testfirm), testfirm)
    announcement = CommittedTaxPath([0.4, 0.4, support, support], [0.0, 0.6, 0.6, 1.0], support)
    solved = solvenoncommitted(announcement, testmodel, testgovernment, Signal(; ϵ = 0.0, σ = 0.5); h = 0.5, timestep = 0.5, maxholdingtime = 0.25, maxiters = 100, tolerance = 1e-8, policytolerance = 1e-8, verbose = 0)
    @test solved.t == [0.0, 0.5, 0.6, 1.0]
    @test count(==(0.6), solved.t) == 1
    @test solved(0.55).τ ≈ solved.u[2].τ
    @test maximum(solved(0.6).τ) ≤ support + 1e-8
    @test solved(0.6 - 1e-8).q ≈ solved(0.6).q atol = 1e-6 rtol = 1e-6
    @test solved(0.6 - 1e-8).u ≈ solved(0.6).u atol = 1e-6 rtol = 1e-6
    @test_throws DomainError solved(-0.01)
    @test_throws DomainError solved(1.01)
end
