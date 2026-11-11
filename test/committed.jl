using Test

import UnPack: @unpack
import BoundaryValueDiffEq as BVP
import SciMLBase
import ForwardDiff as FD
import QuadGK as QGK

include("../src/primitives/constants.jl")
include("../src/primitives/signal.jl")
include("../src/primitives/climate.jl")
include("../src/agents/households.jl")
include("../src/agents/firm.jl")
include("../src/agents/government.jl")
include("../src/solve/model.jl")
include("../src/dynamics/government.jl")
include("../src/solve/government/committed.jl")
include("../src/utils/saving.jl")

testhousehold = Household(; ν = 1.0, ϕ = 1.0, r = 0.1)
testfirm = Firm(; e₀ = 1.0, a₀ = 0.1, A = 1.0, η = 1.0, κ = 1.0, ξ = 2.0)
testmodel = Model(testhousehold, testfirm, Climate(; γ = 0.02))
testproblem = committedproblem(testmodel)

@testset "Full-interval singular BVP" begin
    @test testproblem isa SciMLBase.BVProblem
    @test size(testproblem.singular_term) == (5, 5)
    @test testproblem.singular_term[4, :] == [0, 0, 1, -1, 0]
    @test count(!iszero, testproblem.singular_term) == 2
    left = testproblem.u0(testmodel, 0.0)
    right = testproblem.u0(testmodel, 1.0)
    @test testproblem.singular_term * left ≈ zeros(5)
    residual = zeros(5)
    committedboundary!(residual, [left, right], testmodel, [0.0, 1.0])
    @test maximum(abs, residual) < 1e-12
end

@testset "Inherited initial policy" begin
    initialtax = 0.01
    problem = committedproblem(testmodel; initialtax)
    states = [problem.u0(testmodel, s) for s in (0.0, 1.0)]
    residual = zeros(5)
    problem.f.bc(residual, states, testmodel, [0.0, 1.0])
    @test maximum(abs, residual) < 1e-12
    @test problem.singular_term * first(states) ≈ zeros(5)

    λscale = testmodel.climate.γ * testfirm.A^2 / testhousehold.ν * (testfirm.η - testfirm.a₀)^2
    λinitial = λscale * last(states)[4]
    @test committedtax(testfirm.a₀, λinitial, testmodel) ≈ initialtax
    @test !iszero(λinitial)
    @test committedsolutionfilename(testmodel; initialtax) != committedsolutionfilename(testmodel)
end

@testset "Analytical abatement derivative matches the model primitives" begin
    for a in (0.1, 0.5, 0.99), τ in (0.0, 0.3, 0.8)
        reference = FD.derivative(z -> simplifiedcost(τ, z, 0.2, testhousehold, testfirm, testmodel.climate), a)
        @test ∂ₐsimplifiedcost(τ, a, 0.2, testhousehold, testfirm, testmodel.climate) ≈ reference
    end
end

@testset "Boundary callbacks preserve Dual-valued endpoint vectors" begin
    left = testproblem.u0(testmodel, 0.0)
    right = testproblem.u0(testmodel, 1.0)
    boundary = function (z)
        points = [view(z, 1:5), view(z, 6:10)]
        states = BVP.BoundaryValueDiffEqCore.EvalSol(points, [0.0, 1.0], nothing)
        residual = similar(z, 5)
        committedboundary!(residual, states, testmodel, states.t)
        return residual
    end
    endpoints = vcat(left, right)
    @test maximum(abs, boundary(endpoints)) < 1e-12
    expected = zeros(eltype(endpoints), 5, 10)
    expected[1, 1] = expected[2, 2] = expected[3, 4] = expected[4, 6] = expected[5, 9] = 1
    expected[3, 3] = -1
    @test FD.jacobian(boundary, endpoints) ≈ expected
    for s in (0.0, 0.5)
        field = function (x)
            dx = similar(x)
            committeddrift!(dx, x, testmodel, s)
            return dx
        end
        @test all(isfinite, FD.jacobian(field, testproblem.u0(testmodel, s)))
    end
end

@testset "Transformed equations reproduce the state and costate system" begin
    x, s = [0.4, 0.8, -0.1, -0.02, 0.1], 0.5
    dx = similar(x)
    committeddrift!(dx, x, testmodel, s)
    dx .+= testproblem.singular_term * x / s

    ascale = testfirm.η - testfirm.a₀
    tscale = sqrt(2testfirm.ξ * ascale / (testmodel.climate.γ * testfirm.A^2))
    qscale = testfirm.ξ * ascale / tscale
    λscale = testmodel.climate.γ * testfirm.A^2 / testhousehold.ν * ascale^2
    duration = tscale * exp(x[5])
    a = testfirm.η - ascale * x[1]
    q = c(testfirm) + qscale * x[2]
    λq = s * λscale * x[4]
    λₐ = testfirm.ξ * λscale * x[3] / duration - q
    τ = committedtax(a, λq, testmodel)
    source = y(n(τ, a, testhousehold, testfirm), testfirm) * τ
    sourcea = testfirm.A * τ * ∂ₐn(τ, a, testhousehold, testfirm)
    costa = FD.derivative(z -> simplifiedcost(τ, z, 0.0, testhousehold, testfirm, testmodel.climate), a)
    qdot = -qscale * dx[2] / duration
    Pdot = -testfirm.ξ * λscale * dx[3] / duration^2
    @test ascale * dx[1] / duration ≈ (q - c(testfirm)) / testfirm.ξ
    @test qdot ≈ testhousehold.r * q - source
    @test -λscale * (x[4] + s * dx[4]) / duration ≈ -(q + λₐ) / testfirm.ξ
    @test Pdot - qdot ≈ testhousehold.r * λₐ - costa + λq * sourcea
    @test dx[5] == 0
end

@testset "Tax minimises the bounded Hamiltonian below full abatement" begin
    for a in (0.2, 0.6, 0.99), λq in (-0.1, -0.001, 0.0, 0.1)
        tax = committedtax(a, λq, testmodel)
        objective(τ) = simplifiedcost(τ, a, 0.0, testhousehold, testfirm, testmodel.climate) - λq * y(n(τ, a, testhousehold, testfirm), testfirm) * τ
        taxes = range(0.0, inv(testfirm.η); length = 2001)
        @test 0 ≤ tax ≤ inv(testfirm.η)
        @test objective(tax) ≤ minimum(objective, taxes) + 1e-12
    end
end

@testset "Committed calendar paths and supporting-price jump" begin
    model = Model(testhousehold, testfirm, Climate(; γ = 0.2))
    solution = solvecommitted(model; verbose = 0)
    @test SciMLBase.successful_retcode(solution.retcode)
    @test solution.boundarygap ≤ 1e-6
    @test first(solution.t) == 0
    @test last(solution.t) == solution.stoppingtime
    @test first(solution.a) ≈ testfirm.a₀ atol = 1e-6
    @test last(solution.a) ≈ testfirm.η atol = 1e-6
    @test last(solution.q) ≈ c(testfirm) atol = 1e-6
    @test abs(first(solution.λq)) ≤ 1e-6
    path = CommittedTaxPath(solution)
    @test path(solution.stoppingtime; side = :left) ≈ model.climate.γ * testfirm.A^2 / testhousehold.ν
    @test path(solution.stoppingtime) ≈ committedsupportingtax(model)
    @test path(solution.stoppingtime + 100) ≈ committedsupportingtax(model)
    @test isfinite(solution.welfarecost)
    @test committedsolutionfilename(model; intervals = 100) != committedsolutionfilename(model; intervals = 200)
end
