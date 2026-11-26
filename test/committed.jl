using Test
import Roots
import StaticArrays as SA
import FastClosures: @closure
import UnPack: @unpack

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
include("../src/solve/committed/solve.jl")

function committedhamiltonian(u, λ, parameters)
    model = parameters.model
    a = first(committedpolicy(model, u[1], u[3], u[4], u[2], λ[2], λ[3], λ[4]; zguess=parameters.zguess))
    a.welfare + λ[1] * a.investment + λ[2] * (model.household.r * u[2] - a.p) + λ[3] * (model.household.r * u[3] - a.profit) + λ[4] * (model.household.r * u[4] - cleanprofit(a.p, u[1], a.investment, model.clean))
end

@testset "Green equity and physical equations" begin
    calibration = Calibration()
    model = GeneralEquilibriumModel(calibration)
    @test model.government.υ == model.dirty.γᵈ
    @test model.bank.θᶜ == model.bank.θ
    @test policycost(0.1, 0.1, model.government) == 0
    @test policycost(0.05, 0.1, model.government) ≈ model.government.ψ * 0.05^2 / 2
    @test !hasproperty(model, :υ)
    @test lendinglimit(1.0, 2.0, model.bank) > lendinglimit(1.0, 0.0, model.bank)
    @test bankequity(1.0, 2.0, model.bank) - liquidityrequirement(lendinglimit(1.0, 2.0, model.bank), 1.0, 2.0, model.bank) ≈ 0 atol=1e-14

    # Compare analytical equilibrium derivatives with independent market-clearing perturbations.
    for (xᶜ, vᶜ, τ) in ((0.8, 0.5, 0.08), (0.8, 10.0, 0.08), (0.4, 3.0, 1.0), (0.05, 10.0, 0.02))
        inputs = SA.SVector(xᶜ, 0.6, vᶜ, τ)
        a = committedallocation(model, inputs[1], inputs[2], inputs[3], 4.5, inputs[4])
        ∂w, ∂p, ∂π = committedderivatives(model, a, 0.0, 0.0)
        @test 0 < a.xᵈ <= model.dirty.kᵈ
        @test 0 <= a.h <= model.dirty.η * a.xᵈ
        @test a.e ≈ model.dirty.η * a.xᵈ - a.h
        for j in 1:4
            h = 1e-5 * max(abs(inputs[j]), 1.0)
            direction = SA.SVector{4}(ntuple(k -> k == j ? h : 0.0, 4))
            left, right = inputs - direction, inputs + direction
            al = committedallocation(model, left[1], left[2], left[3], 4.5, left[4])
            ar = committedallocation(model, right[1], right[2], right[3], 4.5, right[4])
            @test ∂w[j] ≈ (ar.welfare - al.welfare) / (2h) rtol=1e-4 atol=1e-7
            @test ∂p[j] ≈ (ar.p - al.p) / (2h) rtol=1e-4 atol=1e-7
            @test ∂π[j] ≈ (ar.profit - al.profit) / (2h) rtol=1e-4 atol=1e-7
        end
    end

    reference = committedstationary(model, 0.03)
    @test reference.vᶜ ≈ reference.q * reference.xᶜ
    @test reference.p ≈ model.household.r * model.clean.c
    inputs = SA.SVector(reference.xᶜ, reference.vᵈ, reference.q)
    tail, ∂f, ∂g, ∂τ = committedtail(model, inputs[1], inputs[2], inputs[3])
    @test tail.e > 0
    for j in 1:3
        h = 1e-5 * max(abs(inputs[j]), 1.0)
        direction = SA.SVector{3}(ntuple(k -> k == j ? h : 0.0, 3))
        left, right = inputs - direction, inputs + direction
        al = first(committedtail(model, left[1], left[2], left[3]))
        ar = first(committedtail(model, right[1], right[2], right[3]))
        fl = left[2] - al.profit / model.household.r
        fr = right[2] - ar.profit / model.household.r
        @test ∂f[j] ≈ (fr - fl) / (2h) rtol=1e-4 atol=1e-7
        @test ∂g[j] ≈ (ar.welfare - al.welfare) / (2h * model.household.r) rtol=1e-4 atol=1e-7
        @test ∂τ[j] ≈ (ar.τ - al.τ) / (2h) rtol=1e-4 atol=1e-7
    end

    parameters = CommittedParameters(model, 0.4, reference.z)
    scale = SA.SVector(1.0, 4.0, 3.125, 4.0, 20.0, 5.0, 6.4, 5.0)
    scaled = ScaledCommittedParameters(parameters, 80.0, scale)
    u = SA.SVector(0.4, 4.5, 0.6, 3.0, -1.0, 0.1, 0.2, 0.3)
    rhs = @inferred dynamics(u, parameters, 0.0)
    λ = SA.SVector(u[5], u[6], u[7], u[8])
    for j in 1:4
        h = 1e-5 * max(abs(u[j]), 1.0)
        direction = SA.SVector{8}(ntuple(k -> k == j ? h : 0.0, 8))
        slope = (committedhamiltonian(u + direction, λ, parameters) - committedhamiltonian(u - direction, λ, parameters)) / (2h)
        @test rhs[j + 4] ≈ model.household.r * λ[j] - slope rtol=1e-3 atol=1e-6
    end
    du = zeros(8)
    dynamics!(du, collect(u ./ scale), scaled, 0.0)
    @test du ≈ 80 .* rhs ./ scale
    a = first(committedpolicy(model, u[1], u[3], u[4], u[2], u[6], u[7], u[8]))
    @test rhs[4] ≈ model.household.r * u[4] - cleanprofit(a.p, u[1], a.investment, model.clean)
    @test rhs[6] ≈ -((1 + u[8]) * u[2] + u[5]) / model.clean.ξ
    h = 1e-5
    forward = u + h * rhs
    backward = u - h * rhs
    optiondrift = ((forward[4] - forward[2] * forward[1]) - (backward[4] - backward[2] * backward[1])) / (2h)
    @test optiondrift ≈ model.household.r * (u[4] - u[2] * u[1]) - model.clean.ξ * a.investment^2 / 2 atol=1e-8

    endpoint = SA.SVector(inputs[1], inputs[3], inputs[2], inputs[1] * inputs[3], ∂g[1], ∂g[3], ∂g[2], 0.0)
    @test maximum(abs, terminalboundary(endpoint, parameters)) < 1e-8
    @test maximum(abs, terminalboundary(endpoint ./ scale, scaled)) < 1e-8
    # Construct endpoint costates from the constrained terminal Lagrangian.
    λvᶜ, valuationmultiplier, investmentmultiplier = 0.3, 0.2, 0.5
    gradient = ∂g + valuationmultiplier * ∂f + SA.SVector(0, 0, investmentmultiplier)
    endpoint = SA.SVector(inputs[1], inputs[3], inputs[2], inputs[1] * inputs[3], gradient[1] - inputs[3] * λvᶜ, gradient[3] - inputs[1] * λvᶜ, gradient[2], λvᶜ)
    a, taxmultiplier, recoveredinvestment = committedterminal(endpoint, parameters)
    @test taxmultiplier ≈ 0 atol=1e-10
    @test recoveredinvestment ≈ investmentmultiplier atol=1e-10
    @test maximum(abs, terminalboundary(endpoint, parameters)) < 1e-8
end

@testset "Firm KKT conditions and permit clearing" begin
    dirty = DirtyEnergy()
    for p in (0.10, 0.30, 1.0), τ in (0.02, 0.10, 0.80)
        xd = dirtyoutput(p, τ, dirty)
        h = dirtyabatement(τ, xd, dirty)
        profit = dirtyprofit(p, τ, xd, dirty)
        @test 0 <= xd <= dirty.kᵈ
        @test 0 <= h <= dirty.η * xd
        for trialxd in range(0, dirty.kᵈ; length=31), trialh in range(0, dirty.η * trialxd; length=31)
            trialprofit = p * trialxd - dirty.γᵈ * trialxd^2 / 2 - dirty.ζ * trialh^2 / 2 - τ * (dirty.η * trialxd - trialh)
            @test profit >= trialprofit - 1e-12
        end
        if 0 < h < dirty.η * xd && 0 < xd < dirty.kᵈ
            @test dirty.ζ * h ≈ τ
            @test (p - dirty.γᵈ * xd) / dirty.η ≈ τ
        end
        if h < dirty.η * xd
            @test permitprice(dirty.η * xd - h, p, dirty) ≈ τ
        end
    end
    @test dirtyoutput(0.30, 1.0, dirty) > 0
    @test dirtyemissions(1.0, dirtyoutput(0.30, 1.0, dirty), dirty) == 0
end
