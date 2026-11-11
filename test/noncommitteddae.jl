using Test

import Roots
import ForwardDiff as FD
import SparseArrays as SP
import DifferentiationInterface as DI
import SparseMatrixColorings as SMC
import ADTypes
import LinearSolve as LS
import NonlinearSolve as NLS
import SciMLBase

include("../src/primitives/constants.jl")
include("../src/primitives/signal.jl")
include("../src/primitives/climate.jl")
include("../src/agents/households.jl")
include("../src/agents/firm.jl")
include("../src/agents/government.jl")
include("../src/solve/model.jl")
include("../src/dynamics/government.jl")
include("../src/solve/government/noncommittedmarkov.jl")
include("../src/solve/government/noncommitteddae.jl")
include("../src/utils/saving.jl")

testhousehold = Household(; ν = 1.0, ϕ = 1.0, r = 0.1)
testfirm = Firm(; e₀ = 1.0, a₀ = 0.1, A = 1.0, η = 1.0, κ = 1.0, ξ = 2.0)
testmodel = Model(testhousehold, testfirm, Climate(; γ = 0.02))
testgovernment = Government(; κₚ = 0.5, b = 0.3)
testsignal = Signal(; ϵ = 1.0, σ = 0.5)

@testset "Explicit welfare coefficients and constant installation price" begin
    @test c(testfirm) == testfirm.κ * testfirm.a₀
    @test k(0.2, testfirm) ≈ c(testfirm) * 0.2 + testfirm.ξ * 0.2^2 / 2
    @test α(0.5, testfirm) ≈ (0.5 - c(testfirm)) / testfirm.ξ
    for a in (testfirm.a₀, 0.55, testfirm.η), τ in (0.0, 0.2, 0.8)
        labour = n(τ, a, testhousehold, testfirm)
        resourcecost = k(0.2, testfirm) + v(labour, testhousehold) - y(labour, testfirm) * (1 - d(e(labour, a, testfirm), testmodel.climate))
        @test simplifiedcost(τ, a, 0.2, testhousehold, testfirm, testmodel.climate) ≈ resourcecost
    end
    @test κ₁(testfirm.η, testhousehold, testfirm, testmodel.climate) == 0
    @test κ₂(testfirm.η, testhousehold, testfirm, testmodel.climate) == 0
end

@testset "Announcement jumps retain the correct one-sided mandate" begin
    path = CommittedTaxPath([0.6, 0.6, 0.3, 0.3, 0.1], [0.0, 0.4, 0.4, 1.0, 1.0], 0.1)
    parameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    jump = parameters.horizon - 0.4
    @test noncommitteddaemandate(parameters, 0.0, nothing) == 0.1
    @test noncommitteddaemandate(parameters, 0.0, (1.0, 0.4)) == 0.3
    @test noncommitteddaemandate(parameters, jump, (1.0, 0.4)) == 0.3
    @test noncommitteddaemandate(parameters, jump, (0.4, 0.0)) == 0.6
    @test noncommitteddaemandate(parameters, jump, nothing) == 0.3
    @test noncommitteddaemandate(parameters, 0.3, (1.0, 0.4)) == 0.3
    @test noncommitteddaemandate(parameters, 0.8, (0.4, 0.0)) == 0.6
end

@testset "Temporal schemes have distinct saved-solution identities" begin
    path = CommittedTaxPath([0.4, 0.4], [0.0, 1.0])
    parameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    finerparameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5, timestep = 0.5)
    grid = parameters.grid
    legacyparameters = NonCommittedMarkovParameters(testmodel, testgovernment, testsignal, path, grid, 1.0, 1e-8, 1e-7, 0.5, 1000)
    times = [0.0, 1.0]
    values = [NonCommittedMarkovValue(t, grid, zeros(size(grid)), zeros(size(grid)), zeros(size(grid)), zeros(size(grid))) for t in times]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), length(times))
    legacy = NonCommittedMarkovSolution(times, values, legacyparameters, diagnostics)
    current = NonCommittedMarkovSolution(times, values, parameters, diagnostics)
    finer = NonCommittedMarkovSolution(times, values, finerparameters, diagnostics)
    @test noncommittedsolutionkey(legacy) != noncommittedsolutionkey(current)
    @test noncommittedsolutionkey(current) != noncommittedsolutionkey(finer)
end

@testset "Zero announcement has exact constant-flow NPV" begin
    path = CommittedTaxPath([0.0, 0.0], [0.0, 1.0])
    parameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    q, tax = zeros(size(grid)), zeros(size(grid))
    welfare = [κ₀(physicalstate(grid, i, j)[2], testhousehold, testfirm, testmodel.climate) / testhousehold.r for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    value = NonCommittedMarkovValue(1.0, grid, q, copy(welfare), welfare, tax)
    x = noncommitteddaestate(value, parameters)
    residual = similar(x)
    noncommitteddaeresidual!(residual, zeros(length(x)), x, parameters, 0.0)
    @test maximum(abs, residual) < 1e-12
    @test maximum(abs, noncommitteddaederivative(x, parameters, 0.0)) < 1e-12
end

@testset "Valuation uses public beliefs and government value uses the true measure" begin
    mandate, tax = 0.6, 0.2
    path = CommittedTaxPath([mandate, mandate], [0.0, 1.0])
    parameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.25)
    grid = parameters.grid
    q = [0.02 + 0.01physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    uᵍ = [1 + physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    u = [2 + 2physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    value = NonCommittedMarkovValue(0.5, grid, q, uᵍ, u, fill(tax, size(grid)))
    x = noncommitteddaestate(value, parameters)
    dx, residual = zeros(length(x)), similar(x)
    D, R = reshape(dx, 4, size(grid)...), reshape(residual, 4, size(grid)...)
    i, j = 3, 2
    φ, a = physicalstate(grid, i, j)
    τᵉ = noncommittedexpectedtax(φ, tax, mandate)
    μ, _ = noncommittedbeliefmoments(tax, mandate, φ, testsignal)
    D[1, i, j] = (y(n(τᵉ, a, testhousehold, testfirm), testfirm) * τᵉ - testhousehold.r * q[i, j]) / parameters.qscale
    D[2, i, j] = (simplifiedcost(tax, a, 0, testhousehold, testfirm, testmodel.climate) - testhousehold.r * uᵍ[i, j]) / parameters.valuescale
    D[3, i, j] = (simplifiedcost(τᵉ, a, 0, testhousehold, testfirm, testmodel.climate) - testhousehold.r * u[i, j]) / parameters.valuescale
    noncommitteddaeresidual!(residual, dx, x, parameters, 0.5)
    @test R[1, i, j] ≈ 0 atol = 1e-11
    @test R[2, i, j] ≈ -μ / (testhousehold.r * parameters.valuescale) atol = 1e-11
    @test R[3, i, j] ≈ -2μ / (testhousehold.r * parameters.valuescale) atol = 1e-11
end

@testset "Supporting-price cap valuation and upper obstacle" begin
    output = testfirm.A^2 / testhousehold.ν
    support = testhousehold.r * c(testfirm) / output
    path = CommittedTaxPath([support, support], [0.0, 1.0])
    parameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    q = [physicalstate(grid, i, j)[1] * c(testfirm) for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    baseline = κ₀(testfirm.η, testhousehold, testfirm, testmodel.climate) / testhousehold.r
    value = NonCommittedMarkovValue(1.0, grid, q, fill(baseline, size(grid)), fill(baseline, size(grid)), zeros(size(grid)))
    x = noncommitteddaestate(value, parameters)
    dx, residual = zeros(length(x)), similar(x)
    noncommitteddaeresidual!(residual, dx, x, parameters, 0.0)
    R = reshape(residual, 4, size(grid)...)
    @test maximum(abs, R[:, :, end]) < 1e-12

    mandate = 0.4
    parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
    x = noncommitteddaestate(value, parameters)
    dx, residual = zeros(length(x)), similar(x)
    X, D, R = reshape(x, 4, size(grid)...), reshape(dx, 4, size(grid)...), reshape(residual, 4, size(grid)...)
    X[1, :, end] .= c(testfirm) / parameters.qscale
    i, j = size(grid)
    noncommitteddaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] ≈ 0 atol = 1e-12

    X[1, i, j] = 1.1c(testfirm) / parameters.qscale
    noncommitteddaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] > 0

    X[1, i, j] = 0.9c(testfirm) / parameters.qscale
    noncommitteddaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] < 0
    D[1, i, j] = (output * mandate - testhousehold.r * 0.9c(testfirm)) / parameters.qscale
    noncommitteddaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] ≈ 0 atol = 1e-12
end

@testset "Hidden deviations freeze firms' expectations and the filter" begin
    mandate, tax = 0.6, 0.2
    parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.25)
    grid = parameters.grid
    i, j = 3, 2
    φ, a = physicalstate(grid, i, j)
    τᵉ = noncommittedexpectedtax(φ, tax, mandate)
    B = φ * (1 - φ) * (testsignal.ϵ / testsignal.σ)^2 * (mandate - tax)
    taxes = range(0.0, mandate; length = 12001)

    for (backward, forward) in ((-0.25, 0.25), (-2.0, -1.5), (1.0, -1.0))
        uᵍ = [begin
            distance = physicalstate(grid, row, column)[1] - φ
            distance * (distance ≤ 0 ? backward : forward)
        end for row in 1:size(grid, 1), column in 1:size(grid, 2)]
        value = NonCommittedMarkovValue(0.5, grid, fill(0.7, size(grid)), uᵍ, copy(uᵍ), fill(tax, size(grid)))
        x = noncommitteddaestate(value, parameters)
        X = reshape(x, 4, size(grid)...)
        candidate, improvement = noncommitteddaetax(X, tax, i, j, mandate, parameters)
        objective(τ̂) = κ₁(a, testhousehold, testfirm, testmodel.climate) * τ̂ + κ₂(a, testhousehold, testfirm, testmodel.climate) * τ̂^2 / 2 + B * (τ̂ - τᵉ) * (backward + forward) / 2
        objectives = map(objective, taxes)
        @test abs(candidate - taxes[argmin(objectives)]) ≤ 1.01step(taxes)
        @test objective(candidate) ≤ minimum(objectives) + 1e-12
        @test improvement ≈ max(objective(tax) - objective(candidate), 0) atol = 1e-10
    end
end

@testset "Concave belief slopes retain a tax fixed point" begin
    i, j = 2, 2
    a = (testfirm.a₀ + testfirm.η) / 2
    curvature = κ₂(a, testhousehold, testfirm, testmodel.climate)
    mandate = -2κ₁(a, testhousehold, testfirm, testmodel.climate) / curvature
    parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    uᵍ = [begin
        distance = physicalstate(grid, row, column)[1] - 0.5
        curvature * distance * (distance ≤ 0 ? 1 : -2)
    end for row in 1:size(grid, 1), column in 1:size(grid, 2)]
    value = NonCommittedMarkovValue(0.5, grid, zeros(size(grid)), uᵍ, copy(uᵍ), zeros(size(grid)))
    x = reshape(noncommitteddaestate(value, parameters), 4, size(grid)...)
    residual(τ) = τ - first(noncommitteddaetax(x, τ, i, j, mandate, parameters))
    tax = Roots.find_zero(residual, (0.0, mandate), Roots.Brent())
    @test tax / mandate ≈ 2 / 3 atol = 1e-10
    @test abs(residual(tax)) < 1e-12
end

@testset "No-learning taxes remain accurate near the static optimum" begin
    signal = Signal(; ϵ = 0.0, σ = 0.5)
    a = (testfirm.a₀ + testfirm.η) / 2
    statictax = -κ₁(a, testhousehold, testfirm, testmodel.climate) / κ₂(a, testhousehold, testfirm, testmodel.climate)
    for mandate in (statictax - 1e-8, statictax + 1e-8)
        parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, signal; h = 0.5)
        x = zeros(4, size(parameters.grid)...)
        x[2, 1, :] .= 1
        x[2, end, :] .= -2
        for conjecture in (0.0, mandate / 2, mandate - 1e-12, mandate)
            candidate, _ = noncommitteddaetax(x, conjecture, 2, 2, mandate, parameters)
            @test candidate == clamp(statictax, 0, mandate)
        end
    end
end

@testset "Selected equilibrium is a hidden best response" begin
    a = (testfirm.a₀ + testfirm.η) / 2
    curvature = κ₂(a, testhousehold, testfirm, testmodel.climate)
    static = -κ₁(a, testhousehold, testfirm, testmodel.climate) / curvature
    for mandate in (static / 2, 2static), slope in (-4curvature, 0.0, 4curvature)
        parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
        x = zeros(4, size(parameters.grid)...)
        for j in axes(x, 3), i in axes(x, 2)
            x[2, i, j] = slope * physicalstate(parameters.grid, i, j)[1] / parameters.valuescale
        end
        tax, _ = noncommitteddaetax(x, 0, 2, 2, mandate, parameters; equilibrium = true)
        response, improvement = noncommitteddaetax(x, tax, 2, 2, mandate, parameters)
        @test tax ≈ response atol = 1e-12
        @test improvement < 1e-12
        if mandate < static
            @test tax == mandate
        end
    end
end

@testset "Frozen belief diffusion admits every hidden deviation" begin
    mandate = 0.6
    parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.25)
    grid = parameters.grid
    Δφ = grid.h * span(grid.φ)
    x = zeros(4, size(grid)...)
    baseline, field = similar(x), similar(x)
    for i in 2:(size(grid, 1) - 1), tax in (0.0, mandate / 3, mandate - 1e-8, mandate)
        fill!(x, 0)
        x[4, :, :] .= tax / parameters.taxscale
        noncommitteddaefield!(baseline, x, parameters, 0.5)
        x[1, i + 1, 2] = 1
        noncommitteddaefield!(field, x, parameters, 0.5)
        variance = field[1, i, 2] - baseline[1, i, 2]
        φ = physicalstate(grid, i, 2)[1]
        τᵉ = noncommittedexpectedtax(φ, tax, mandate)
        B = φ * (1 - φ) * (testsignal.ϵ / testsignal.σ)^2 * (mandate - tax)
        for deviation in range(0.0, mandate; length = 11)
            drift = B * (deviation - τᵉ) / (2Δφ)
            @test min(variance + drift, variance - drift) ≥ -1e-12
        end
    end
end

@testset "Sparse DAE Jacobian matches the combined residual derivative" begin
    mandate = 0.6
    parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    prototype = noncommitteddaejacobian(parameters)
    autodiff = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector = ADTypes.KnownJacobianSparsityDetector(prototype), coloring_algorithm = SMC.GreedyColoringAlgorithm())
    residual! = (residual, dx, x, parameters, s) -> noncommitteddaeresidual!(residual, dx, x, parameters, s)
    pattern = falses(size(prototype))
    for column in axes(prototype, 2), entry in SP.nzrange(prototype, column)
        pattern[SP.rowvals(prototype)[entry], column] = true
    end

    for active in (false, true)
        q = [0.2 + 0.01physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
        q[:, end] .= active ? 1.2c(testfirm) : 0.5c(testfirm)
        uᵍ = [1 + 0.2physicalstate(grid, i, j)[1] + 0.03physicalstate(grid, i, j)[2] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
        u = [2 + 0.3physicalstate(grid, i, j)[1] + 0.01physicalstate(grid, i, j)[2] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
        value = NonCommittedMarkovValue(0.5, grid, q, uᵍ, u, fill(0.2, size(grid)))
        x = noncommitteddaestate(value, parameters)
        dx = similar(x)
        noncommitteddaefield!(dx, x, parameters, 0.5)
        jacobian! = noncommitteddaejacobian(residual!, dx, x, parameters, 0.5, autodiff)
        for γ in (0.0, 0.7, 1.0)
            residual = function (z)
                result = similar(z)
                residual!(result, dx .+ γ .* (z .- x), z, parameters, 0.5)
                return result
            end
            reference = FD.jacobian(residual, x)
            jacobian = copy(prototype)
            jacobian!(jacobian, dx, x, parameters, γ, 0.5)
            @test size(prototype) == (length(x), length(x))
            @test all((abs.(reference) .≤ 1e-10) .| pattern)
            @test Matrix(jacobian) ≈ reference atol = 1e-10
        end
    end
end

@testset "Weak capacity activity has a consistent combined Jacobian" begin
    mandate = 0.6
    parameters = NonCommittedDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    q = fill(0.2, size(grid))
    q[:, end] .= c(testfirm)
    value = NonCommittedMarkovValue(0.5, grid, q, zeros(size(grid)), zeros(size(grid)), fill(0.2, size(grid)))
    x = noncommitteddaestate(value, parameters)
    dx = similar(x)
    noncommitteddaefield!(dx, x, parameters, 0.5)
    prototype = noncommitteddaejacobian(parameters)
    autodiff = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector = ADTypes.KnownJacobianSparsityDetector(prototype), coloring_algorithm = SMC.GreedyColoringAlgorithm())
    residual! = (residual, dx, x, parameters, s) -> noncommitteddaeresidual!(residual, dx, x, parameters, s)
    jacobian! = noncommitteddaejacobian(residual!, dx, x, parameters, 0.5, autodiff)
    matrices = map((0.0, 1.0, 1000.0)) do γ
        jacobian = copy(prototype)
        jacobian!(jacobian, dx, x, parameters, γ, 0.5)
        return Matrix(jacobian)
    end
    capindices = 1 .+ 4 .* ((size(grid, 2) - 1) * size(grid, 1) .+ (0:(size(grid, 1) - 1)))
    @test matrices[3] ≈ matrices[1] + 1000(matrices[2] - matrices[1]) atol = 1e-10
    @test matrices[3][capindices, :] == matrices[1][capindices, :]
end

@testset "Calendar intervals integrate from the future towards today" begin
    mandate, sunset = 0.02, 0.5
    path = CommittedTaxPath([mandate, mandate, 0.0, 0.0], [0.0, sunset, sunset, 1.0])
    model = Model(testhousehold, testfirm, Climate(; γ = 0.0))
    signal = Signal(; ϵ = 0.0, σ = 0.5)
    flow = y(n(mandate, testfirm.a₀, testhousehold, testfirm), testfirm) * mandate
    errors = map((0.25, 0.125, 0.0625)) do timestep
        solution = solvenoncommitteddae(path, model, testgovernment, signal; h = 0.5, timestep, savestep = 0.25, tolerance = 1e-9, policytolerance = 1e-9, verbose = 0)
        @test solution.t == [0.0, 0.25, 0.5, 0.75, 1.0]
        return maximum(abs(value.q[end, 1] + flow * expm1(-testhousehold.r * max(sunset - value.t, 0)) / testhousehold.r) for value in solution.u)
    end
    @test errors[2] < 0.6errors[1]
    @test errors[3] < 0.6errors[2]
    @test errors[3] < 4e-5
end

@testset "Roundoff-close mesh dates retain every output" begin
    path = CommittedTaxPath([0.02, 0.02, 0.0, 0.0], [0.0, 0.53, 0.53, 1.03])
    model = Model(testhousehold, testfirm, Climate(; γ = 0.0))
    signal = Signal(; ϵ = 0.0, σ = 0.5)
    solution = solvenoncommitteddae(path, model, testgovernment, signal; h = 0.5, timestep = 0.25, tolerance = 1e-9, policytolerance = 1e-9, verbose = 0)
    @test solution.t == sort!(vcat(collect(range(0.0, 1.03; length = 6)), 0.53))
    @test [value.t for value in solution.u] == solution.t
    @test maximum(d -> max(d.valuegap, d.bellmangap, d.policygap), solution.diagnostics) < 1e-9
end

@testset "Zero-announcement DAE solution retains calendar values" begin
    path = CommittedTaxPath([0.0, 0.0], [0.0, 1.0])
    solution = solvenoncommitteddae(path, testmodel, testgovernment, testsignal; h = 0.5, timestep = 0.5, tolerance = 1e-9, policytolerance = 1e-9, verbose = 0)
    @test solution.t == [0.0, 0.5, 1.0]
    for value in solution.u
        @test maximum(abs, value.q) ≤ 1e-9
        @test maximum(abs, value.τ) ≤ 1e-9
        @test value.uᵍ ≈ value.u atol = 1e-9
        for j in axes(value.u, 2), i in axes(value.u, 1)
            _, a = physicalstate(value.grid, i, j)
            @test value.u[i, j] ≈ κ₀(a, testhousehold, testfirm, testmodel.climate) / testhousehold.r atol = 1e-8
        end
    end
end

@testset "Calendar jumps use one-sided policy interpolation" begin
    grid = NonCommittedMarkovGrid(testmodel, 0.5)
    path = CommittedTaxPath([0.6, 0.6, 0.1, 0.1], [0.0, 0.4, 0.4, 1.0], 0.1)
    parameters = NonCommittedDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5, timestep = 0.25)
    values = [NonCommittedMarkovValue(t, grid, fill(q, size(grid)), fill(q + 1, size(grid)), fill(q + 2, size(grid)), fill(τ, size(grid))) for (t, q, τ) in ((0.0, 1.0, 0.3), (0.4, 2.0, 0.05), (1.0, 3.0, 0.04))]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), 3)
    solution = NonCommittedMarkovSolution([0.0, 0.4, 1.0], values, parameters, diagnostics)
    @test all(==(0.3), solution(0.3).τ)
    @test all(==(0.05), solution(0.4).τ)
    @test solution(0.3).q ≈ fill(1.75, size(grid))
    @test solution(0.3).u ≈ fill(3.75, size(grid))
    @test path(0.4; side = :left) == 0.6
    @test path(0.4) == 0.1

    @test_throws DomainError solution(-0.01)
    @test_throws DomainError solution(1.01)
end
