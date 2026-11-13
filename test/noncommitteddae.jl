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
import OrdinaryDiffEq as ODE
import JLD2
import LogExpFunctions as LE

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
include("../src/dynamics/committed.jl")
include("../src/utils/saving.jl")
include("../src/utils/arguments.jl")
include("../src/dynamics/markovsimulation.jl")
include("../scripts/plotting/reputationutils.jl")

testhousehold = Household(; ν = 1.0, ϕ = 1.0, r = 0.1)
testfirm = Firm(; e₀ = 1.0, a₀ = 0.1, A = 1.0, η = 1.0, κ = 1.0, ξ = 2.0)
testmodel = Model(testhousehold, testfirm, Climate(; γ = 0.02))
testgovernment = Government(; κᵇ = 0.5, τ̄ = 0.3)
testsignal = Signal(; ϵ = 1.0, σ = 0.5)

@testset "Behavioural command-line calibration uses model tax units" begin
    _, _, preference, _, _ = initmodels(["--preferred-tax=25", "--behavioural-curvature=0.004"])
    @test preference.τ̄ == 25taxfactor
    @test preference.κᵇ == 0.004
    path = CommittedTaxPath([0.6, 0.6], [0.0, 1.0])
    @test_throws ArgumentError NCDAEParameters(path, testmodel, Government(; κᵇ = 0.0), testsignal)
    @test_throws ArgumentError NCDAEParameters(path, testmodel, Government(; τ̄ = -0.1), testsignal)
end

@testset "Default behavioural preference matches its counterfactual utility target" begin
    government = Government()
    benefit = -behaviouralcost(defaultdeviationtax - defaultdeviationcut, defaultdeviationtax, government)
    @test government.κᵇ ≈ 0.06581428571428571
    @test government.τ̄ == 10taxfactor
    @test benefit / y₀ ≈ defaultdeviationbenefit
    @test behaviouralcost(defaultdeviationtax, defaultdeviationtax, government) == 0
end

@testset "EU investment costs match the transition calibration" begin
    european = Firm()
    household = Household()
    output = y(n(0, european.a₀, household, european), european)
    horizon = defaultadjustmenthorizon
    investment = (european.η - european.a₀) / horizon
    surplus = Climate().γ * output^2 / household.r - c(european)
    @test c(european) > 0
    @test european.ξ > 0
    @test surplus > 0
    @test k(investment, european) / output ≈ defaultinvestmentshare
    @test surplus / european.ξ * (horizon - (1 - exp(-household.r * horizon)) / household.r) ≈ european.η - european.a₀
end

@testset "Explicit welfare coefficients and constant installation price" begin
    @test c(testfirm) == testfirm.κ * testfirm.a₀
    @test k(0.2, testfirm) ≈ c(testfirm) * 0.2 + testfirm.ξ * 0.2^2 / 2
    @test α(0.5, testfirm) ≈ (0.5 - c(testfirm)) / testfirm.ξ
    for a in (testfirm.a₀, 0.55, testfirm.η), τ in (0.0, 0.2, 0.8)
        labour = n(τ, a, testhousehold, testfirm)
        resourcecost = k(0.2, testfirm) + v(labour, testhousehold) - y(labour, testfirm) * (1 - d(e(labour, a, testfirm), testmodel.climate))
        @test simplifiedcost(τ, a, 0.2, testhousehold, testfirm, testmodel.climate) ≈ resourcecost
        @test simplifiedcost(τ, a, 0.2, testhousehold, testfirm, testmodel.climate; relative = true) ≈ resourcecost - κ₀(testfirm.η, testhousehold, testfirm, testmodel.climate)
    end
    @test κ₁(testfirm.η, testhousehold, testfirm, testmodel.climate) == 0
    @test κ₂(testfirm.η, testhousehold, testfirm, testmodel.climate) == 0
end

@testset "Nonlinear labour supply uses the primitive welfare flow" begin
    household = Household(; ν = 0.8, ϕ = 2.0, r = 0.1)
    for a in (0.1, 0.55, 0.99), τ in (0.0, 0.2, 0.8)
        labour = n(τ, a, household, testfirm)
        resourcecost = k(0.2, testfirm) + v(labour, household) - y(labour, testfirm) * (1 - d(e(labour, a, testfirm), testmodel.climate))
        cost(tax) = simplifiedcost(tax, a, 0.2, household, testfirm, testmodel.climate)
        @test cost(τ) ≈ resourcecost
        @test simplifiedcost(τ, a, 0.2, household, testfirm, testmodel.climate; relative = true) ≈ resourcecost - κ₀(testfirm.η, household, testfirm, testmodel.climate)
        @test ∂ₜsimplifiedcost(τ, a, household, testfirm, testmodel.climate) ≈ FD.derivative(cost, τ) atol = 1e-11
        @test ∂ₜₜsimplifiedcost(τ, a, household, testfirm, testmodel.climate) ≈ FD.derivative(tax -> FD.derivative(cost, tax), τ) atol = 1e-11
        @test ∂ₐsimplifiedcost(τ, a, 0.2, household, testfirm, testmodel.climate) ≈ FD.derivative(state -> simplifiedcost(τ, state, 0.2, household, testfirm, testmodel.climate), a) atol = 1e-11
    end

    for a in (0.1, 0.55, 0.99)
        preferred = statictax(a, household, testfirm, testmodel.climate)
        objective(τ) = simplifiedcost(τ, a, 0.0, household, testfirm, testmodel.climate)
        taxes = range(0.0, inv(testfirm.η); length = 12001)
        @test objective(preferred) ≤ minimum(objective, taxes) + 1e-12
    end
    zerotaxlabour = n(0, testfirm.η, household, testfirm)
    capcost = v(zerotaxlabour, household) - y(zerotaxlabour, testfirm)
    @test simplifiedcost(0.8, testfirm.η, 0.0, household, testfirm, testmodel.climate) ≈ capcost
end

@testset "Announcement jumps retain the correct one-sided mandate" begin
    path = CommittedTaxPath([0.6, 0.6, 0.3, 0.3, 0.1], [0.0, 0.4, 0.4, 1.0, 1.0], 0.1)
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    jump = parameters.horizon - 0.4
    @test ncdaemandata(parameters, 0.0, nothing) == 0.1
    @test ncdaemandata(parameters, 0.0, (1.0, 0.4)) == 0.3
    @test ncdaemandata(parameters, jump, (1.0, 0.4)) == 0.3
    @test ncdaemandata(parameters, jump, (0.4, 0.0)) == 0.6
    @test ncdaemandata(parameters, jump, nothing) == 0.3
    @test ncdaemandata(parameters, 0.3, (1.0, 0.4)) == 0.3
    @test ncdaemandata(parameters, 0.8, (0.4, 0.0)) == 0.6
end

@testset "Temporal schemes have distinct saved-solution identities" begin
    path = CommittedTaxPath([0.4, 0.4], [0.0, 1.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    finerparameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5, timestep = 0.5)
    grid = parameters.grid
    legacyparameters = NonCommittedMarkovParameters(testmodel, testgovernment, testsignal, path, grid, 1.0, 1e-8, 1e-7, 0.5, 1000)
    times = [0.0, 1.0]
    values = [NCMarkovValue(t, grid, zeros(size(grid)), zeros(size(grid)), zeros(size(grid)), zeros(size(grid))) for t in times]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), length(times))
    legacy = NonCommittedMarkovSolution(times, values, legacyparameters, diagnostics)
    current = NonCommittedMarkovSolution(times, values, parameters, diagnostics)
    finer = NonCommittedMarkovSolution(times, values, finerparameters, diagnostics)
    @test ncsolutionkey(legacy) != ncsolutionkey(current)
    @test ncsolutionkey(current) != ncsolutionkey(finer)
    alternative = NCDAEParameters(path, testmodel, Government(; κᵇ = 1.0, τ̄ = testgovernment.τ̄), testsignal; h = 0.5)
    @test ncsolutionkey(current) != ncsolutionkey(NonCommittedMarkovSolution(times, values, alternative, diagnostics))
    alternative = NCDAEParameters(path, testmodel, Government(; κᵇ = testgovernment.κᵇ, τ̄ = 0.2), testsignal; h = 0.5)
    @test ncsolutionkey(current) != ncsolutionkey(NonCommittedMarkovSolution(times, values, alternative, diagnostics))
    mktempdir() do directory
        filename = joinpath(directory, "current.jld2")
        key = ncsolutionkey(current)
        JLD2.jldopen(filename, "w") do file
            file["ncversion"] = ncversion
            file["$key/solution"] = current
        end
        loaded = loadreputationequilibrium(filename)
        @test loaded.solution.parameters.government == testgovernment
        @test ncsolutionkey(loaded.solution) == key
        JLD2.jldopen(joinpath(directory, "legacy.jld2"), "w") do file
            file["noncommitted/legacy"] = true
        end
        @test isnothing(loadreputationequilibrium(joinpath(directory, "legacy.jld2")))
    end
end

@testset "Zero announcement has exact constant-flow NPV" begin
    path = CommittedTaxPath([0.0, 0.0], [0.0, 1.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    q, tax = zeros(size(grid)), zeros(size(grid))
    welfare = [κ₀(physicalstate(grid, i, j)[2], testhousehold, testfirm, testmodel.climate) / testhousehold.r for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    value = NCMarkovValue(1.0, grid, q, copy(welfare), welfare, tax)
    x = ncstate(value, parameters)
    residual = similar(x)
    ncdaeresidual!(residual, zeros(length(x)), x, parameters, 0.0)
    @test maximum(abs, residual) < 1e-12
    @test maximum(abs, ncdaederivative(x, parameters, 0.0)) < 1e-12
end

@testset "Valuation uses public beliefs and government value uses the true measure" begin
    mandate, tax = 0.6, 0.2
    path = CommittedTaxPath([mandate, mandate], [0.0, 1.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.25)
    grid = parameters.grid
    q = [0.02 + 0.01physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    uᵍ = [1 + physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    u = [2 + 2physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    value = NCMarkovValue(0.5, grid, q, uᵍ, u, fill(tax, size(grid)))
    x = ncstate(value, parameters)
    dx, residual = zeros(length(x)), similar(x)
    D, R = reshape(dx, 4, size(grid)...), reshape(residual, 4, size(grid)...)
    i, j = 3, 2
    φ, a = physicalstate(grid, i, j)
    τᵉ = ncexpectedtax(φ, tax, mandate)
    μ, _ = ncbeliefmoments(tax, mandate, φ, testsignal)
    D[1, i, j] = (y(n(τᵉ, a, testhousehold, testfirm), testfirm) * τᵉ - testhousehold.r * q[i, j]) / parameters.qscale
    D[2, i, j] = (simplifiedcost(τᵉ, a, 0, testhousehold, testfirm, testmodel.climate) - testhousehold.r * uᵍ[i, j]) / parameters.valuescale
    D[3, i, j] = (simplifiedcost(τᵉ, a, 0, testhousehold, testfirm, testmodel.climate) - testhousehold.r * u[i, j]) / parameters.valuescale
    ncdaeresidual!(residual, dx, x, parameters, 0.5)
    @test R[1, i, j] ≈ 0 atol = 1e-11
    @test R[2, i, j] ≈ -μ / (testhousehold.r * parameters.valuescale) atol = 1e-11
    @test R[3, i, j] ≈ -2μ / (testhousehold.r * parameters.valuescale) atol = 1e-11
end

@testset "Supporting-price cap valuation and upper obstacle" begin
    output = testfirm.A^2 / testhousehold.ν
    support = testhousehold.r * c(testfirm) / output
    path = CommittedTaxPath([support, support], [0.0, 1.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    q = fill(c(testfirm), size(grid))
    baseline = κ₀(testfirm.η, testhousehold, testfirm, testmodel.climate) / testhousehold.r
    value = NCMarkovValue(1.0, grid, q, fill(baseline, size(grid)), fill(baseline, size(grid)), fill(support, size(grid)))
    x = ncstate(value, parameters)
    dx, residual = zeros(length(x)), similar(x)
    ncdaeresidual!(residual, dx, x, parameters, 0.0)
    R = reshape(residual, 4, size(grid)...)
    @test maximum(abs, R[:, :, end]) < 1e-12

    mandate = 0.4
    parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
    x = ncstate(value, parameters)
    dx, residual = zeros(length(x)), similar(x)
    X, D, R = reshape(x, 4, size(grid)...), reshape(dx, 4, size(grid)...), reshape(residual, 4, size(grid)...)
    X[1, :, end] .= c(testfirm) / parameters.qscale
    i, j = size(grid)
    ncdaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] ≈ 0 atol = 1e-12

    X[1, i, j] = 1.1c(testfirm) / parameters.qscale
    ncdaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] > 0

    X[1, i, j] = 0.9c(testfirm) / parameters.qscale
    ncdaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] < 0
    D[1, i, j] = (output * mandate - testhousehold.r * 0.9c(testfirm)) / parameters.qscale
    ncdaeresidual!(residual, dx, x, parameters, 0.0)
    @test R[1, i, j] ≈ 0 atol = 1e-12
end

@testset "Behavioural costs are signed and vanish on the conjectured policy" begin
    for conjecture in (0.0, 0.2, 0.6), tax in (0.0, 0.3, 0.6)
        cost(τ̂) = behaviouralcost(τ̂, conjecture, testgovernment)
        @test cost(conjecture) == 0
        @test FD.derivative(cost, tax) ≈ testgovernment.κᵇ * (tax - testgovernment.τ̄) atol = 1e-15
        @test FD.derivative(τ̂ -> FD.derivative(cost, τ̂), tax) ≈ testgovernment.κᵇ
    end
    @test behaviouralcost(testgovernment.τ̄, 0.6, testgovernment) < 0
    @test behaviouralcost(0.8, 0.6, testgovernment) > 0
end

@testset "Hidden best responses freeze allocations and filtering under both labour rules" begin
    mandate, conjecture = 0.6, 0.2
    taxes = range(0.0, mandate; length = 12001)
    for household in (testhousehold, Household(; ν = 0.8, ϕ = 2.0, r = 0.1))
        model = Model(household, testfirm, testmodel.climate)
        parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), model, testgovernment, testsignal; h = 0.25)
        grid = parameters.grid
        i, j = 3, 2
        φ, a = physicalstate(grid, i, j)
        τᵉ = ncexpectedtax(φ, conjecture, mandate)
        B = φ * (1 - φ) * (testsignal.ϵ / testsignal.σ)^2 * (mandate - conjecture)
        for (backward, forward) in ((-0.25, 0.25), (-2.0, -1.5), (1.0, -1.0), (0.2, 0.4))
            uᵍ = [begin
                distance = physicalstate(grid, row, column)[1] - φ
                distance * (distance ≤ 0 ? backward : forward)
            end for row in 1:size(grid, 1), column in 1:size(grid, 2)]
            value = NCMarkovValue(0.5, grid, fill(0.7, size(grid)), uᵍ, copy(uᵍ), fill(conjecture, size(grid)))
            x = reshape(ncstate(value, parameters), 4, size(grid)...)
            candidate, improvement = ncdaetax(x, conjecture, i, j, mandate, parameters)
            objective(τ̂) = behaviouralcost(τ̂, conjecture, testgovernment) + B * (τ̂ - τᵉ) * (backward + forward) / 2
            objectives = map(objective, taxes)
            @test abs(candidate - taxes[argmin(objectives)]) ≤ 1.01step(taxes)
            @test objective(candidate) ≤ minimum(objectives) + 1e-12
            @test improvement ≈ max(objective(conjecture) - objective(candidate), 0) atol = 1e-10
            field = similar(x)
            ncdaefield!(field, x, parameters, 0.5)
            @test field[2, :, :] == field[3, :, :]
        end
    end
end

@testset "Consistent behavioural policies satisfy the frozen best response and boundary conditions" begin
    for mandate in (0.1, 0.3, 0.6), slope in (-2.0, -0.01, 0.0, 0.2, 2.0)
        parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
        grid = parameters.grid
        x = zeros(4, size(grid)...)
        for j in axes(x, 3), i in axes(x, 2)
            x[2, i, j] = slope * physicalstate(grid, i, j)[1] / parameters.valuescale
        end
        tax, _ = ncdaetax(x, 0, 2, 2, mandate, parameters; equilibrium = true)
        response, improvement = ncdaetax(x, tax, 2, 2, mandate, parameters)
        @test tax ≈ response atol = 1e-12
        @test improvement < 1e-12
        @test 0 ≤ tax ≤ mandate
        @test first(ncdaetax(x, tax, 1, 2, mandate, parameters)) == min(testgovernment.τ̄, mandate)
        @test first(ncdaetax(x, tax, 2, size(grid, 2), mandate, parameters)) == min(testgovernment.τ̄, mandate)
        if slope < 0 && mandate > testgovernment.τ̄
            ρ = -0.25 * (testsignal.ϵ / testsignal.σ)^2 * slope
            @test tax ≈ (testgovernment.κᵇ * testgovernment.τ̄ + ρ * mandate) / (testgovernment.κᵇ + ρ)
        end
    end
end

@testset "No-learning policies equal the bounded behavioural target" begin
    signal = Signal(; ϵ = 0.0, σ = 0.5)
    for mandate in (testgovernment.τ̄ - 1e-8, testgovernment.τ̄ + 1e-8)
        parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, signal; h = 0.5)
        x = zeros(4, size(parameters.grid)...)
        x[2, 1, :] .= 1
        x[2, end, :] .= -2
        for conjecture in (0.0, mandate / 2, mandate - 1e-12, mandate)
            @test first(ncdaetax(x, conjecture, 2, 2, mandate, parameters)) == min(testgovernment.τ̄, mandate)
        end
    end
end

@testset "Simulated capacity policy retains the behavioural target" begin
    path = CommittedTaxPath([0.6, 0.6], [0.0, 1.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    values = [ncinitial(t, 0.6, parameters) for t in (0.0, 1.0)]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), 2)
    solution = NonCommittedMarkovSolution([0.0, 1.0], values, parameters, diagnostics)
    dynamics = MarkovDynamics(solution, [value.τ ./ 0.6 for value in values], false, 1.0)
    objects = dynamicvalues([0.0, testfirm.η, 0.0], 0.5, dynamics)
    @test objects[2] == testgovernment.τ̄
    @test objects[6] == 0
    @test objects[9] == 0
end

@testset "Frozen belief diffusion admits every hidden deviation" begin
    mandate = 0.6
    parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.25)
    grid = parameters.grid
    Δφ = grid.h * span(grid.φ)
    x = zeros(4, size(grid)...)
    baseline, field = similar(x), similar(x)
    for i in 2:(size(grid, 1) - 1), tax in (0.0, mandate / 3, mandate - 1e-8, mandate)
        fill!(x, 0)
        x[4, :, :] .= tax / parameters.taxscale
        ncdaefield!(baseline, x, parameters, 0.5)
        x[1, i + 1, 2] = 1
        ncdaefield!(field, x, parameters, 0.5)
        variance = field[1, i, 2] - baseline[1, i, 2]
        φ = physicalstate(grid, i, 2)[1]
        τᵉ = ncexpectedtax(φ, tax, mandate)
        B = φ * (1 - φ) * (testsignal.ϵ / testsignal.σ)^2 * (mandate - tax)
        for deviation in range(0.0, mandate; length = 11)
            drift = B * (deviation - τᵉ) / (2Δφ)
            @test min(variance + drift, variance - drift) ≥ -1e-12
        end
    end
end

@testset "Sparse DAE Jacobian matches the combined residual derivative" begin
    for model in (testmodel, Model(Household(; ν = 0.8, ϕ = 2.0, r = 0.1), testfirm, testmodel.climate))
        mandate = 0.6
        parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), model, testgovernment, testsignal; h = 0.5)
        grid = parameters.grid
        prototype = ncdaejacobian(parameters)
        autodiff = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector = ADTypes.KnownJacobianSparsityDetector(prototype), coloring_algorithm = SMC.GreedyColoringAlgorithm())
        residual! = (residual, dx, x, parameters, s) -> ncdaeresidual!(residual, dx, x, parameters, s)
        pattern = falses(size(prototype))
        for column in axes(prototype, 2), entry in SP.nzrange(prototype, column)
            pattern[SP.rowvals(prototype)[entry], column] = true
        end

        for active in (false, true), slope in (-0.01, 0.2)
            q = [0.2 + 0.01physicalstate(grid, i, j)[1] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
            q[:, end] .= active ? 1.2c(testfirm) : 0.5c(testfirm)
            uᵍ = [1 + slope * physicalstate(grid, i, j)[1] + 0.03physicalstate(grid, i, j)[2] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
            u = [2 + 0.3physicalstate(grid, i, j)[1] + 0.01physicalstate(grid, i, j)[2] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
            value = NCMarkovValue(0.5, grid, q, uᵍ, u, fill(0.2, size(grid)))
            x = ncstate(value, parameters)
            dx = similar(x)
            ncdaefield!(dx, x, parameters, 0.5)
            jacobian! = ncdaejacobian(residual!, dx, x, parameters, 0.5, autodiff)
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
end

@testset "Weak capacity activity has a consistent combined Jacobian" begin
    mandate = 0.6
    parameters = NCDAEParameters(CommittedTaxPath([mandate, mandate], [0.0, 1.0]), testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    q = fill(0.2, size(grid))
    q[:, end] .= c(testfirm)
    value = NCMarkovValue(0.5, grid, q, zeros(size(grid)), zeros(size(grid)), fill(0.2, size(grid)))
    x = ncstate(value, parameters)
    dx = similar(x)
    ncdaefield!(dx, x, parameters, 0.5)
    prototype = ncdaejacobian(parameters)
    autodiff = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector = ADTypes.KnownJacobianSparsityDetector(prototype), coloring_algorithm = SMC.GreedyColoringAlgorithm())
    residual! = (residual, dx, x, parameters, s) -> ncdaeresidual!(residual, dx, x, parameters, s)
    jacobian! = ncdaejacobian(residual!, dx, x, parameters, 0.5, autodiff)
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
        solution = solvencdae(path, model, testgovernment, signal; h = 0.5, timestep, savestep = 0.25, tolerance = 1e-9, policytolerance = 1e-9, verbose = 0)
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
    solution = solvencdae(path, model, testgovernment, signal; h = 0.5, timestep = 0.25, tolerance = 1e-9, policytolerance = 1e-9, verbose = 0)
    @test solution.t == sort!(vcat(collect(range(0.0, 1.03; length = 6)), 0.53))
    @test [value.t for value in solution.u] == solution.t
    @test maximum(d -> max(d.valuegap, d.bellmangap, d.policygap), solution.diagnostics) < 1e-9
end

@testset "Zero-announcement DAE solution retains calendar values" begin
    for household in (testhousehold, Household(; ν = 0.8, ϕ = 2.0, r = 0.1))
        model = Model(household, testfirm, testmodel.climate)
        path = CommittedTaxPath([0.0, 0.0], [0.0, 1.0])
        solution = solvencdae(path, model, testgovernment, testsignal; h = 0.5, timestep = 0.5, tolerance = 1e-9, policytolerance = 1e-9, verbose = 0)
        @test solution.t == [0.0, 0.5, 1.0]
        for value in solution.u
            @test maximum(abs, value.q) ≤ 1e-9
            @test maximum(abs, value.τ) ≤ 1e-9
            @test value.uᵍ ≈ value.u atol = 1e-9
            for j in axes(value.u, 2), i in axes(value.u, 1)
                _, a = physicalstate(value.grid, i, j)
                @test value.u[i, j] ≈ κ₀(a, household, testfirm, model.climate) / household.r atol = 1e-8
            end
        end
    end
end

@testset "Calendar jumps use one-sided policy interpolation" begin
    grid = NonCommittedMarkovGrid(testmodel, 0.5)
    path = CommittedTaxPath([0.6, 0.6, 0.1, 0.1], [0.0, 0.4, 0.4, 1.0], 0.1)
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5, timestep = 0.25)
    values = [NCMarkovValue(t, grid, fill(q, size(grid)), fill(q + 1, size(grid)), fill(q + 2, size(grid)), fill(τ, size(grid))) for (t, q, τ) in ((0.0, 1.0, 0.3), (0.4, 2.0, 0.05), (1.0, 3.0, 0.04))]
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

@testset "Known-commitment valuation interpolation uses the certainty boundary" begin
    path = CommittedTaxPath([0.0, 0.0], [0.0, 1.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    values = [NCMarkovValue(t, grid, [10i + physicalstate(grid, i, j)[2] + 2t for i in 1:size(grid, 1), j in 1:size(grid, 2)], zeros(size(grid)), zeros(size(grid)), zeros(size(grid))) for t in (0.0, 1.0)]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), 2)
    solution = NonCommittedMarkovSolution([0.0, 1.0], values, parameters, diagnostics)
    @test committedvaluation(0.35, 0.5, solution) ≈ 31.35
    @test committedvaluation(testfirm.η, 1.0, solution) ≈ 33.0
end

@testset "Known-commitment characteristic follows investment and respects capacity" begin
    path = CommittedTaxPath([0.0, 0.0], [0.0, 20.0])
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    rate = 0.1
    q = c(testfirm) + testfirm.ξ * rate
    values = [NCMarkovValue(t, grid, fill(q, size(grid)), zeros(size(grid)), zeros(size(grid)), zeros(size(grid))) for t in (0.0, 20.0)]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), 2)
    solution = NonCommittedMarkovSolution([0.0, 20.0], values, parameters, diagnostics)
    times = [0.0, 1.0, 20.0]
    result = simulatecharacteristic(solution, times; timestep = 0.1)
    @test result[:, 1] == ones(3)
    @test result[2, 4] ≈ testfirm.a₀ + rate atol = 1e-7
    @test result[2, 10] ≈ testfirm.η - testfirm.a₀ - rate / 2 atol = 1e-7
    @test result[end, 4] ≈ testfirm.η atol = 1e-7
    @test result[end, 6] == 0
    @test result[end, 9] == 0
    @test result[end, 10] ≈ (testfirm.η - testfirm.a₀)^2 / (2rate) atol = 1e-6
end

@testset "Known-commitment characteristic preserves announcement jumps" begin
    path = CommittedTaxPath([0.6, 0.6, 0.1, 0.1], [0.0, 0.4, 0.4, 1.0], 0.1)
    parameters = NCDAEParameters(path, testmodel, testgovernment, testsignal; h = 0.5)
    grid = parameters.grid
    values = [NCMarkovValue(t, grid, fill(c(testfirm), size(grid)), zeros(size(grid)), zeros(size(grid)), zeros(size(grid))) for t in (0.0, 1.0)]
    diagnostics = fill(NonCommittedMarkovDiagnostics(0, 0.0, 0.0, 0.0, 0.0), 2)
    solution = NonCommittedMarkovSolution([0.0, 1.0], values, parameters, diagnostics)
    times = [0.0, 0.4, 0.4, 1.0]
    result = simulatecharacteristic(solution, times)
    @test result[:, 2] == [0.6, 0.6, 0.1, 0.1]
    @test result[:, 4] == fill(testfirm.a₀, 4)
    @test result[2, 10] == result[3, 10]
    before = e(n(0.6, testfirm.a₀, testhousehold, testfirm), testfirm.a₀, testfirm)
    after = e(n(0.1, testfirm.a₀, testhousehold, testfirm), testfirm.a₀, testfirm)
    @test result[end, 10] ≈ 0.4before + 0.6after atol = 1e-7
end
