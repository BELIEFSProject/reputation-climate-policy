function ncdaemandata(parameters, s, interval)
    if !isnothing(interval)
        if s == parameters.horizon - first(interval)
            return parameters.τᶜ(first(interval); side = :left)
        elseif s == parameters.horizon - last(interval)
            return parameters.τᶜ(last(interval))
        end
    end
    return parameters.τᶜ(parameters.horizon - s)
end

function ncdaegenerator(x, component, i, j, λₐ, λ⁺, λ⁻)
    center = x[component, i, j]
    value = zero(center)
    if i > 1
        value += λ⁻ * (x[component, i - 1, j] - center)
    end
    if i < size(x, 2)
        value += λ⁺ * (x[component, i + 1, j] - center)
    end
    if j < size(x, 3)
        value += λₐ * (x[component, i, j + 1] - center)
    end
    return value
end

"Value rates in remaining calendar time s=T-t; tax is an algebraic residual."
function ncdaefield!(field, state, parameters, s; interval = nothing)
    model, grid = parameters.model, parameters.grid
    household, firm, climate = model.household, model.firm, model.climate
    x = reshape(state, 4, size(grid)...)
    f = reshape(field, size(x))
    τᶜ = ncdaemandata(parameters, s, interval)
    Δφ, Δa = grid.h * span(grid.φ), grid.h * span(grid.A)

    @inbounds Threads.@threads :dynamic for idx in CartesianIndices(grid)
        i, j = idx.I
        φ, a = physicalstate(grid, i, j)
        q = parameters.qscale * x[1, i, j]
        τ = clamp(parameters.taxscale * x[4, i, j], 0, τᶜ)
        τᵉ = ncexpectedtax(φ, τ, τᶜ)
        investment = j == size(grid, 2) ? 0 : α(q, firm)
        μᵩ, σᵩ = ncbeliefmoments(τ, τᶜ, φ, parameters.signal)
        B = φ * (1 - φ) * (parameters.signal.ϵ / parameters.signal.σ)^2 * (τᶜ - τ)
        # Freeze enough diffusion to keep both belief rates non-negative under every hidden deviation.
        variance = max(σᵩ^2 / (2Δφ^2), B * max(τᵉ, τᶜ - τᵉ) / (2Δφ))
        λₐ = investment / Δa
        λ⁺, λ⁻ = max(variance + μᵩ / (2Δφ), 0), max(variance - μᵩ / (2Δφ), 0)

        # Firm valuation uses the public measure, under which beliefs are martingales.
        f[1, i, j] = ncdaegenerator(x, 1, i, j, λₐ, variance, variance) + y(n(τᵉ, a, household, firm), firm) * τᵉ / parameters.qscale - household.r * x[1, i, j]
        if j == size(grid, 2)
            f[2, i, j], f[3, i, j] = 0, 0
        else
            f[2, i, j] = ncdaegenerator(x, 2, i, j, λₐ, λ⁺, λ⁻) + simplifiedcost(τᵉ, a, investment, household, firm, climate; relative = true) / parameters.valuescale - household.r * x[2, i, j]
            f[3, i, j] = ncdaegenerator(x, 3, i, j, λₐ, λ⁺, λ⁻) + simplifiedcost(τᵉ, a, investment, household, firm, climate; relative = true) / parameters.valuescale - household.r * x[3, i, j]
        end
        # The nonlinear tax is already an algebraic state: enforce its exact best-response fixed point.
        candidate, _ = ncdaetax(x, τ, i, j, τᶜ, parameters; equilibrium = true)
        f[4, i, j] = x[4, i, j] - candidate / parameters.taxscale
    end
    return nothing
end

function ncdaeresidual!(residual, derivative, state, parameters, s; interval = nothing)
    ncdaefield!(residual, state, parameters, s; interval)
    x = reshape(state, 4, size(parameters.grid)...)
    dx, r = reshape(derivative, size(x)), reshape(residual, size(x))
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate

    @inbounds for j in axes(x, 3), i in axes(x, 2)
        r[1, i, j] = (dx[1, i, j] - r[1, i, j]) / household.r
        if j == size(x, 3)
            # Upper obstacle: max(r*(q-c), q_s-fq)=0, in normalised units.
            obstacle = x[1, i, j] - c(firm) / parameters.qscale
            if FD.value(obstacle) ≥ FD.value(r[1, i, j])
                r[1, i, j] = obstacle
            end
            r[2, i, j], r[3, i, j] = x[2, i, j], x[3, i, j]
        else
            r[2, i, j] = (dx[2, i, j] - r[2, i, j]) / household.r
            r[3, i, j] = (dx[3, i, j] - r[3, i, j]) / household.r
        end
    end
    return nothing
end

"Conservative stencil, including couplings that vanish at a mimicking initial guess."
function ncdaejacobian(parameters)
    nφ, na = size(parameters.grid)
    rows, columns = Int[], Int[]
    for j in 1:na, i in 1:nφ
        node = i + (j - 1) * nφ
        neighbours = [node]
        if i > 1
            push!(neighbours, node - 1)
        end
        if i < nφ
            push!(neighbours, node + 1)
        end
        if j < na
            push!(neighbours, node + nφ)
        end
        for neighbour in neighbours, row in 1:4, column in 1:4
            push!(rows, 4node - 4 + row)
            push!(columns, 4neighbour - 4 + column)
        end
    end
    return SP.sparse(rows, columns, ones(typeof(parameters.qscale), length(rows)), 4nφ * na, 4nφ * na)
end

"Prepare the sparse combined Jacobian ∂ₓR + γ∂ₓ̇R for implicit time steps."
function ncdaejacobian(residual!, derivative, state, parameters, s, autodiff)
    buffer = similar(state)
    combined! = (residual, z, dx, x, parameters, γ, s) -> residual!(residual, dx .+ γ .* (z .- x), z, parameters, s)
    preparation = DI.prepare_jacobian(combined!, buffer, autodiff, state, DI.Constant(derivative), DI.Constant(state), DI.Constant(parameters), DI.Constant(one(s)), DI.Constant(s))
    jacobian! = function (jacobian, dx, x, parameters, γ, s)
        DI.jacobian!(combined!, buffer, jacobian, preparation, autodiff, x, DI.Constant(dx), DI.Constant(x), DI.Constant(parameters), DI.Constant(γ), DI.Constant(s))
        return nothing
    end
    return jacobian!
end

function ncstate(value::NCMarkovValue{T}, parameters) where T
    x = zeros(T, 4, size(parameters.grid)...)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    baseline = κ₀(firm.η, household, firm, climate) / household.r
    x[1, :, :] .= value.q ./ parameters.qscale
    x[2, :, :] .= (value.uᵍ .- baseline) ./ parameters.valuescale
    x[3, :, :] .= (value.u .- baseline) ./ parameters.valuescale
    x[4, :, :] .= value.τ ./ parameters.taxscale
    return vec(x)
end

function ncdaevalue(state, parameters, s; calendar = parameters.horizon - s)
    x = reshape(state, 4, size(parameters.grid)...)
    household, firm, climate = parameters.model.household, parameters.model.firm, parameters.model.climate
    baseline = κ₀(firm.η, household, firm, climate) / household.r
    @views value = NCMarkovValue(calendar, parameters.grid, parameters.qscale .* x[1, :, :], baseline .+ parameters.valuescale .* x[2, :, :], baseline .+ parameters.valuescale .* x[3, :, :], clamp.(parameters.taxscale .* x[4, :, :], 0, parameters.τᶜ(calendar)))

    return value
end

function ncdaeprojecttax!(state, parameters, s; interval = nothing)
    x = reshape(state, 4, size(parameters.grid)...)
    mandate = ncdaemandata(parameters, s, interval)

    @inbounds for j in axes(x, 3), i in axes(x, 2)
        candidate, _ = ncdaetax(x, parameters.taxscale * x[4, i, j], i, j, mandate, parameters; equilibrium = true)
        x[4, i, j] = candidate / parameters.taxscale
    end
end

function ncdaederivative(state, parameters, s; interval = nothing)
    derivative = similar(state)
    ncdaefield!(derivative, state, parameters, s; interval)
    x = reshape(state, 4, size(parameters.grid)...)
    dx = reshape(derivative, size(x))
    dx[4, :, :] .= 0
    @inbounds for i in axes(x, 2)
        if x[1, i, end] ≥ c(parameters.model.firm) / parameters.qscale
            dx[1, i, end] = min(dx[1, i, end], 0)
        end
    end

    return derivative
end

function ncdaediagnostics(state, derivative, parameters, s; interval = nothing, iterations = 0)
    residual = similar(state)
    ncdaeresidual!(residual, derivative, state, parameters, s; interval)
    r = reshape(residual, 4, size(parameters.grid)...)
    x = reshape(state, size(r))
    mandate = ncdaemandata(parameters, s, interval)
    improvement = zero(parameters.taxscale)
    @inbounds for j in axes(x, 3), i in axes(x, 2)
        τ = clamp(parameters.taxscale * x[4, i, j], 0, mandate)
        _, gain = ncdaetax(x, τ, i, j, mandate, parameters)
        improvement = max(improvement, gain)
    end
    return NonCommittedMarkovDiagnostics(iterations, maximum(abs, view(r, 1, :, :)), max(maximum(abs, view(r, 2, :, :)), maximum(abs, view(r, 3, :, :))), maximum(abs, view(r, 4, :, :)), improvement)
end

