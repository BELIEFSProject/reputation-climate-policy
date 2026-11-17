"Hidden best response, or largest consistent tax for projection, with conjectures held fixed."
function ncdaetax(x, τ, i, j, τᶜ, parameters; equilibrium = false)
    grid, government = parameters.grid, parameters.government
    φ, a = physicalstate(grid, i, j)
    if iszero(τᶜ)
        return zero(τ), zero(τ)
    end

    coefficient = φ * (1 - φ) * (parameters.signal.ϵ / parameters.signal.σ)^2
    B = coefficient * (τᶜ - τ)
    ∂ᵩu = zero(τ)
    if 1 < i < size(grid, 1) && a < parameters.model.firm.η
        Δφ = grid.h * span(grid.φ)
        ∂ᵩu = parameters.valuescale * (x[2, i + 1, j] - x[2, i - 1, j]) / (2Δφ)
    end

    κ, τ̄ = government.κᵇ, government.τ̄

    τ̂ = clamp(τ̄ - B * ∂ᵩu / κ, 0, τᶜ)
    if equilibrium
        uppergradient = κ * (τᶜ - τ̄)
        if uppergradient ≤ 0
            τ̂ = τᶜ
        else
            lowergradient = -κ * τ̄ + coefficient * τᶜ * ∂ᵩu
            gain = max(-lowergradient, 0)
            τ̂ = τᶜ * gain / (uppergradient + gain)
        end
    end
    Δh = max((τ - τ̂) * (-κ * τ̄ + B * ∂ᵩu + κ * (τ + τ̂) / 2), 0)

    return τ̂, Δh
end

