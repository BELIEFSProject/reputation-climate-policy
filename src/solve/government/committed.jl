## Optimal tax
committedtaxresidual(τ, (x, model)) = committedtaxresidual(τ, x, model)
function committedtaxresidual(τ, x, model::ModelParameters)
    @unpack household, firm, climate = model
    a, _, _, λᵨ = x

    labourwedge = emissivity(a, firm) * τ
    marginaldamages = 2climate.γ * e(n(τ, a, household, firm), a, firm)

    investment = τ + n(τ, a, household, firm) / ∂ₜn(τ, a, household, firm)

    return labourwedge + investment * household.r * λᵨ - marginaldamages
end
function committedtax(x::TX, model::ModelParameters) where {T, TX <: AbstractVector{T}}
    𝟎 = zero(x[1])

    ϵ = emissivity(x[1], model.firm)
    iszero(ϵ) && return 𝟎
    taxupperbound = (one(ϵ) - √eps(typeof(model.firm.η))) / model.firm.η
    foc = Base.Fix2(committedtaxresidual, (x, model))

    fl = foc(𝟎)
    fu = foc(taxupperbound)

    if fl ≥ 0
        return 𝟎
    elseif fu ≤ 0
        taxupperbound
    else
        return Roots.find_zero(foc, (𝟎, taxupperbound), Roots.Brent())
    end
end
function committedtax(x::TX, model::LinearLabourModelParameters) where {T, TX <: AbstractVector{T}}
    @unpack household, firm, climate = model
    a, _, _, λᵨ = x
    ϵ = emissivity(a, firm)
    if ϵ ≈ 0
        return zero(a)
    end

    num = (2climate.γ * firm.A^2 * ϵ / household.ν) + household.r * λᵨ / ϵ
    den = ϵ + 2household.r * λᵨ + 2climate.γ * firm.A^2 * ϵ^2 / household.ν

    return clamp(num / den, 0, 1 / firm.η)
end

function committedtail(a, model::LinearLabourModelParameters)
    @unpack household, firm, climate = model

    ϵ = emissivity(a, firm)
    ϵ′ = emissivity′(a, firm)
    levycoefficient = firm.A^2 / household.ν
    marginalvalue = household.r * c(a, firm)
    marginalvaluederivative = household.r * c′(a, firm)

    damagecoefficient = 2climate.γ * levycoefficient
    unconstrainedtax = clamp(damagecoefficient / (1 + damagecoefficient * ϵ), 0, 1 / firm.η)
    unconstrainedmarginalvalue = levycoefficient * unconstrainedtax * (1 - ϵ * unconstrainedtax)

    if unconstrainedmarginalvalue ≤ marginalvalue
        tax = unconstrainedtax
        taxderivative = -ϵ′ * tax^2
        marginalvalue = unconstrainedmarginalvalue
        marginalvaluederivative = levycoefficient * (taxderivative * (1 - 2ϵ * tax) - ϵ′ * tax^2)
    else
        discriminant = max(1 - 4ϵ * marginalvalue / levycoefficient, 0)
        tax = clamp(2marginalvalue / (levycoefficient * (1 + √discriminant)), 0, 1 / firm.η)
        taxderivative = (marginalvaluederivative + levycoefficient * ϵ′ * tax^2) / (levycoefficient * (1 - 2ϵ * tax))
    end

    labour = n(tax, a, household, firm)
    labouraderivative = ∂ₐn(tax, a, household, firm)
    labourtaxderivative = -firm.A * ϵ / household.ν
    welfarewedge = 2climate.γ * e(labour, a, firm) - ϵ * tax
    welfareaderivative = firm.A * welfarewedge * labouraderivative + climate.γ * (firm.A * labour)^2 * ϵ′
    welfaretaxderivative = firm.A * welfarewedge * labourtaxderivative
    valuegradient = (welfareaderivative + welfaretaxderivative * taxderivative) / household.r

    return tax, marginalvalue, marginalvaluederivative, valuegradient
end

## State-Costate system
initialguess(model) = initialguess(SA.SVector, model)
function initialguess(TV, model::LinearLabourModelParameters)
    @unpack firm = model

    _, marginalvalue, _, valuegradient = committedtail(firm.a₀, model)
    λᵨ₀ = zero(valuegradient)

    return TV(firm.a₀, marginalvalue, valuegradient, λᵨ₀)
end

function driftcommitted!(dx, x, model::AbstractModelParameters, _)
    @unpack household, firm, climate = model
    a, q, λₐ, λᵨ = x

    τᶜ = committedtax(x, model)
    labour = n(τᶜ, a, household, firm)
    labour′ = ∂ₐn(τᶜ, a, household, firm)

    isinvesting = q > household.r * c(a, firm) && a < firm.η

    da = isinvesting ? α(q, a, household, firm) : zero(a)
    dq = household.r * (q - firm.A * labour * τᶜ + firm.κ * da)

    λₐwedge = isinvesting ?
                (firm.κ / firm.ξ) * λₐ + (firm.κ^2 * a / firm.ξ) + household.r * λᵨ * firm.κ^2 / firm.ξ :
                zero(a)

    dλₐ = household.r * λₐ + climate.γ * (firm.A * labour)^2
            - firm.A * (2climate.γ * e(labour, a, firm) - emissivity(a, firm) * τᶜ) * labour′
            + household.r * λᵨ * (firm.A * τᶜ * labour′)
            + λₐwedge

    dλᵨ = isinvesting ?
            -firm.κ / firm.ξ * λᵨ - λₐ / (household.r * firm.ξ) - q / (household.r^2 * firm.ξ) :
            zero(a)

    dx[1] = da
    dx[2] = dq
    dx[3] = dλₐ
    dx[4] = dλᵨ

    return dx
end


## Outer Optimization
bvpbounds(model) = bvpbounds(SA.SVector, model)
function bvpbounds(TV, model::AbstractModelParameters; λmax = Inf)
    @unpack firm = model

    lb = TV(firm.a₀, zero(firm.a₀), -λmax, -λmax)
    ub = TV(firm.η, λmax, λmax, λmax)

    return lb, ub
end

function initialcondition(x₀::TX, model::LinearLabourModelParameters) where {T, TX <: AbstractVector{T}}
    initialcondition!(Vector{Float64}(undef, 2), x₀, model)
end
function initialcondition!(res, x₀, model::LinearLabourModelParameters)
    res[1] = x₀[1] - model.firm.a₀
    res[2] = x₀[4]

    return res
end

function terminalcondition(x̄::TX, model::LinearLabourModelParameters) where {T, TX <: AbstractVector{T}}
    terminalcondition!(Vector{Float64}(undef, 2), x̄, model)
end
function terminalcondition!(res, x̄, model::LinearLabourModelParameters)
    a, q, λₐ, λᵨ = x̄
    _, marginalvalue, marginalvaluederivative, valuegradient = committedtail(a, model)

    res[1] = q - marginalvalue
    res[2] = λₐ - valuegradient + λᵨ * marginalvaluederivative

    return res
end

"Drift of state co-state system with welfare as last variable"
function driftwelfarecommitted!(dz, z, model, t)
    # State co-state drift
    dx = @view dz[1:4]
    x = @view z[1:4]

    driftcommitted!(dx, x, model, t)

    # Welfare costs drift
    @unpack household, firm, climate = model
    a = x[1]
    ȧ = dx[1]
    τᶜ = committedtax(x, model)

    dw = exp(-household.r * t) * w(τᶜ, a, ȧ, household, firm, climate)
    dz[5] = dw

    return dz

end
