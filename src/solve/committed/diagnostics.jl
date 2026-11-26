"Independent discounted cash-flow integrals and welfare, including the stationary tail."
function committedvaluation(solution, scale, model; tolerance=1e-7)
    horizon = solution.prob.p.horizon
    r = model.household.r
    integrand = @closure s -> begin
        local u = SA.SVector{8}(solution(s)) .* scale
        local a = first(committedpolicy(model, u[1], u[3], u[4], u[2], u[6], u[7], u[8]; zguess=solution.prob.p.parameters.zguess))
        horizon * exp(-r * horizon * s) * SA.SVector(a.p, a.profit, cleanprofit(a.p, u[1], a.investment, model.clean), a.welfare)
    end
    integral = zero(SA.SVector{4, eltype(scale)})
    for j in 1:(length(solution.t) - 1)
        value, error = QuadGK.quadgk(integrand, solution.t[j], solution.t[j + 1]; atol=tolerance, rtol=tolerance)
        integral += value
    end
    terminal = SA.SVector{8}(solution(1)) .* scale
    tail = first(committedtail(model, terminal[1], terminal[3], terminal[2]))
    values = integral + exp(-r * horizon) * SA.SVector(terminal[2], terminal[3], terminal[4], tail.welfare / r)
    initial = SA.SVector{8}(solution(0)) .* scale
    residuals = (SA.SVector(values[1], values[2], values[3]) - SA.SVector(initial[2], initial[3], initial[4])) ./ SA.SVector(scale[2], scale[3], scale[4])

    values, residuals
end
