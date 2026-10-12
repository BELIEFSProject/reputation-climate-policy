function w(τ, a, ȧ, household::AbstractHousehold, firm::Firm, climate::Climate)
    labour = n(τ, a, household, firm)
    damages = d(e(labour, a, firm), climate)

    return k(a, ȧ, firm) + v(labour, household) - y(labour, firm) * (1 - damages)
end

function statictax(a, household::AbstractHousehold, firm::Firm, climate::Climate)
    res = Optim.optimize(τ -> w(τ, a, 0, household, firm, climate), 0, inv(firm.η), Optim.Brent())

    return Optim.minimizer(res)
end