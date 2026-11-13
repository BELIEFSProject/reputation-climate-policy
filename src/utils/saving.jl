import SHA

const solutionhashlength = 12
const ncversion = "behavioural-v1"

function parameterstring(x)
    # `string` uses Julia's shortest round-trippable representation for floats, so distinct parameter values are not collapsed by display rounding.
    replace(string(x), "+" => "")
end

function dynamicsolutionlabel(household::AbstractHousehold, firm::Firm)
    ϕlabel = isa(household, LinearHousehold) ? 1 : household.ϕ

    join((
        "nu$(parameterstring(household.ν))",
        "varphiL$(parameterstring(ϕlabel))",
        "householddiscount$(parameterstring(household.r))",
        "e0$(parameterstring(firm.e₀))",
        "a0$(parameterstring(firm.a₀))",
        "A$(parameterstring(firm.A))",
        "etaE$(parameterstring(firm.η))",
        "kappa$(parameterstring(firm.κ))",
        "xi$(parameterstring(firm.ξ))",
        "firmdiscount$(parameterstring(household.r))",
    ), "_")
end

function solutionlabel(household::AbstractHousehold, firm::Firm, climate::Climate)
    join((
        dynamicsolutionlabel(household, firm),
        "r$(parameterstring(household.r))",
        "gamma$(parameterstring(climate.γ))",
    ), "_")
end

function signallabel(signal::Signal)
    join((
        "epsilon$(parameterstring(signal.ϵ))",
        "sigma$(parameterstring(signal.σ))",
    ), "_")
end

governmentlabel(government::Government) = "kappab$(parameterstring(government.κᵇ))_preferredtax$(parameterstring(government.τ̄))"

function solutionlabel(household::AbstractHousehold, firm::Firm, signal::Signal, climate::Climate)
    join((solutionlabel(household, firm, climate), signallabel(signal)), "_")
end

function committedsolutionfilename(model; initialtax = nothing, intervals = 100)
    label = join(("committed-full-interval-bvp-v3", solutionlabel(model.household, model.firm, model.climate), parameterstring(c(model.firm)), parameterstring(initialtax), string(intervals)), "_")
    return "committed-simplified-$(first(bytes2hex(SHA.sha256(label)), solutionhashlength)).jld2"
end

function ncsolutionkey(solution)
    parameters = solution.parameters
    model, path = parameters.model, parameters.τᶜ
    settings = [string(field, "=", parameterstring(getfield(parameters, field))) for field in fieldnames(typeof(parameters)) if getfield(parameters, field) isa Real]
    label = join((ncversion, string(nameof(typeof(parameters))), solutionlabel(model.household, model.firm, parameters.signal, model.climate), governmentlabel(parameters.government), parameterstring(c(model.firm)), parameterstring(parameters.grid.h), parameterstring(parameters.grid.φ.lower), parameterstring(parameters.grid.φ.upper), parameterstring(parameters.grid.A.lower), parameterstring(parameters.grid.A.upper), join(settings, ","), join(parameterstring.(solution.t), ","), join(parameterstring.(path.ts), ","), join(parameterstring.(path.taxpath), ","), parameterstring(path.tailtax)), "_")
    return "noncommitted/$(bytes2hex(SHA.sha256(label)))"
end
