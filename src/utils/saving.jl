import SHA

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

function solutionlabel(household::AbstractHousehold, firm::Firm, signal::Signal, climate::Climate)
    join((solutionlabel(household, firm, climate), signallabel(signal)), "_")
end

"Short, stable filename determined by every committed-solution parameter."
function solutionfilename(household::AbstractHousehold, firm::Firm, climate::Climate)
    label = solutionlabel(household, firm, climate)
    digest = label |> SHA.sha256 |> bytes2hex
    return "solution-$digest.jld2"
end
