function parameterargumentsettings()
    settings = ArgParse.ArgParseSettings(description = "Model parameter overrides.")

    ArgParse.@add_arg_table! settings begin
        "--nu", "--ν"
            arg_type = Float64
            dest_name = "ν"
            default = nothing
            help = "Scale of labour disutility."
        "--inverse-frisch", "--varphi-L", "--φᴸ"
            arg_type = Float64
            dest_name = "ϕ"
            default = nothing
            help = "Inverse Frisch elasticity of labour supply."
        "--productivity", "--A"
            arg_type = Float64
            dest_name = "A"
            default = nothing
            help = "Total factor productivity."
        "--energy-share", "--alpha", "--α"
            arg_type = Float64
            dest_name = "α"
            default = nothing
            help = "CES energy share."
        "--substitution", "--rho", "--ρ"
            arg_type = Float64
            dest_name = "ρ"
            default = nothing
            help = "CES substitution parameter."
        "--discount", "--r"
            arg_type = Float64
            dest_name = "r"
            default = nothing
            help = "Discount rate."
        "--epsilon", "--eps", "--ϵ", "--ε"
            arg_type = Float64
            dest_name = "ϵ"
            default = nothing
            help = "Signal drift sensitivity."
        "--sigma", "--σ"
            arg_type = Float64
            dest_name = "σ"
            default = nothing
            help = "Signal volatility."
        "--gamma", "--γ"
            arg_type = Float64
            dest_name = "γ"
            default = nothing
            help = "Damage coefficient."
        "--preferred-tax"
            arg_type = Float64
            dest_name = "τ̄"
            default = nothing
            help = "Behaviourally preferred tax in USD/tCO2e."
        "--behavioural-curvature", "--kappa-b"
            arg_type = Float64
            dest_name = "κᵇ"
            default = nothing
            help = "Behavioural curvature in model tax and annual cost units."
    end

    return settings
end

function parseparameterarguments(args = ARGS)
    ArgParse.parse_args(args, parameterargumentsettings(); as_symbols = true)
end

function parameterkwargs(parsed, fields)
    kwargs = Dict{Symbol, Float64}()

    for field in fields
        value = get(parsed, field, nothing)

        if !isnothing(value)
            kwargs[field] = value
        end
    end

    return kwargs
end

initmodels(args = ARGS; calibration = Calibration()) = initmodels(parseparameterarguments(args); calibration)
initmodels(calibration::Calibration) = initmodels(String[]; calibration)

function initmodels(parsed::AbstractDict; calibration = Calibration())
    firmkwargs = parameterkwargs(parsed, (:A, :α, :ρ))
    firm = Firm(; calibration, firmkwargs...)
    householdkwargs = parameterkwargs(parsed, (:ν, :ϕ, :r))
    household = Household(; calibration, householdkwargs...)
    governmentkwargs = parameterkwargs(parsed, (:κᵇ, :τ̄))
    if haskey(governmentkwargs, :τ̄)
        governmentkwargs[:τ̄] *= calibration.taxfactor
    end
    government = Government(; calibration, governmentkwargs...)

    signal = Signal(; calibration, parameterkwargs(parsed, (:ϵ, :σ))...)
    climate = Climate(; calibration, parameterkwargs(parsed, (:γ, ))...)

    return household, firm, government, signal, climate
end
