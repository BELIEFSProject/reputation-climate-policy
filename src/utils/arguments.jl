import ArgParse

function parameterargumentsettings()
    settings = ArgParse.ArgParseSettings(description = "Model parameter overrides.")

    ArgParse.@add_arg_table! settings begin
        "--e0", "--e₀"
            arg_type = Float64
            dest_name = "e₀"
            default = nothing
            help = "Baseline emissions."
        "--a0", "--a₀"
            arg_type = Float64
            dest_name = "a₀"
            default = nothing
            help = "Benchmark abatement."
        "--kappa", "--κ"
            arg_type = Float64
            dest_name = "κ"
            default = nothing
            help = "Marginal abatement cost slope."
        "--xi", "--ξ"
            arg_type = Float64
            dest_name = "ξ"
            default = nothing
            help = "Investment-rate adjustment cost."
        "--firm-discount", "--firm-r"
            arg_type = Float64
            dest_name = "firm_r"
            default = nothing
            help = "Firm discount rate."
        "--nu", "--ν"
            arg_type = Float64
            dest_name = "ν"
            default = nothing
            help = "Scale of labour disutility."
        "--inverse-frisch", "--varphi-L", "--φᴸ"
            arg_type = Float64
            dest_name = "φᴸ"
            default = nothing
            help = "Inverse Frisch elasticity of labour supply."
        "--productivity", "--A"
            arg_type = Float64
            dest_name = "A"
            default = nothing
            help = "Total factor productivity."
        "--emissions-intensity", "--eta-E", "--ηᴱ"
            arg_type = Float64
            dest_name = "ηᴱ"
            default = nothing
            help = "Emissions intensity of output."
        "--discount", "--r"
            arg_type = Float64
            dest_name = "r"
            default = nothing
            help = "Discount rate."
        "--delta", "--δ"
            arg_type = Float64
            dest_name = "δ"
            default = nothing
            help = "Political cost of taxation coefficient."
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
        "--zeta", "--ζ"
            arg_type = Float64
            dest_name = "ζ"
            default = nothing
            help = "TCRE."
        "--m0", "--m₀"
            arg_type = Float64
            dest_name = "m₀"
            default = nothing
            help = "Initial cumulative emissions."
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

function initmodels(args = ARGS)
    parsed = parseparameterarguments(args)
    
    firmkwargs = parameterkwargs(parsed, (:e₀, :a₀, :A, :ηᴱ, :κ, :ξ))
    firm = Firm(; firmkwargs...)
    householdkwargs = parameterkwargs(parsed, (:ν, :φᴸ, :r))
    household = Household(; householdkwargs...)

    signal = Signal(; parameterkwargs(parsed, (:ϵ, :σ))...)
    climate = Climate(; parameterkwargs(parsed, (:γ, ))...)

    return household, firm, signal, climate
end
