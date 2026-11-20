# x[t, h] = τ[t] + ε[t, h], with independent ε[t, h] ~ N(0, observationvariance).
# τ[t] = τ[t-1] + η[t], with η[t] ~ N(0, processvariance * elapsed[t]).
# Process variance is per calendar day; all horizons have unit loading and no term premium.
struct HorizonLocalLevel
    τ::Vector{Float64}
    variance::Vector{Float64}
    noise::Matrix{Float64}
    processvariance::Float64
    observationvariance::Float64
    loglikelihood::Float64
end

function filterlocallevel!(τpath, variancepath, prices, observedhorizons, elapsed, processvariance, observationvariance)
    τ = 0.0
    variance = 0.0
    initialised = false
    loglikelihood = 0.0
    for i in axes(prices, 1)
        initialised && (variance += processvariance * elapsed[i])
        for h in observedhorizons[i]
            price = prices[i, h]
            if !initialised
                # Exact diffuse initialisation: the first observation fixes the level's origin.
                τ = price
                variance = observationvariance
                initialised = true
                continue
            end
            innovation = price - τ
            innovationvariance = variance + observationvariance
            loglikelihood -= (log(2pi) + log(innovationvariance) + innovation^2 / innovationvariance) / 2
            gain = variance / innovationvariance
            τ += gain * innovation
            variance *= observationvariance / innovationvariance
        end
        if !isempty(τpath)
            τpath[i] = initialised ? τ : NaN
            variancepath[i] = initialised ? variance : NaN
        end
    end
    return loglikelihood
end

function fithorizonlocallevel(prices, dates)
    size(prices, 1) == length(dates) || throw(DimensionMismatch("Price rows must match trading dates."))
    elapsed = vcat(0.0, Float64.(Dates.value.(diff(dates))))
    all(>(0), elapsed[2:end]) || error("Trading dates must be strictly increasing.")
    observedhorizons = [findall(isfinite, row) for row in eachrow(prices)]
    observeddates = findall(horizons -> !isempty(horizons), observedhorizons)
    length(observeddates) >= 3 || error("The local-level model needs at least three observed trading dates.")
    any(horizons -> length(horizons) >= 2, observedhorizons) || error("Overlapping horizons are needed to distinguish common shocks from observation noise.")

    dailymeans = [Statistics.mean(prices[i, observedhorizons[i]]) for i in observeddates]
    meanelapsed = Statistics.mean(Dates.value.(diff(dates[observeddates])))
    initialprocessvariance = max(Statistics.var(diff(dailymeans)) / meanelapsed, 1e-8)
    dispersion = [Statistics.var(prices[i, observedhorizons[i]]) for i in observeddates if length(observedhorizons[i]) >= 2]
    initialobservationvariance = max(Statistics.mean(dispersion), 1e-8)
    emptypath = Float64[]

    function objective(logvariances)
        processvariance, observationvariance = exp.(logvariances)
        all(x -> isfinite(x) && x > 0, (processvariance, observationvariance)) || return Inf
        likelihood = filterlocallevel!(emptypath, emptypath, prices, observedhorizons, elapsed, processvariance, observationvariance)
        return isfinite(likelihood) ? -likelihood : Inf
    end

    initialguess = log.([initialprocessvariance, initialobservationvariance])
    optimum = Optim.optimize(objective, initialguess, Optim.NelderMead(), Optim.Options(iterations = 2000))
    Optim.converged(optimum) || error("Local-level variance estimation failed to converge: $optimum")
    processvariance, observationvariance = exp.(Optim.minimizer(optimum))
    τ = fill(NaN, length(dates))
    variance = similar(τ)
    loglikelihood = filterlocallevel!(τ, variance, prices, observedhorizons, elapsed, processvariance, observationvariance)
    isfinite(loglikelihood) || error("The fitted local-level likelihood is not finite.")
    # These are filtered observation residuals, rather than directly observed shocks.
    noise = prices .- τ
    return HorizonLocalLevel(τ, variance, noise, processvariance, observationvariance, loglikelihood)
end
