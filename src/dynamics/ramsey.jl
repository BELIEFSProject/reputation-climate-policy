"Ramsey observables on calendar dates; columns match the Markov simulations."
function ramsey(benchmark, τᶜ, ts)
    household, firm = benchmark.model.household, benchmark.model.firm

    benchmarkvalues = zeros(eltype(ts), length(ts), 10)
    for (row, t) in enumerate(ts)
        if t < benchmark.stoppingtime
            index = min(searchsortedlast(benchmark.t, t), length(benchmark.t) - 1)
            weight = (t - benchmark.t[index]) / (benchmark.t[index + 1] - benchmark.t[index])
            a = clamp((1 - weight) * benchmark.a[index] + weight * benchmark.a[index + 1], firm.a₀, firm.η)
            q = (1 - weight) * benchmark.q[index] + weight * benchmark.q[index + 1]
        else
            a, q = firm.η, c(firm)
        end
        side = row < length(ts) && ts[row + 1] == t ? :left : :right
        τ = τᶜ(t; side)
        labour = n(τ, a, household, firm)
        emissions = e(labour, a, firm)
        cumulative = row == 1 ? 0 : benchmarkvalues[row - 1, 10] + (t - ts[row - 1]) * (benchmarkvalues[row - 1, 9] + emissions) / 2
        benchmarkvalues[row, :] .= (1, τ, τ, a, q, a ≥ firm.η ? 0 : α(q, firm), labour, y(labour, firm), emissions, cumulative)
    end

    return benchmarkvalues
end
