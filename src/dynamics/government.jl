struct CommittedTaxPath{TI, T, TT}
    taxpath::TI
    ts::T
    tailtax::TT
end

CommittedTaxPath(taxpath, ts) = CommittedTaxPath(taxpath, ts, last(taxpath))

"Repeated times store the left and right tax at a jump; evaluation is right-continuous."
function (path::CommittedTaxPath)(t; side = :right)
    t < first(path.ts) && return first(path.taxpath)
    if side == :left
        i = searchsortedfirst(path.ts, t)
        i ≤ length(path.ts) && path.ts[i] == t && return path.taxpath[i]
    end
    t >= last(path.ts) && return path.tailtax

    i = searchsortedlast(path.ts, t)
    weight = (t - path.ts[i]) / (path.ts[i + 1] - path.ts[i])
    return (1 - weight) * path.taxpath[i] + weight * path.taxpath[i + 1]
end
