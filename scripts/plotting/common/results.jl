struct ReputationEquilibrium{S}
    solution::S
    filename::String
end

struct ReputationProjection{V, M, R, T}
    t::V
    belief::M
    tax::M
    expectedtax::M
    investment::M
    abatement::M
    emissions::M
    cumulativeemissions::M
    nearzero::M
    regimes::R
    taxwedge::R
    learningvariance::M
    welfareloss::T
end

struct ReputationTiming{V, T}
    t::V
    progress::V
    emissions::V
    terminalabatement::V
    terminalemissions::V
    cumulativeemissions::V
    baselineabatement::T
    baselineemissions::T
    baselinecumulativeemissions::T
    stationary::Bool
end

