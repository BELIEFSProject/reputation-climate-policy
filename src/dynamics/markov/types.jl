"Simulation state: log odds, installed abatement and cumulative emissions."
mutable struct MarkovState{T} <: SA.FieldVector{3, T}
    z::T
    a::T
    m::T
end

"Physical observables and the signal separation χ."
struct MarkovObservables{T} <: SA.FieldVector{10, T}
    φ::T
    τ::T
    τᵉ::T
    a::T
    q::T
    α::T
    n::T
    y::T
    e::T
    χ::T
end

struct MarkovParameters{S, F, T}
    solution::S
    fractions::F
    committed::Bool
    stop::T
end
