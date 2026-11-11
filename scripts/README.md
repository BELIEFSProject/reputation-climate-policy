Run the committed benchmark first, then the non-committed Markov equilibrium:

```powershell
julia --project=. scripts/committed.jl
julia --project=. scripts/noncommitted.jl
```

Both scripts use the current primitives in `src/` and preserve existing JLD2 results. Model overrides can be supplied through the options defined in `src/utils/arguments.jl`. When changing household, firm or climate parameters, recompute the committed benchmark with the same overrides before solving the non-committed problem. To vary only signal noise, reuse the announcement:

```powershell
julia --project=. scripts/noncommitted.jl --sigma=0.76
```

The committed solver is `src/solve/government/committed.jl`. The current non-committed solver is `src/solve/government/noncommitteddae.jl`; `noncommittedmarkov.jl` supplies its shared grid, value representation, filtering moments and initial values. Legacy parameter structs are retained to read saved solutions; their inactive fields are not calibration options.

Generate figures from saved equilibria with:

```powershell
julia --project=. scripts/plotting/reputation.jl
julia --project=. scripts/plotting/commitment.jl
julia --project=. scripts/plotting/dynamic.jl
```

See `scripts/plotting/README.md` for the experiments, saved-equilibrium selection and figure outputs.
