Run the committed benchmark first, then the non-committed Markov equilibrium:

```powershell
julia --project=. scripts/committed.jl
julia --project=. scripts/noncommitted.jl
```

The benchmark Frisch elasticity is `0.5`, corresponding to `ϕ = 2`. The normalisation remains `ν = A`, giving zero-tax labour equal to one and output equal to `y₀`; the damage coefficient is unchanged. Both solvers evaluate the full welfare flow for nonlinear labour supply. The policy solvers currently support inverse Frisch elasticities `1` and `2` only. Use `--inverse-frisch=1` for the previous unit-elasticity case, recomputing the committed benchmark and non-committed equilibrium with the same option. Saved-solution identities include the labour elasticity, so these calibrations are kept separate.

The EU investment-cost defaults preserve the original global cost/GDP calibration for the same initial abatement fraction and progress path `a/η`. The reference cost scale is `3e-5 * 41.9^2 = 0.0526683`, transferred from the original global gross emissions of 41.9 Gt/year. Consequently, `κ = 0.0526683*y₀/(r*η²)` and `ξ = 0.0526683*y₀*H²/((1+r*H)*η²)`, with the existing adjustment parameter `H = 30` years. This is a regional rescaling of the previous cost calibration, not a new empirical estimate for EU technology costs. At the current EU quantities, `κ ≈ 2909.19`, `ξ ≈ 27085.54` and the supporting price `r*c/y₀ ≈ $44.48/tCO₂e`. Changing these defaults changes the saved-solution identities; recompute both equilibria. The adjustment parameter does not fix the realised transition duration.

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

Use `julia --project=. scripts/plotting/calibration.jl` for figures of the current calibration alone; no saved equilibrium is required.

See `scripts/plotting/README.md` for the experiments, saved-equilibrium selection and figure outputs.
