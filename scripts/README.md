Run the committed benchmark first, then the non-committed Markov equilibrium:

```powershell
julia --project=. scripts/committed.jl
julia --project=. scripts/noncommitted.jl
```

The committed BVP materialises its existing initial-guess path on the mesh before initialisation. This avoids an installed SciMLBase `remake` bug that drops `singular_term` for a function-valued guess. The solver checks that its cache retains the singular matrix and uses finite-difference Jacobians around the bounded tax rule. A failed BVP stops before results can be saved.

Committed cache identities remain `committed-full-interval-bvp-v3`; non-committed identities now use `behavioural-v1`. Earlier non-committed objectives are excluded from current equilibrium selection. Reuse a current committed benchmark when changing only behavioural preferences, then recompute the non-committed equilibria and figures. Committed results predating `v3` require recomputation because they omitted the singular costate equation.

The default labour supply is unit elastic, corresponding to `ϕ = 1`. The normalisation remains `ν = A`, giving zero-tax labour equal to one and output equal to `y₀`; the damage coefficient is unchanged. Both solvers evaluate the full welfare flow for nonlinear labour supply. The policy solvers currently support inverse Frisch elasticities `1` and `2` only. Use `--inverse-frisch=2` for a Frisch elasticity of `0.5`, recomputing the committed benchmark and non-committed equilibrium with the same option. Saved-solution identities include the labour elasticity, so these calibrations are kept separate.

The simplified investment-cost calibration uses two illustrative targets: a 24-year transition from 2026 to 2050, and expenditure `k((η-a₀)/24)/y₀ = 0.015` along a constant-progress reference path. The [European Commission's 2040 assessment](https://ec.europa.eu/commission/presscorner/api/files/document/print/en/qanda_24_589/QANDA_24_589_EN.pdf) reports additional annual energy-system investment of about 1.5% of GDP relative to 2011–2020. This motivates the scale only: gross investment, replacement expenditure and fuel savings do not identify the net resource cost in this model. Matching the policy date is also a calibration choice, rather than an independently estimated technological moment.

Ignoring labour wedges, let `b = γ*y₀²/r - c`. Optimal investment satisfies `ȧ(t) = b/ξ * (1-exp(-r*(H-t)))` until full abatement, giving `η-a₀ = b/ξ * (H-(1-exp(-r*H))/r)`. Solving this relation jointly with the expenditure target determines `c ≈ 56.15` and `ξ ≈ 581.12`, with `κ = c/a₀ ≈ 2391.54` retaining the existing `Firm` representation. The permanent supporting price is about `$36.56/tCO₂e`. The full BVP retains labour distortions and chooses its own stopping time, approximately 24 years; the calibration target is not a terminal-date constraint. The damage coefficient and discount rate are unchanged. Parameter overrides do not refit the two moments automatically. Changing the defaults changes saved-solution identities; recompute both equilibria.

The non-committed government has a signed behavioural cost `κᵇ/2 * [(τ̂-τ̄)²-(τ*-τ̄)²]`, where `τ̂` is the hidden action and `τ*` is firms' conjectured policy. The conjecture, expected tax `τᵉ = φ*τᶜ+(1-φ)*τ*`, investment and filtering gain remain fixed during a deviation. GE resource costs are evaluated at `τᵉ` under both government types. The behavioural term is zero when `τ̂=τ*` and is excluded from household welfare. It is signed: moving towards the preferred tax can give a private benefit. This is a reference-dependent preference assumption, not a non-negative adjustment penalty or a GE resource cost.

The hidden first-order condition is `κᵇ*(τ*-τ̄) + φ*(1-φ)*(ϵ/σ)²*(τᶜ-τ*)*∂φuᵍ = 0`, subject to `0 ≤ τ* ≤ τᶜ`. The implementation solves its consistent interior or corner policy explicitly and checks deviations with the filter frozen. With valuable reputation, the tax is a weighted average of the preferred tax and the announcement. At either belief endpoint, or at full abatement where continuation resource costs are belief independent, the policy is `min(τ̄,τᶜ)`. The positive preferred tax replaces the previous zero-rate capacity convention; tax payments remain zero at capacity. Government and household equilibrium values coincide, although hidden deviations give the government the additional behavioural motive.

The committed benchmark retains the welfare Ramsey objective. Its publicly announced policy is taken as its behavioural reference, so the behavioural term vanishes along any credibly announced plan. In the hidden game, unilateral deviations cannot reset that reference. This convention is part of the preference specification; it is not justified by subtracting an arbitrary state-dependent term from an ordinary control problem.

The illustrative defaults are `τ̄ = 10 USD/tCO₂e` and `κᵇ ≈ 0.0658142857143` in model units. A hidden cut from `50` to `40 USD/tCO₂e`, holding the conjecture at `50`, gives a private utility gain equal to `10⁻⁶*y₀`, or about `23.035 million USD-equivalent/year`. The coefficient follows directly from `κᵇ = 10⁻⁶*y₀/[Δτ*(τreference-τ̄-Δτ/2)]`. This is a preference normalisation, not an empirical estimate or an expenditure target. It avoids fitting a policy gap that depends on equilibrium discretisation. Increasing `κᵇ` strengthens the incentive to cut towards `τ̄`; it does not penalise the size of a hidden deviation from `τ*`.

Run `julia --project=. scripts/behavioural.jl` to reproduce this calculation, or supply the usual primitive overrides to calculate its implied coefficient for another output level. The script prints the coefficient without changing defaults or saved results. The Ramsey technology and damage calibration are unchanged.

With the current saved Ramsey announcement, backward solves at `h=1/40` and `h=1/80`, time step `0.05` and requested tolerances `1e-6` converged. At `φ=0.5`, calendar time `12.1761` and the Ramsey abatement stock at that date (`a/η≈0.7673`), the non-committed taxes were approximately `14.0642` and `14.0629 USD/tCO₂e`, against an announcement of `50.3116`. The finest run's maximum normalised residual was `9.9e-8`; government and household equilibrium cost values agreed. These are conditional policies at a common reference state, not simulated mean paths or a general mesh-convergence guarantee. The diagnostic runs did not overwrite saved equilibria.

Use `--preferred-tax=10` to specify the preferred tax in USD/tCO₂e and `--behavioural-curvature` to override `κᵇ` in model units. These parameters do not change physical primitives or require a new committed announcement. Government calibration is included in non-committed cache identities and plot-run identities. Older non-committed files are preserved but excluded by the `behavioural-v1` marker; recompute the non-committed equilibria before plotting.

The stationary tail starts with a larger behavioural curvature, using the initial static welfare curvature only to choose a numerical starting scale, then reduces it geometrically until reaching the requested preference. The final model uses the requested curvature and signal throughout the backward induction. Its Jacobian uses the same sparse stencil and differentiation interface as the backward steps. Internally the two NPV costs are centred at the constant zero-emissions continuation value, preserving precision in the small belief derivatives. Extracted and saved values include this constant and remain ordinary NPVs. No discount, grid, time-step or residual tolerance is changed by this centring.

Both scripts use the current primitives in `src/` and preserve existing JLD2 results. Model overrides can be supplied through the options defined in `src/utils/arguments.jl`. When changing household, firm or climate parameters, recompute the committed benchmark with the same overrides before solving the non-committed problem. To vary only signal noise, reuse the announcement:

```powershell
julia --project=. scripts/noncommitted.jl --sigma=0.76
```

The non-committed script accepts `--h` for the common grid step in normalised beliefs and abatement, defaulting to `1/80`. Supply a decimal whose reciprocal is an integer; for example, `julia --project=. --threads=4 scripts/noncommitted.jl --h=0.025` uses `h = 1/40`. The solver retains its existing grid validation, and saved-solution identities distinguish grid resolutions. Changing `h` does not require recomputing the committed announcement.

Use `--timestep` to change the maximum backward implicit step in calendar years, defaulting to `0.05`. For a convergence check, compare `--h=0.025 --timestep=0.05`, `--h=0.0125 --timestep=0.05` and `--h=0.0125 --timestep=0.025`. Each numerical configuration has a separate saved-solution identity; reuse the same committed announcement. The commitment plotting script exports an abatement and valuation benchmark to assess these refinements.

The script sets `tolerance = policytolerance = 1e-6` for the normalised residuals. The stationary tail uses the requested maximum residual tolerance `min(tolerance,policytolerance)`; backward Newton steps use `min(tolerance,policytolerance)/10 = 1e-7`. Both use the maximum absolute residual, matching the saved diagnostics, rather than a dimension-dependent Euclidean norm. Failure checks remain active, and the saved-solution identities include these tolerances. These are nonlinear-solve tolerances, not bounds on time or state discretisation error.

Implicit steps use Newton with backtracking. The line search explicitly uses a forward-mode Jacobian-vector product; its default analytic-Jacobian helper would otherwise allocate a dense square work array despite the main sparse Jacobian. If Newton fails, the same equations are retried from its best iterate using a trust region, at the same tolerance and with a fresh sparse Jacobian at every retry iteration. A failed retry still stops the solve. The iteration log includes both attempts.

The committed solver is `src/solve/government/committed.jl`. The non-committed implementation is in `src/solve/government/nc/`, with its dependencies explicitly listed by each script. The unused legacy temporal-scheme parameter struct has been removed.

Non-committed function and variable names use `nc`: the solver entry point is `solvencdae`, the script result is `ncsolution`, and saved-solution keys are computed by `ncsolutionkey`. Filenames, JLD2 group names and existing type names are retained for compatibility with saved results.

Generate figures from saved equilibria with:

```powershell
julia --project=. scripts/plotting/reputation.jl
julia --project=. scripts/plotting/commitment.jl
julia --project=. scripts/plotting/dynamic.jl
```

Use `julia --project=. scripts/plotting/calibration.jl` for figures of the current calibration alone; no saved equilibrium is required.

See `scripts/plotting/README.md` for the experiments, saved-equilibrium selection and figure outputs.

## Shared implementation

```text
src/solve/government/nc/
    grid.jl                         state grid and solution representations
    parameters.jl                   numerical scales
    policy.jl                       hidden Hamiltonian and tax rule
    operator.jl                     generator, residual and Jacobian
    solve.jl                        implicit backward solve
src/dynamics/
    ramsey.jl                       Ramsey observables shared by plots
    markov/types.jl                  named FieldVector states and observables
    markov/observables.jl            observables, μ! and σ!
    markov/ensemble.jl               stochastic ensemble simulation
scripts/plotting/common/
    results.jl                      projection result representations
    loading.jl                      common equilibrium selection
    projection.jl                   forward projection maths
```

Each script lists its dependencies directly using paths relative to its own directory: for example, `includet("../src/utils/arguments.jl")` in `scripts/` and `includet("../../src/utils/arguments.jl")` in `scripts/plotting/`. Tests likewise use file-relative `include` paths. Include-only entry points remain removed; include paths use neither `abspath` nor `@__DIR__`. Parameter-range prechecks have been removed; genuine solver-failure reporting, economic control constraints and the singular-term check remain. Unsupported committed closed forms still report their elasticity restriction. Saved equilibrium types remain compatible; the unused legacy temporal-scheme type and the simulation parameter wrapper have been removed. Dynamic plots rerun ensembles rather than caching paths. Existing data files are left untouched.

Stochastic states use `MarkovState <: StaticArrays.FieldVector` with fields `z`, `a`, `m`. Observables use `MarkovObservables <: FieldVector`, retaining both named fields and the existing column indexing. Heterogeneous simulation parameters use `MarkovParameters` with named fields rather than positional tuple entries.

## Calibration objects and package imports

All package imports belong to the calling scripts or tests. Files under `src/` define the model and numerical methods without importing packages.

`src/primitives/calibration.jl` replaces the former global constants with one immutable `Calibration` object:

```julia
calibration = Calibration(ξ = 3, σ = 0.19)
household, firm, government, signal, climate = initmodels(calibration)
```

Individual primitives also accept the object, for example `Firm(calibration)` and `Household(calibration)`. In a research script, edit its `calibration = Calibration()` line; `initmodels(; calibration)` additionally applies command-line overrides. CLI preferred-tax values remain in USD/tCO₂e, while `Calibration(τ̄ = ...)` uses model units, as do the primitive constructors.

Dependent defaults are recomputed when constructing a calibration: changing `y₀` updates `A`, `ν`, emissions intensity and the cost calibration. Explicit keyword values override those defaults. CLI overrides retain the existing behaviour of overriding individual primitives without refitting the other calibration targets. The primitive and saved-solution representations are unchanged, so existing model files and their parameter-based identities remain usable.
