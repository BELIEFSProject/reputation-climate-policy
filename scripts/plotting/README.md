The plotting scripts are `calibration.jl` for the primitives and `reputation.jl`, `commitment.jl` and `dynamic.jl` for the current Markov solution.

## Calibration

Run `julia --project=. scripts/plotting/calibration.jl` to illustrate the current primitives without loading or solving an equilibrium. The script accepts the same parameter overrides as the solver scripts. Seven separate 16:9 PDF/PNG figures are saved under `PLOTPATH/calibration-<12-character hash>/`, using the publication theme and `#FAFAFA` background:

- `labour`: tax responses at initial abatement, comparing the baseline Frisch elasticity with 0.5 and 1.
- `emissions`: emissions across installed abatement at three fixed tax rates, with the initial emissions reference.
- `welfare`: labour distortion, climate damages and their combined change relative to zero tax, holding abatement fixed and investment at zero.
- `prices`: the preferred static tax, pre-contact Ramsey price and lowest permanent supporting price. The static rule excludes the capacity endpoint, where the zero-rate convention would otherwise produce a misleading drop.
- `costs`: installation and adjustment costs as shares of zero-tax output against annual abatement progress.
- `investment`: the private investment response to `q/c`, including the irreversible-investment threshold.
- `learning`: the time for a unit signal-to-noise ratio at fixed tax gaps and three noise levels, `t = [σ/(ϵ*Δτ)]²`. This is an information diagnostic, not a posterior hitting time or an equilibrium forecast.

The 24-year constant-progress reference in the cost and investment figures is illustrative; it does not impose a stopping time or change the adjustment-cost calibration. `parameters.csv` records the primitives and derived reference quantities. Each calibration has its own figure folder; saved numerical solutions are untouched.

## Stochastic dynamics of the current Markov solution

Run `julia --project=. scripts/plotting/dynamic.jl` for Monte Carlo paths under the saved Markov equilibrium. The script selects the current baseline calibration, with the same `REPUTATION_SOLUTION` override as the reputation script, and compares every otherwise matching saved noise equilibrium. It does not solve new equilibria or change their calibration.

The configuration selects a fixed initial belief `φ₀ = 0.5`, `npaths = 500`, simulation step `Δt = 0.01` years and saved spacing `saveinterval = 0.25` years. Both experiments use the same Gaussian innovations across noise levels. The `uniform` experiment also uses the same initial draws across noise levels. All trajectories condition on a non-committed government by default; set `committed = true` to condition on commitment instead. Uniform sampling describes heterogeneity in initial public beliefs, rather than sampling government types.

The simulated states are log odds, abatement and cumulative emissions. For `z = log(φ/(1−φ))` and `χ = (ϵ/σ)(τᶜ−τⁿᶜ)`, the physical belief process is `dz = −χ² dt/2 + χ dW` under non-commitment and `dz = χ² dt/2 + χ dW` under commitment. Recovering beliefs with the logistic function preserves their bounds. Abatement follows `da = α(q) dt`, subject to capacity, and cumulative emissions integrate realised emissions. Labour uses the equilibrium expected tax under either true type. Cumulative emissions are a passive outcome, not an additional equilibrium state.

Pointwise interpolation uses the saved valuation grids and tax fractions relative to the current announcement. Interpolating fractions preserves pooling between announcement knots. The last interior tax is extended through the final abatement cell, and the zero-rate capacity convention is applied only at full abatement; this avoids smoothing the tax discontinuity into positive-emissions states. Announcement jumps split the integration intervals and are plotted with both one-sided tax limits and continuous states. These are simulations of the physical diffusion, without the artificial diffusion and abatement jumps of the numerical Markov chain, so their averages need not equal the existing forward-chain projections on a finite grid.

Twenty separate 16:9 PDF/PNG figures are written under `PLOTPATH/dynamic-<12-character hash>/{fixed,uniform}/`: policy, expected policy, belief, abatement, investment valuation, investment, labour, output, emissions and cumulative emissions. Each overlays noise levels with median paths, pointwise 10–90% bands and six faint sample paths, on the `#FAFAFA` background. Bands in the uniform experiment include both initial heterogeneity and subsequent learning. Exact pooling has no learning, so fixed-prior bands may collapse; no additional reputational shock is imposed.

The physical paths and initial draws are cached under `DATAPATH/simulations/dynamic-<hash>.jld2`, with the source equilibria and simulation settings. Existing matching caches are read without replacement. Changing the simulation settings produces a new cache; changing only plotting options reuses it. `summary.csv` alongside the figures contains the plotted quantiles in the axis units. The simulation step and trajectory count should be varied before interpreting small differences as economic effects.

The script logs the initial and maximum cross-trajectory belief standard deviations and maximum physical belief volatility for each experiment and noise level. Here `σφ = (ϵ/σ)φ(1−φ)(τᶜ−τⁿᶜ)`: increasing observation noise `σ` does not directly add posterior volatility. Pooling gives zero learning, and near-pooling can produce bands too narrow to see on the full state-variable axes. Uniform initial beliefs still have dispersion even without subsequent learning.

The script produces the fixed-prior and uniform-prior figures across matching saved noise equilibria in one run:

```powershell
julia --project=. scripts/plotting/dynamic.jl
```

## Reputation and the current Markov solution

Run `julia --project=. scripts/plotting/reputation.jl` after saving a solution with `scripts/noncommitted.jl`. The script reads saved equilibria; it does not solve new ones or overwrite numerical results. It selects the most recently saved current equilibrium matching the calibration and baseline signal noise. To select a particular file in PowerShell:

```powershell
$env:REPUTATION_SOLUTION = "C:\path\to\noncommitted-simplified-73318941b311.jld2"
julia --project=. scripts/plotting/reputation.jl
```

Run `julia --project=. scripts/plotting/commitment.jl` for the commitment comparisons alone. It uses the same saved-equilibrium selection and `REPUTATION_SOLUTION` override, and writes the figures and `paths.csv` in the `commitment/` subdirectory.

Solution filenames and corresponding figure-folder names use the first 12 hexadecimal hash characters. The full internal non-committed cache keys are retained. Existing solution files and figure folders have been moved to the short names.

Each plot is exported separately as a 16:9 PDF and a 1920×1080 PNG under `PLOTPATH/<solution-name>/reputation/`, with a `#FAFAFA` figure, axis and legend background for Moloch slides. The current configuration uses a 50% initial reputation and unexpected public news near year 10 that multiplies the odds of commitment by 0.1. These choices, the forward projection step and the near-zero threshold are declared at the top of the script. The model calibration and equilibrium selection are taken from the saved solution.

- `reputation-mechanism-{policy,investment-value,investment}`: actual and expected policy, investment value and investment as reputation varies, holding calendar time and abatement fixed. The abatement stock is the grid point nearest the baseline mean immediately before the news.
- `reputation-transition-{belief,policy,abatement,emissions}`: expected reputation, implemented policy, abatement and cumulative emissions with and without the news. The two cases have the same distribution before the intervention. Repeated news dates draw the instantaneous posterior update explicitly.
- `commitment/{policy,expected-policy,belief,abatement,valuation,investment,labour,output,emissions,cumulative-emissions}`: compare known commitment with type-conditional mean paths under a committed and a non-committed government, all on the same Markov grid. Both private-information cases start from the same initial reputation and receive no additional news shock. Belief and abatement are shares, investment is a capacity share per year, valuation is `q/c`, labour is relative to zero-tax labour, and output is in trillion USD/year. `commitment/paths.csv` contains these three paths and the continuous Ramsey reference.
- `commitment/benchmark-abatement`: compare the known-commitment Markov simulation with the saved continuous Ramsey BVP. The BVP is matched by model parameters and the complete announced policy, including its tail. Keeping this numerical check separate prevents a discrepancy between the two approximations from being attributed to reputation. The short filenames in this subdirectory also avoid Windows path-length limits.
- `commitment/policy-gap`: the expected tax shortfall conditional on being below full abatement, comparing the two true types with the same initial reputation. `commitment/diagnostics.csv` reports capacity probabilities, conditional actual and expected rates, and the contribution of below-capacity states to the unconditional tax shortfall. At full abatement the non-committed rate is zero by convention; a difference in mean rates there need not imply a difference in allocations or emissions.
- `reputation-shock-timing-{emissions,abatement}`: vary the arrival date of the same adverse news. The horizontal axis measures baseline abatement progress from `a₀` towards baseline long-run expected abatement; labels at selected points give the corresponding pre-shock expected emissions. The emissions figure shows the percentage change in cumulative emissions to the common saved horizon. The companion figure reports the percentage change in long-run abatement.
- `ets-information-channel-{separation,belief-volatility,best-response}`: how signal noise changes learning, belief volatility and the government's hidden best response at fixed baseline forecasts and continuation values. This is a local information diagnostic, not a re-solved equilibrium comparison.
- `ets-equilibrium-comparison-{emissions,near-zero,welfare}`: emitted only when otherwise matching saved equilibria with different signal noise are available. These compare cumulative emissions, the probability of reaching 95% abatement by the common horizon, and household welfare costs. Compatible solutions must use the same model, announced policy, grid, time step, tolerances and saved dates.
- `tax-regime-map-{incentive,announcement-gap,reputation-wedge}`: the intrinsic downward-deviation incentive `τᶜ−τP(a)`, the stored policy's distance from the announcement, and its difference from the feasible static benchmark `min(τP(a),τᶜ)`. The capacity column is excluded because the static tax is indeterminate there.
- `tax-regime-exposure-{incentive,pooling,wedge}`: the mean transition in the intrinsic incentive regions, probability-weighted pooling shares, and the mean absolute reputational tax wedge in USD/tCO₂e. Both shares and wedge magnitudes condition on being below full capacity and are omitted once that probability is below `1e-10`.

The reputation script exports 21 separate plots when matching noise equilibria are available, or 18 with only the baseline. The commitment script exports 12 additional figures independently. Earlier multi-panel exports are preserved but are no longer regenerated.

Under the committed type the actual tax is the announcement and the belief drift is `φ(1−φ)²χ²`; under the non-committed type it is `−φ²(1−φ)χ²`. Firms use the same equilibrium expected policy and valuation functions at a given public state under either true type. Both forward generators retain the solved chain's common diffusion stencil. The comparison therefore distinguishes known commitment from commitment that firms must learn about. Repeated announcement dates retain the left and right limits of tax and allocation jumps. Cumulative emissions use the same endpoint quadrature and common horizon across all three cases. These are conditional mean paths, computed from forward probabilities rather than Monte Carlo draws.

Tax figures report unconditional means, including the zero-rate convention at full abatement under non-commitment. A gap in mean tax rates can therefore reflect mass at full abatement, where the rate no longer affects labour or emissions. Interpret the rate figures together with abatement and emissions.

The timing experiment selects up to `nshocks` deterministic dates nearest equally spaced baseline progress levels. Progress at a shock date `s` is `(E[a_s]−a₀)/(E[a_∞]−a₀)`; it is measured relative to the no-news long-run outcome, which can be below `η`. The news always multiplies posterior odds by the same `oddsratio`. All dates are propagated together on one forward mesh, sharing each implicit factorisation; neither the policy functions nor the model calibration are re-solved. `reputation-shock-timing.csv` records the shock dates, pre-shock emissions, baseline progress, outcome levels and percentage changes. Cumulative emissions always end at the same calendar horizon, including the common history before each shock.

Long-run labels are used only if the stationary tail generator leaves both abatement and emissions unchanged at every grid state. Beliefs can continue moving at full abatement without changing these outcomes. If this check fails, the plots use terminal-outcome labels and progress towards the baseline terminal stock instead; the CSV records the distinction. This experiment does not assume that the saved horizon itself is a steady state.

`tax-regime-grid.csv` reports these diagnostics at every saved date and noise calibration, including the maximum tax variation across beliefs and across abatement stocks. It checks the saved announcement gap in intrinsically pooling states and the static-tax rule at both belief boundaries. Its tax quantities are in model tax units; divide by `taxfactor` for USD/tCO₂e. `tax-regime-visited.csv` reports forward probability masses, the physical posterior variance `σφ²`, and signed/absolute tax wedges in model tax units, with the before/after news dates distinguished. `tax-regime-exposure.csv` integrates pooling mass and reputation-sensitive policy mass over time and divides by integrated below-capacity mass; it also reports mean signed and absolute tax wedges in USD/tCO₂e. These exposure measures describe pre-capacity state-time, not an unweighted average of grid points or conditional shares. Capacity mass is reported separately. Distinguish a policy difference above solver tolerance from an economically large difference: both incidence and magnitude are reported.

Observed pooling means the tax is within `max(10*policytolerance*taxscale, 32eps(T)*max(1,taxscale))` of the announcement. A strongly disciplined interior solution can be numerically close to the ceiling even when `τᶜ>τP`, so differences between the two binary masks alone do not imply a solver error. A flat current tax also does not establish that investment is independent of reputation: continuation values can depend on future separating states and the capacity boundary.

Pooling shares can be jagged because the moving regime threshold crosses discrete abatement grid points and reclassifies their probability mass. These jumps in a binary diagnostic do not imply oscillations in the tax itself; interpret them alongside the continuous wedge magnitudes.

Forward distributions use the transpose of the same local Lax--Friedrichs Markov generator as the value solver and fully implicit probability steps. All projections condition on the true government being non-committed. Baseline learning already allows endogenous reputational erosion; the intervention is additional adverse public news. Emissions use firms' expected policy through labour demand. Cumulative emissions are an outcome integral, with no added climate state or feedback. The projection stops at the saved horizon; no terminal path is extrapolated.

Welfare uses the household cost value `u`, rather than the government's reduced-form objective `uᵍ`. The news effect is the discounted change in expected continuation costs at the news date. Welfare levels are reported relative to baseline learning without the additional news, in percent of one year's zero-tax output; these are non-annualised NPV costs. Positive cost changes mean lower welfare. The forward and backward time meshes can differ, so this value-function diagnostic need not exactly equal a separately integrated numerical welfare stream. `reputation-comparison.csv` contains the corresponding numerical outcomes.

Here `σ` is public-signal noise. Reducing it represents the information channel of curbing non-policy market noise. It can improve discipline while also making a non-committed government easier to identify. The model does not directly identify the effects of an allowance-price collar, reserve rule or other ETS instrument on `σ`, policy means or investment risk. The script therefore does not presume that lower noise improves welfare or that a reputation loss must reduce taxes.

Additional noise equilibria can be generated separately, keeping the calibration and numerical settings fixed:

```powershell
foreach ($sigma in 0.19, 0.38, 0.76) {
    julia --project=. scripts/noncommitted.jl --sigma=$sigma
}
```

The solver preserves existing result files, so omit calibrations that are already saved. Refine the forward projection step to assess projection accuracy and use separately solved finer grids to assess equilibrium discretisation error.

