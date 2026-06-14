# Implementation Notes

This document records the mathematical reasoning and engineering decisions behind every module in `src/downstream_doe/`.  Read it alongside `plan.md` (the *what*) and the individual source files (the *how*).

---

## Phase 1 — The Virtual Laboratory

### 1.1 Chromatography (`models/chromatography.py`)

#### Mathematical model

The single-component **Equilibrium-Dispersive (ED)** model collapses the full General Rate Model into a single effective PDE by assuming instantaneous local equilibrium between mobile and stationary phases.  Starting from the general mass balance:

```
ε·∂c/∂t + (1-ε)·∂q/∂t + u·∂c/∂z = D_ap·∂²c/∂z²
```

Since `q = q(c)` at equilibrium, `∂q/∂t = (dq/dc)·∂c/∂t`, so this simplifies to the **retarded convection-dispersion equation**:

```
R(c) · ∂c/∂t + u · ∂c/∂z = D_ap · ∂²c/∂z²

R(c) = ε + (1-ε)·H(c, salt, pH)
```

where `H = dq/dc` is the local slope of the binding isotherm (the Henry's constant in the linear limit).  `R` determines how slowly the protein band moves relative to the mobile phase.

#### Steric Mass Action (SMA) isotherm — linearised form

For ion-exchange chromatography, the SMA isotherm captures the fundamental salt/pH dependency of protein binding:

```
q = K · exp(ν_pH · (pH − pH_ref)) · (Λ / c_salt)^ν · c
```

This is the dilute-limit (linear, Henry) form of the full SMA isotherm.  The full implicit SMA expression is:

```
q_p = K · c_p · ((Λ − σ·q_p) / c_salt)^ν
```

which requires a Newton iteration at every spatial cell and time step.  For the Phase 1–3 range of the project, the linearised form is physically accurate (the protein concentration in the peak is much less than the resin capacity Λ) and orders-of-magnitude faster.

The key insight for the DoE: `H = K·(Λ/c_salt)^ν`, so:
- **Salt** enters as `c_salt^{−ν}` — increasing salt dramatically reduces binding (exponential in ν).
- **pH** enters via `exp(ν_pH·(pH − pH_ref))` — this factor makes pH a tunable CPP.

#### Numerical method — Method of Lines

Spatial discretisation on `N` finite-volume cells (cell size `Δz = L/N`):

| Term | Scheme | Rationale |
|---|---|---|
| Convection `u·∂c/∂z` | Upwind: `u(c_i − c_{i-1})/Δz` | Numerically stable for all Péclet numbers; first-order accuracy is sufficient here |
| Dispersion `D_ap·∂²c/∂z²` | Central: `D(c_{i+1}−2c_i+c_{i-1})/Δz²` | Second-order, stable for all Δz since D_ap is always positive |
| Inlet BC | Dirichlet ghost cell: `c_0 = c_{in}(t)` | Simplest; Danckwerts BC is more rigorous but requires solving for the ghost cell |
| Outlet BC | Neumann zero-gradient: `c_N = c_{N-1}` | Physically correct for a column discharging into open air |

Time integration uses `scipy.integrate.solve_ivp` with the **BDF** (Backward Differentiation Formula) solver.  BDF is chosen because:
1. The retardation factor `R` can be `O(10^4)` for strongly-retained proteins, making the ODE system very stiff.
2. BDF handles stiffness implicitly and selects step sizes based on the actual time scale of change, not the Courant limit.

**Stability note:** The explicit CFL condition for upwind + forward Euler would require `Δt ≤ Δz/u`, but with `R ~ 10^4` the effective condition is `Δt ≤ R·Δz/u` — orders of magnitude more permissive.  However, the dispersion stability condition `Δt ≤ Δz²/(2D)` can be severe for small `D`.  BDF avoids all of this automatically.

#### Load pulse and column loading

The inlet concentration profile is a rectangular pulse:

```
c_in(t) = c_feed  for  0 ≤ t ≤ t_load
c_in(t) = 0       for  t > t_load
```

The loading time is derived from the specified load density:

```
t_load = (load_density [g/L_resin] × V_resin [m³]) / (c_feed [g/L] × Q [m³/s])
```

where `c_feed = 1 g/L` is fixed (a representative chromatography feed).

#### Performance metrics

`pool_metrics` integrates the outlet chromatogram between cut-points using the trapezoidal rule (`numpy.trapezoid`):

- **Yield** = mass in pool / total injected mass
- **Purity** = target-component mass in pool / total-component mass in pool
- **Productivity** = pool mass / total cycle time

---

### 1.1b Multi-mode chromatography engine (`models/chromatography/`)

The single-component isocratic model above (`legacy.py`, kept verbatim for backward
compatibility) is generalised into a **transport-dispersive engine** (`engine.py`) that
covers every common downstream mode and supports **gradient elution**.  The original module
is now a package; `from downstream_doe.models.chromatography import …` resolves the same
legacy names plus the new API.

#### Unified governing equations (linear driving force)

Per finite-volume cell the state is `[c_i, q_i, m]` (mobile, stationary, modulator):

```
modulator   :  ∂m/∂t   = −u·∂m/∂z + D·∂²m/∂z²                     (unretained tracer)
mobile  c_i :  ∂c_i/∂t = −u·∂c_i/∂z + D·∂²c_i/∂z² − φ·∂q_i/∂t      φ = (1−ε)/ε
stationary  :  ∂q_i/∂t = k_m,i·(q*_i(c, m, pH) − q_i)             (LDF mass transfer)
```

Carrying `q` **explicitly** (instead of folding the isotherm into a constant retardation
factor) is the key design choice: it lets the inlet modulator vary in time without a
`dH/dt` source term and without a per-cell equilibrium (Newton) solve.  The
equilibrium-dispersive limit is recovered as `k_m → ∞`; finite `k_m` adds the mass-transfer
band broadening that sets the plate count, so it doubles as a "resolution knob".

#### One isotherm core, mode = modulator law (`isotherms.py`)

Every mode uses one explicit, multi-component competitive-Langmuir equilibrium,

```
q*_i = q_max,i·b_i(m,pH)·c_i / (1 + Σ_j b_j(m,pH)·c_j)      (nonlinear / overload)
q*_i = H_i·c_i,   H_i = q_max,i·b_i                          (linear / dilute, Henry)
```

and only the affinity law `b_i(m,pH)` changes:

| Mode | modulator `m` | `b_i(m,pH)` | Elutes by |
|---|---|---|---|
| CEX / AEX (`SMALaw`) | salt (mM) | `β_i·(Λ/m)^ν_i·exp(ν_pH,i·(pH−pH_ref))` | increasing salt |
| HIC (`SaltingOutLaw`) | salt (mM) | `β_i·exp(K_s,i·m)` | decreasing salt |
| RP-HPLC (`LinearSolventStrengthLaw`) | organic φ | `β_i·exp(−S_i·φ)` | increasing φ |

CEX vs AEX differ only in the sign of `ν_pH` (presets in `modes.py`).  In the dilute limit
`H_i = q_max,i·b_i` reproduces the legacy linearised SMA Henry constant exactly, so the
original tests and the DoE/UQ layers are unaffected.  "High-resolution IEX" is simply the
SMA law run nonlinear (finite `q_max`, competition) on a fine grid under a shallow gradient.

#### Elution program — the "linearly changing eluate" (`program.py`)

`ElutionProgram` is an ordered list of `Segment`s sized in **column volumes**, each holding
or **linearly ramping** the modulator (the gradient).  `Injection.from_load_density`
reuses the legacy `V_inject = load_density·V_resin / c_feed` logic to size the feed pulse.
`compile()` converts CV to seconds via `t_CV = V_c/Q = L/(u·ε)` and yields the inlet
`m_in(t)` (piecewise-linear, step-aware) and feed `c_in(t)`.

#### Numerics

Same method-of-lines discretisation as the legacy model (upwind convection, central
dispersion, Dirichlet inlet, zero-gradient outlet), integrated with stiff `BDF`.  Because
strong binding makes the system very stiff, a **Jacobian sparsity pattern** is handed to the
solver (`_jacobian_sparsity`): `c_i,j` couples to its transport neighbours and — through the
competitive isotherm — to every component's `c` and the modulator in the *same* cell.  This
replaces the dense finite-difference Jacobian with a few graph-coloured evaluations and cut
a representative gradient run from ~17 s to ~1 s.  The modulator is floored at `1e-9` to keep
the `(Λ/m)^ν` law finite under dispersion overshoot.

#### Separation metrics (`metrics.py`)

Beyond `pool_metrics`, `peak_moments` returns moment-based area / retention time / variance
(robust to skewed gradient peaks), `plate_count` gives `N = t_R²/σ²`, and `resolution`
computes `Rs = 2·Δt_R/(w_a+w_b)` with `w = 4σ`.

---

### 1.1c General Rate Model (`models/chromatography/grm.py`)

A second, independent chromatography solver built on the **finite volume method** via
[`PyFVTool`](https://github.com/FiniteVolumeTransportPhenomena/PyFVTool) (a Python port of
JFVM.jl). It exists for two reasons the lumped method-of-lines (MoL) engine cannot serve:

1. **Mechanistic mass transfer.** Instead of one lumped LDF coefficient `k_m`, the GRM
   resolves the *film* (boundary-layer) resistance around each bead **and** the *pore
   diffusion* inside it as a radial PDE per axial cell:

   ```
   bulk:     ∂c_i/∂t = -u ∂c_i/∂z + D_ax ∂²c_i/∂z² - (1-ε)/ε·(3/R_p)·k_f,i·(c_i - c_p,i|_R)
   particle: ε_p ∂c_p,i/∂t + (1-ε_p) ∂q_i/∂t = ε_p D_p,i·(1/r²)∂_r(r² ∂_r c_p,i)
   film BC:  ε_p D_p,i ∂c_p,i/∂r|_R = k_f,i (c_i - c_p,i|_R);   symmetry at r=0
   ```

   `k_f` defaults to the Wilson–Geankoplis correlation (`ε·Sh = 1.09 Re^⅓ Sc^⅓`), `D_p`
   to `ε_p·D_m/τ`; both are overridable. The same `Isotherm` (CEX/AEX/HIC/RP, competitive
   nonlinear) supplies `q* = q*(c_p, m, pH)`. The modulator is advanced separately on the
   bulk mesh (unretained tracer) and assumed to equilibrate instantly in the pores.

2. **Mass conservation.** The MoL engine's stiff BDF integration can drive a strongly-bound
   band negative and lose the *entire* injected mass under steep gradients (observed: 100 %
   loss). The GRM is **fully implicit** (backward Euler), assembling the coupled bulk +
   per-cell bead system into one global sparse matrix, and is conservative to machine
   precision.

**Three numerical points that make it exact:**

- **Identical-flux coupling.** The film exchange is added so the *same* discrete flux is a
  sink in the bulk cell and a source in the bead surface cell → exact bulk↔bead balance.
- **True shell volume.** PyFVTool's spherical `mesh.cellvolume` is the midpoint
  approximation `4π r_c² Δr`, but `diffusionTerm` conserves against the *true* shell volume
  `4/3·π(r_out³ − r_in³)`. The GRM uses the true shell volume for all mass accounting and
  film normalisation. (Using `cellvolume` leaks ~1/Nr².)
- **Storage-form adsorption.** The bead balance differences the change in *stored* mass
  `[ε_p c_p + (1-ε_p) q*(c_p,m)]ⁿ⁺¹ − [...]ⁿ` exactly (Picard-linearised), rather than via
  a `dq/dc` Jacobian increment. Because the old storage uses the old modulator, this single
  device captures both the nonlinear isotherm and the changing gradient with **no separate
  `dq/dt` source term**, and conserves mass exactly. (A Jacobian-increment discretisation
  leaks mass that grows with binding strength; a separate `dq*/dm·dm/dt` source is only
  O(Δt)-accurate.)

The GRM is the reference for mechanism studies and mass-transfer-limited separations; the
MoL engine remains the fast tool for benign design-space sweeps. Tests in `tests/test_grm.py`;
the MoL-vs-GRM comparison is documented in `doc/` (Chapter 1, §"general rate model").

---

### 1.2 UF/DF (`models/ufdf.py`)

#### Flux model — gel + pressure

Two limiting regimes govern permeate flux:

1. **Pressure-limited** (clean membrane, low concentration):
   ```
   J_pressure = TMP / (μ · R_m)
   ```
   where `μ` = water viscosity (10⁻³ Pa·s) and `R_m` = membrane hydraulic resistance.

2. **Mass-transfer limited** (concentration polarisation, gel layer):
   ```
   J_gel = k · ln(C_gel / C_bulk)
   ```
   derived from the film model (steady-state polarisation layer with gel concentration `C_gel = 500 g/L`).

The combined model takes the minimum: `J = min(J_pressure, J_gel)`.  This correctly transitions from pressure-controlled at low concentration to film-controlled at high concentration (the plateau region of the flux-vs-TMP curve).

The mass-transfer coefficient from the Lévêque/Graetz turbulent Sherwood correlation for hollow-fibre modules:

```
k(v) = k_ref · (v / v_ref)^0.8,   k_ref = 2×10⁻⁵ m/s at v_ref = 1 m/s
```

This makes **cross-flow velocity** a physical CPP: higher flow → better mass transfer → higher flux at the same TMP.

#### Mass balances (ODE system)

State: `y = [V(t), C(t)]` — retentate volume (m³) and concentration (g/L).

**UF phase** (concentrating):
```
dV/dt = −J(C) · A_membrane
dC/dt = J(C) · A_membrane · C · (1 − S) / V
```
where `S` = observed sieving coefficient (S=0 → perfect retention, all protein stays).

**DF phase** (buffer exchange at constant volume):
```
dV/dt = 0   (buffer in = permeate out)
dC/dt = −J(C) · A_membrane · C · (1 − S) / V   (dilution by buffer)
```

The system is mildly stiff (due to the logarithmic flux term) and integrated with `RK45`.

**Mass conservation check:** For `S = 0`, `d(V·C)/dt = C·dV/dt + V·dC/dt = −J·A·C + J·A·C = 0`. ✓

---

### 1.3 Perturbation (`perturbation.py`)

The noise model applies in sequence:

```
y_obs = y_true
      × (1 + CV · ε_proportional)    ← multiplicative (fraction of signal)
      + σ_add · ε_additive            ← additive (constant floor)
      + slope · x                     ← linear drift
      + bias                          ← calibration offset
```

where `ε ~ N(0,1)`.  The lognormal form for parameter jitter,

```
θ_batch = θ_true · exp(N(0, σ_rel))  ≈  θ_true · (1 ± σ_rel)  for small σ_rel
```

is used because physical parameters (rate constants, binding affinities) are positive definite.

---

## Phase 2 — Full Factorial DoE (`doe/factorial.py`, `doe/analysis.py`)

### Design generation

For a k-factor, 2-level design, all 2^k combinations of coded levels {−1, +1} are enumerated via `itertools.product`.  Physical values are obtained by linear decoding: `x_physical = centre + coded × half_range`.

Center points are appended at factor midpoints and labelled separately (`run_type = "center_point"`).  The run order is randomised with a fixed seed (42) to guard against time trends without sacrificing reproducibility.

### Response-surface analysis (`statsmodels.formula.api`)

The OLS model formula is built programmatically:

```
response ~ pH + salt + load_density + pH:salt + pH:load_density + salt:load_density
```

`statsmodels`' patsy integration parses this Wilkinson-Rogers formula and automatically expands interactions.  The Type-II ANOVA (partial sum of squares) is used via `anova_lm(model, typ=2)` — this tests each effect *adjusted for all others*, which is appropriate when factors are orthogonal (as in a full factorial) and consistent with standard CMC practice.

**Implementation note:** patsy cannot parse Python reserved words (e.g. `yield`) as column names.  The module aliases the response column to `__response__` internally before building the formula.

### Proven Acceptable Ranges (PAR)

For each factor, a univariate scan is performed: vary the factor over its training range while holding all others at their mean (centre point).  The factor range where the model prediction stays within the specification bounds `[lo, hi]` is reported as the PAR.

This is a first-order / linear PAR derivation — adequate for a robustness report based on a factorial design.  A full multivariate design space could be visualised from the response surface contour plots.

---

## Phase 3 — Latin Hypercube Sampling (`doe/lhs.py`)

**Why LHS over random sampling:** With `n` samples in `k` dimensions, a random design has `~1/n^(1/k)` expected spacing per dimension (the curse of dimensionality).  An LHS guarantees exactly one sample per "row" and "column" in each marginal dimension, giving uniformity in all projections regardless of `k`.

**Implementation:** `scipy.stats.qmc.LatinHypercube` with the `optimization="random-cd"` criterion minimises centered L₂-discrepancy (`CD`) by iterative column shuffling.  `CD` is a global measure of uniformity — designs with low `CD` are nearly uniform in all projections, which is optimal for space-filling.

**Coverage metrics:**
- `discrepancy` — centered L₂-discrepancy (lower → better uniformity).
- `min_pairwise_dist` — minimum Euclidean distance between any two points (higher → better spread, avoids clustering).
- `mean_pairwise_dist` — average pairwise distance.

The test `test_lhs_lower_discrepancy_than_random` empirically verifies that the optimised LHS beats a random sample on discrepancy — this is the core LHS claim.

---

## Phase 4 — Bayesian Optimisation (`optimization/surrogate.py`, `optimization/bayesopt.py`)

### GP surrogate (`botorch.models.SingleTaskGP`)

**Why botorch:** botorch (backed by GPyTorch and PyTorch) provides:
- Exact GP inference with Matérn-5/2 kernel (default) — smooth enough for chromatography/UF response surfaces.
- Analytic marginal log-likelihood maximisation via L-BFGS-B (`fit_gpytorch_mll`).
- Well-tested acquisition function optimisers.

**Normalisation:**
- Inputs normalised to [0,1] per factor using the training data bounds (via `botorch.utils.transforms.normalize`).
- Output standardised (subtract mean, divide by std) so the MLL landscape is scale-invariant.
- Predictions are un-normalised before returning so callers work in physical units.

**Why not scikit-learn GaussianProcessRegressor:** sklearn's GP is excellent for small datasets but botorch:
1. Handles batch evaluations natively (important for the BO loop).
2. Integrates directly with acquisition function optimisers.
3. Supports GPU acceleration for large datasets.

### Bayesian optimisation loop

The **Expected Improvement (EI)** acquisition is used:

```
EI(x) = E[max(f(x) − f*, 0)]
```

where `f*` is the current best observed value.  EI trades off exploration (high posterior variance) and exploitation (high posterior mean), which is optimal in the `n → ∞` limit under Gaussian assumptions.

The `LogExpectedImprovement` variant is used for numerical stability (avoids underflow when EI is very small).

**Multi-objective extension:** For simultaneous maximisation of yield *and* purity, `qLogNoisyExpectedHypervolumeImprovement` (qNEHVI) is the appropriate acquisition.  The surrogate setup is identical; only the acquisition function and reference point change.  This is documented in the `bayesopt.py` module but not yet implemented — it is a natural Phase 4 extension.

**Constraint handling:** Currently implemented as a penalty on oracle responses (infeasible points get `maximize = 0`).  A proper constrained BO would use a separate GP for each constraint and a `ConstrainedExpectedImprovement` acquisition.

---

## Phase 5 — Uncertainty Quantification (`uq/inverse.py`, `uq/uncertainty.py`)

### Inverse modeling

**Least-squares:** `scipy.optimize.least_squares` with `method="trf"` (trust-region reflective) handles box constraints naturally and is robust to near-flat loss landscapes.  The Jacobian at the solution approximates the Fisher information matrix, giving a frequentist confidence interval.

**MCMC (emcee):** The `emcee.EnsembleSampler` uses the affine-invariant ensemble algorithm (Goodman & Weare 2010).  Advantages:
1. Only one tuning parameter (`n_walkers`).
2. Affine-invariant → handles correlated parameters automatically (no diagonal Gaussian proposal issues).
3. Embarrassingly parallel across walkers.

**Likelihood:** Gaussian with known noise standard deviation `σ` (the *aleatoric* component, taken from the perturbation model):

```
log p(y_obs | θ) = −½ Σ [(y_obs_i − f(θ)_i)² / σ²]
```

**Prior:** Uniform on `[low, high]` per `ParameterPrior`.  For the typical parameter ranges (equilibrium constants, mass-transfer coefficients) this is weakly informative and lets the data dominate.

**Convergence:** The first half of the chain is discarded as burn-in.  In production, convergence should be checked via `az.summary` R-hat values (want R-hat < 1.01) and effective sample size (ESS > 100 per parameter).

### Aleatoric vs epistemic decomposition

Using the **law of total variance** (Eve's law):

```
Var[Y] = E_θ[Var[Y|θ]] + Var_θ[E[Y|θ]]
       = aleatoric       + epistemic
```

- **Aleatoric** = `E[σ²] = σ²_noise` (constant because we use a homoscedastic likelihood).
- **Epistemic** = `Var_θ[E[Y|θ]]` = variance of the posterior-predictive *mean* over parameter draws.

The Monte-Carlo estimator draws `n_draws` parameter vectors from the posterior and evaluates the forward model at each, giving an ensemble of predictions.  The epistemic variance is the sample variance of this ensemble.

**Interpretation:**
- A small epistemic fraction means the data have well-constrained the parameters → low uncertainty from knowledge gaps.
- A large aleatoric fraction means the process is inherently noisy → additional experiments won't reduce this uncertainty.
- This decomposition directly informs tech-transfer risk: epistemic uncertainty can be reduced by more characterisation experiments; aleatoric uncertainty defines the irreducible process variability that must be accommodated in the control strategy.

---

## Engineering decisions and known limitations

| Decision | Rationale |
|---|---|
| Linearised SMA isotherm | Physically accurate in the dilute-protein regime; avoids per-cell Newton iteration; enables analytic Henry's constant as a CPP |
| First-order upwind convection | Numerically stable without oscillations; a second-order WENO scheme would give sharper fronts at the cost of code complexity |
| `BDF` ODE solver for chromatography | Necessary for highly retarded systems (R >> 1); `RK45` would take prohibitively small steps |
| Single-component protein model | Sufficient for Phase 1–3 characterisation; multi-component extension (target + impurity) requires tracking competitive binding and is a natural Phase 1 extension |
| Uniform priors in MCMC | Appropriate when the prior is largely uninformative relative to the data; log-normal priors on rate constants would be more physically principled |
| `LogExpectedImprovement` (single-objective BO) | Simpler than `qNEHVI`; demonstrates the key BO loop logic clearly |
| Python reserved word `yield` → rename to `protein_yield` in user code | The `fit_response_model` function aliases the response column to `__response__` internally, but callers should avoid naming their response columns Python keywords |
| numpy `trapezoid` (not `trapz`) | `np.trapz` was removed in NumPy 2.0; `np.trapezoid` is the current API |
