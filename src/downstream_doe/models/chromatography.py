"""Phase 1.1 — Mechanistic chromatography model (Equilibrium-Dispersive).

The 1-D ED model for a single-protein component is:

    [ε + (1-ε)·H(c_salt)] · ∂c/∂t  =  -u · ∂c/∂z  +  D_ap · ∂²c/∂z²

where H = dq/dc is the isotherm slope (Henry's constant) and ε is the bed
porosity.  This is the *retarded convection-dispersion* equation.

Numerics — method of lines:
  - N finite-volume cells, upwind scheme for convection (first-order, avoids
    oscillations near sharp fronts), central differences for axial dispersion.
  - Danckwerts inlet BC:  u·c_in = u·c[0] − D_ap·(c[1]−c[0])/Δz
  - Zero-gradient outlet BC: c[N] = c[N−1].
  - Time integration: ``scipy.integrate.solve_ivp`` with ``BDF`` (suitable for
    stiff retarded systems with large H values).

Isotherms:
  - **Langmuir** (competitive multi-component): q_i = qm_i·c_i/(1+Σk_j·c_j).
  - **Steric Mass Action (SMA)** — linearised for dilute protein:
        q  = H(salt, pH) · c,    H = K·exp(ν_pH·(pH−pH_ref))·(Λ/salt)^ν
    This makes *salt* and *pH* physical CPPs whose effect is analytically
    grounded.  Full non-linear SMA would require a per-cell Newton solve (see
    the commented extension below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnGeometry:
    """Physical column dimensions and packing."""

    length: float       # m
    diameter: float     # m
    porosity: float     # bed void fraction ε (−)


@dataclass(frozen=True)
class SMAParameters:
    """Linearised Steric Mass Action isotherm parameters.

    The linearised Henry's constant is:
        H = K · exp(ν_pH · (pH − pH_ref)) · (Λ / salt)^ν

    Parameters
    ----------
    equilibrium_constant:
        Dimensionless equilibrium constant K for each component.
    characteristic_charge:
        Characteristic charge ν (determines salt sensitivity).
    steric_factor:
        Steric exclusion factor σ (used in full non-linear SMA).
    ionic_capacity:
        Resin ionic capacity Λ (mM).
    ph_ref:
        Reference pH for the pH-shift exponent (default 7.0).
    nu_ph:
        Sensitivity of ln(H) to pH (default 0 = pH-independent).
    """

    equilibrium_constant: Sequence[float]
    characteristic_charge: Sequence[float]
    steric_factor: Sequence[float]
    ionic_capacity: float
    ph_ref: float = 7.0
    nu_ph: float = 0.0


@dataclass(frozen=True)
class ChromatographyConfig:
    """Full configuration of an ED chromatography run."""

    geometry: ColumnGeometry
    velocity: float         # interstitial velocity u (m/s)
    dispersion: float       # apparent axial dispersion D_ap (m²/s)
    isotherm: SMAParameters
    salt: float             # mobile-phase salt concentration (mM)
    ph: float               # mobile-phase pH (−)
    load_density: float     # mg protein per mL resin (g/L resin)
    n_cells: int = 100      # spatial discretisation cells


# ── Isotherm functions ────────────────────────────────────────────────────────

def langmuir_isotherm(
    c: np.ndarray,
    q_max: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Competitive multi-component Langmuir isotherm.

    Parameters
    ----------
    c:
        Mobile-phase concentrations, shape ``(n_components,)``.
    q_max:
        Maximum binding capacities, same shape.
    k:
        Adsorption equilibrium constants, same shape.

    Returns
    -------
    q:
        Bound concentrations.
    dq_dc:
        Diagonal of the Jacobian ∂q_i/∂c_i (off-diagonals neglected for
        the retardation factor in the ED model).
    """
    c = np.asarray(c, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    k = np.asarray(k, dtype=float)

    denom = 1.0 + float(np.dot(k, c))
    q = q_max * c / denom

    # ∂q_i/∂c_i = q_max_i · (1 + Σ_{j≠i} k_j c_j) / denom²
    cross_sum = np.dot(k, c)  # Σ_j k_j c_j
    dq_dc = q_max * (1.0 + cross_sum - k * c) / denom**2

    return q, dq_dc


def sma_henry_constant(salt: float, ph: float, params: SMAParameters, index: int = 0) -> float:
    """Linearised SMA Henry's constant for component *index*.

    H = K · exp(ν_pH · (pH − pH_ref)) · (Λ / salt)^ν

    At low protein concentration q ≈ H · c, giving an analytically tractable
    retardation factor R = 1 + (1−ε)/ε · H.
    """
    K = float(np.asarray(params.equilibrium_constant)[index])
    nu = float(np.asarray(params.characteristic_charge)[index])
    H = (
        K
        * math.exp(params.nu_ph * (ph - params.ph_ref))
        * (params.ionic_capacity / salt) ** nu
    )
    return H


def sma_isotherm(
    c: np.ndarray,
    salt: float,
    ph: float,
    params: SMAParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearised SMA bound concentration and isotherm slope.

    Returns ``(q, dq_dc)`` for all components (shape ``(n_components,)``).
    """
    n = len(params.equilibrium_constant)
    H = np.array([sma_henry_constant(salt, ph, params, i) for i in range(n)])
    c_arr = np.asarray(c, dtype=float)
    return H * c_arr, H


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(config: ChromatographyConfig, t_eval: np.ndarray) -> dict[str, np.ndarray]:
    """Run the ED column model and return outlet chromatograms.

    The single-component protein is injected as a rectangular pulse whose
    duration is set by *load_density* (the total mass loaded per mL resin).
    The column is initially empty.  After the load phase the inlet switches to
    zero concentration (wash / isocratic elution).

    Parameters
    ----------
    config:
        Full column + process configuration.
    t_eval:
        Time points at which to report the solution (seconds).

    Returns
    -------
    dict with keys:

    * ``"t"`` — reported time points (shape ``(n_t,)``).
    * ``"c_outlet"`` — outlet concentration, shape ``(1, n_t)`` (one component).
    * ``"c_profile"`` — full spatial profile at the last time, shape ``(n_cells,)``.
    * ``"t_load"`` — computed end-of-load time (s).
    * ``"henry"`` — SMA Henry's constant H under these conditions.
    """
    N = config.n_cells
    L = config.geometry.length
    d = config.geometry.diameter
    eps = config.geometry.porosity
    u = config.velocity
    D = config.dispersion

    dz = L / N

    # Column and flow geometry
    A_cross = math.pi / 4.0 * d**2          # cross-sectional area (m²)
    V_column = A_cross * L                   # total column volume (m³)
    V_resin = V_column * (1.0 - eps)         # stationary-phase volume (m³)
    Q_flow = u * A_cross * eps               # volumetric flow rate (m³/s)

    # Feed properties
    c_feed = 1.0  # g/L — a representative protein feed concentration
    # Load volume that delivers load_density mg/mL_resin of protein:
    #   load_density [g/L_resin] * V_resin [m³] / c_feed [g/L] = V_inject [m³]
    V_inject = config.load_density * V_resin  # m³  (since g/L = kg/m³ up to factor 1)
    t_load = V_inject / Q_flow if Q_flow > 0 else float("inf")

    # SMA Henry's constant (linearised; single component, index 0)
    H = sma_henry_constant(config.salt, config.ph, config.isotherm, index=0)
    # Retardation factor R = 1 + (1-ε)/ε * H
    R = 1.0 + (1.0 - eps) / eps * H

    # ── ODE right-hand side (method of lines) ────────────────────────────────
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        c_in = c_feed if t <= t_load else 0.0

        # Ghost-cell extended array [c_in, c[0..N-1], c[N-1]]
        c_ext = np.empty(N + 2)
        c_ext[0] = c_in          # inlet (Dirichlet)
        c_ext[1 : N + 1] = y
        c_ext[N + 1] = y[-1]     # outlet zero-gradient (Neumann)

        # Upwind convective flux: (c_i − c_{i-1}) / dz
        conv = u * (c_ext[1 : N + 1] - c_ext[0:N]) / dz

        # Central dispersion: (c_{i+1} − 2c_i + c_{i-1}) / dz²
        disp = D * (c_ext[2 : N + 2] - 2.0 * c_ext[1 : N + 1] + c_ext[0:N]) / dz**2

        return (-conv + disp) / R

    y0 = np.zeros(N)

    sol = solve_ivp(
        rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
    )

    c_outlet = sol.y[-1, :]  # last spatial cell → outlet chromatogram

    return {
        "t": sol.t,
        "c_outlet": c_outlet[np.newaxis, :],  # shape (1, n_t)
        "c_profile": sol.y[:, -1],            # spatial profile at final time
        "t_load": t_load,
        "henry": H,
        "retardation": R,
    }


# ── Performance metrics ───────────────────────────────────────────────────────

def pool_metrics(
    t: np.ndarray,
    c_outlet: np.ndarray,
    *,
    cut_start: float,
    cut_end: float,
    target_index: int = 0,
) -> dict[str, float]:
    """Compute **yield / purity / productivity** for a pool between cut-points.

    The pool is the fraction of the outlet chromatogram between *cut_start* and
    *cut_end* (both in seconds).  For a single-component run, purity is
    trivially 1.0; purity is meaningful when *c_outlet* contains multiple
    components (rows) — the target component is selected by *target_index*.

    Parameters
    ----------
    t:
        Time axis (s), shape ``(n_t,)``.
    c_outlet:
        Outlet concentrations, shape ``(n_components, n_t)``.
    cut_start, cut_end:
        Pool collection window (s).
    target_index:
        Row index of the target component in *c_outlet*.
    """
    t = np.asarray(t, dtype=float)
    c_outlet = np.atleast_2d(np.asarray(c_outlet, dtype=float))

    mask = (t >= cut_start) & (t <= cut_end)

    if mask.sum() < 2:
        return {"yield": 0.0, "purity": 0.0, "productivity": 0.0}

    # Trapezoid integration over pool window and over full chromatogram
    total_mass = np.trapezoid(c_outlet[target_index], t)
    pool_mass = np.trapezoid(c_outlet[target_index, mask], t[mask])
    all_pool_mass = np.sum([np.trapezoid(c_outlet[i, mask], t[mask]) for i in range(c_outlet.shape[0])])

    protein_yield = pool_mass / total_mass if total_mass > 0 else 0.0
    purity = pool_mass / all_pool_mass if all_pool_mass > 0 else 1.0
    dt = t[-1] - t[0]
    productivity = pool_mass / dt if dt > 0 else 0.0

    return {
        "yield": float(np.clip(protein_yield, 0.0, 1.0)),
        "purity": float(np.clip(purity, 0.0, 1.0)),
        "productivity": float(productivity),
    }
