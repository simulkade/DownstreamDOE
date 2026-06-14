"""Phase 2 — Statistical analysis of factorial designs.

Fits an OLS response-surface model via ``statsmodels``, produces a Type-II
ANOVA table, and derives **proven acceptable ranges (PAR)** for each factor.

Design:
- The formula is constructed programmatically: main effects + optional
  two-way interactions + optional quadratic terms (Response Surface Model).
- PAR derivation inverts the linear model: for each factor, while holding
  all others at their centre points, find the factor range where the predicted
  response stays within the specification bounds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.anova as sa


@dataclass
class AnovaResult:
    """Container for a fitted response model and its ANOVA table."""

    model: object           # statsmodels OLSResults object
    anova_table: pd.DataFrame
    effects: pd.Series      # regression coefficients (main effects & interactions)
    factor_names: list[str] = None   # plain factor names (no interactions/quadratics)


def fit_response_model(
    data: pd.DataFrame,
    response: str,
    factors: Sequence[str],
    *,
    interactions: bool = True,
    quadratic: bool = False,
) -> AnovaResult:
    """Fit an OLS response model and compute its Type-II ANOVA table.

    The formula follows Wilkinson-Rogers notation::

        yield ~ pH + salt + load_density
              + pH:salt + pH:load_density + salt:load_density   (if interactions)
              + I(pH**2) + I(salt**2) + ...                      (if quadratic)

    Parameters
    ----------
    data:
        Design + response table (physical units).
    response:
        Name of the response column.
    factors:
        Factor column names (must exist in *data*).
    interactions:
        Include all two-way interaction terms.
    quadratic:
        Include quadratic (second-order) terms for a response-surface model.
        Requires data from a 3-level (or CCD) design.
    """
    terms: list[str] = list(factors)

    if quadratic:
        for f in factors:
            terms.append(f"I({f}**2)")

    if interactions:
        for f1, f2 in itertools.combinations(factors, 2):
            terms.append(f"{f1}:{f2}")

    # Patsy can't parse Python reserved words (e.g. "yield") as column names.
    # Alias the response to a safe internal name before building the formula.
    _RESPONSE_ALIAS = "__response__"
    data_fit = data.copy()
    data_fit[_RESPONSE_ALIAS] = data_fit[response]

    formula = f"{_RESPONSE_ALIAS} ~ " + " + ".join(terms)

    model = smf.ols(formula, data=data_fit).fit()
    anova_table = sa.anova_lm(model, typ=2)

    return AnovaResult(
        model=model,
        anova_table=anova_table,
        effects=model.params,
        factor_names=list(factors),
    )


def proven_acceptable_ranges(
    result: AnovaResult,
    *,
    response_spec: tuple[float, float],
) -> pd.DataFrame:
    """Derive per-factor proven acceptable ranges (PAR) from the fitted model.

    For each factor *x_i*, all other factors are held at their training-data
    mean (the "centre point"), and the factor range where the model prediction
    stays within *response_spec = (lo, hi)* is reported.

    This is a univariate PAR analysis.  For a multivariate design space see
    the full contour / RSM plots.

    Parameters
    ----------
    result:
        Fitted model from :func:`fit_response_model`.
    response_spec:
        Acceptable response bounds ``(lo, hi)``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``factor``, ``par_low``, ``par_high``, ``factor_range_low``,
        ``factor_range_high``.
    """
    model = result.model
    lo_spec, hi_spec = response_spec

    # Use stored factor names; fall back to model frame columns if not available
    if result.factor_names:
        factor_cols = list(result.factor_names)
    else:
        orig_data = model.model.data.orig_exog
        factor_cols = [c for c in orig_data.columns if c != "__response__"]

    # Build a centre-point DataFrame using the training data means
    orig_data = model.model.data.orig_exog
    centre_df = pd.DataFrame(
        {col: [float(orig_data[col].mean())] for col in factor_cols}
    )

    records = []
    for var in factor_cols:
        # Range of this factor in training data
        x_lo = float(orig_data[var].min())
        x_hi = float(orig_data[var].max())

        # Scan over the factor range with all others at their centres
        x_scan = np.linspace(x_lo, x_hi, 200)
        preds = []
        for xv in x_scan:
            row = centre_df.copy()
            row[var] = xv
            preds.append(float(model.predict(row).iloc[0]))

        preds = np.array(preds)
        in_spec = (preds >= lo_spec) & (preds <= hi_spec)

        if in_spec.any():
            par_lo = float(x_scan[in_spec].min())
            par_hi = float(x_scan[in_spec].max())
        else:
            par_lo = par_hi = float("nan")

        records.append(
            {
                "factor": var,
                "par_low": par_lo,
                "par_high": par_hi,
                "factor_range_low": x_lo,
                "factor_range_high": x_hi,
            }
        )

    return pd.DataFrame(records)
