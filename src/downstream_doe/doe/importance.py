"""Tree-model screening analysis: which features drive the response.

A covering-array screen produces a wide, sparse, binary design (which strains were in each
run) against a noisy response.  Linear ANOVA struggles here — the interesting structure is
nonlinear and interaction-heavy — so we analyse it with **tree ensembles**:

* :func:`random_forest_importance` — ``sklearn`` :class:`~sklearn.ensemble.RandomForestRegressor`,
* :func:`gradient_boosting_importance` — ``xgboost`` :class:`~xgboost.XGBRegressor`.

Both report the same things through :class:`ImportanceResult` so they can be compared head to
head: a cross-validated R² (how much of the response the model actually explains out-of-sample)
and **permutation importance** (how much CV-honest accuracy is lost when each feature is
shuffled).  Permutation importance is used for both models because impurity / gain importances
are not comparable across the two libraries and are biased toward high-cardinality features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score


@dataclass
class ImportanceResult:
    """Fitted screening model plus its cross-validated quality and feature ranking.

    Attributes
    ----------
    model:
        The estimator fitted on the full data.
    cv_score:
        Mean cross-validated R².
    cv_scores:
        Per-fold R² scores.
    importances:
        Permutation importance per feature (mean), sorted descending.
    importances_std:
        Standard deviation of the permutation importance per feature (same index as
        ``importances``).
    predictions:
        Out-of-fold cross-validated predictions, aligned with the input rows.
    feature_names:
        Feature names in input order.
    """

    model: object
    cv_score: float
    cv_scores: np.ndarray
    importances: pd.Series
    importances_std: pd.Series
    predictions: np.ndarray
    feature_names: list[str]


def _analyse(
    estimator,
    X,
    y,
    feature_names: Sequence[str] | None,
    cv: int,
    seed: int | None,
    n_repeats: int,
) -> ImportanceResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if feature_names is None:
        feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
    feature_names = list(feature_names)

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(estimator, X, y, cv=splitter, scoring="r2")
    predictions = cross_val_predict(estimator, X, y, cv=splitter)

    estimator.fit(X, y)
    perm = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=seed, scoring="r2"
    )
    order = np.argsort(perm.importances_mean)[::-1]
    importances = pd.Series(
        perm.importances_mean[order], index=[feature_names[i] for i in order]
    )
    importances_std = pd.Series(
        perm.importances_std[order], index=[feature_names[i] for i in order]
    )

    return ImportanceResult(
        model=estimator,
        cv_score=float(np.mean(cv_scores)),
        cv_scores=cv_scores,
        importances=importances,
        importances_std=importances_std,
        predictions=predictions,
        feature_names=feature_names,
    )


def random_forest_importance(
    X,
    y,
    *,
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 400,
    max_depth: int | None = None,
    cv: int = 5,
    n_repeats: int = 10,
    seed: int | None = None,
) -> ImportanceResult:
    """Random-forest screening analysis (CV R² + permutation importance)."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    return _analyse(rf, X, y, feature_names, cv, seed, n_repeats)


def gradient_boosting_importance(
    X,
    y,
    *,
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    cv: int = 5,
    n_repeats: int = 10,
    seed: int | None = None,
) -> ImportanceResult:
    """XGBoost gradient-boosting screening analysis (CV R² + permutation importance)."""
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    return _analyse(xgb, X, y, feature_names, cv, seed, n_repeats)
