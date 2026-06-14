"""downstream_doe — a virtual laboratory for API process characterization & advanced DoE.

The package is organized to mirror the project's five phases:

* :mod:`downstream_doe.models`        — mechanistic forward models (the "Mechanistic Truth").
* :mod:`downstream_doe.perturbation`  — noise injection turning Truth into a "Virtual Experiment".
* :mod:`downstream_doe.doe`           — classical full-factorial and space-filling (LHS) designs.
* :mod:`downstream_doe.optimization`  — GP surrogates and Bayesian optimization.
* :mod:`downstream_doe.uq`            — inverse modeling and uncertainty quantification.

See ``plan.md`` at the repo root for the full development plan.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
