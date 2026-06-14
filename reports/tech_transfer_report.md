# Tech-Transfer Report (Template)

> Phase 5 deliverable. Populated from the UQ notebook
> (`notebooks/05_uq_parameter_estimation.ipynb`). This template mirrors how a
> CMC team would communicate process understanding and risk to a receiving
> manufacturing site.

## 1. Process & scope
- Unit operation(s): _chromatography / UF-DF_
- Critical process parameters (CPPs) and ranges studied: _…_
- Critical quality attributes (CQAs): _yield, purity, …_

## 2. Process model & characterization
- Mechanistic model used as basis: _ED chromatography (SMA) / UF-DF flux model_
- Design space established from: _full factorial (Phase 2) + LHS (Phase 3)_
- Optimized operating point (Phase 4): _…_

## 3. Parameter estimates (inverse modeling)
## Filled by Phase 5 notebook (auto-generated)

### Parameter Estimates

| Parameter | LS estimate | MCMC mean | 90% credible interval | True value |
|---|---|---|---|---|
| K_eq | 0.0458 | 0.0496 | [0.0366, 0.0565] | 0.05 |
| ν    | 2.5521 | 2.5603 | [2.4351, 2.6750] | 2.5 |

### Uncertainty Summary

- Aleatoric (measurement noise): **28.0%** of total variance
- Epistemic (parameter uncertainty): **72.0%** of total variance

### Recommendation for tech transfer

⚠️  Additional scouting runs recommended — epistemic uncertainty is >30% of total.

## Filled by Phase 5 notebook (auto-generated)

### Parameter Estimates

| Parameter | LS estimate | MCMC mean | 90% credible interval | True value |
|---|---|---|---|---|
| K_eq | 0.0458 | 0.0537 | [0.0364, 0.0569] | 0.05 |
| ν    | 2.5521 | 2.5450 | [2.4306, 2.6783] | 2.5 |

### Uncertainty Summary

- Aleatoric (measurement noise): **18.8%** of total variance
- Epistemic (parameter uncertainty): **81.2%** of total variance

### Recommendation for tech transfer

⚠️  Additional scouting runs recommended — epistemic uncertainty is >30% of total.

| Parameter | Point estimate | 95% credible interval | True value (virtual) |
|---|---|---|---|
| _k_eq_ | _…_ | _[…, …]_ | _…_ |
| _ν (charge)_ | _…_ | _[…, …]_ | _…_ |
| _mass-transfer coeff_ | _…_ | _[…, …]_ | _…_ |

## 4. Uncertainty quantification
- **Aleatoric** (irreducible process/measurement noise): _… variance_
- **Epistemic** (parameter/model knowledge gaps): _… variance_
- Dominant contributor and implication: _…_

## 5. Impact on transfer to manufacturing
- Robustness margin at the operating point given quantified uncertainty: _…_
- Recommended control strategy / monitoring: _…_
- Residual risks and proposed mitigations: _…_

## 6. Conclusion
_Statement on readiness for transfer and confidence level._
