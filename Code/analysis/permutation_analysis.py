"""Freedman-Lane permutation test for a nested linear-model comparison:
does adding a term (e.g. cohort) to a model improve on a reduced model
(e.g. age + sex alone) more than chance, without assuming normality of
residuals? Used for Table S3 (cohort effect on each LLM metric,
controlling for age and sex).

Procedure: fit the reduced model, permute its residuals (not the raw
outcome) and add them back onto its fitted values, refit both reduced
and full models on the permuted outcome, and compare the resulting
nested F-statistics against the observed one -- this is what makes it
valid under a reduced-model null rather than a null of no association
at all.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

N_PERM = 5000


def nested_F(fit_reduced, fit_full) -> float:
    rss_r, rss_f = np.sum(fit_reduced.resid**2), np.sum(fit_full.resid**2)
    df_num = fit_reduced.df_resid - fit_full.df_resid
    df_den = fit_full.df_resid
    return ((rss_r - rss_f) / df_num) / (rss_f / df_den)


def freedman_lane(data: pd.DataFrame, metric: str, f_reduced: str, f_full: str,
                   n_perm: int = N_PERM, seed: int = 0) -> tuple:
    """Returns (F_obs, p_perm) for the term(s) present in f_full but not
    f_reduced."""
    rng = np.random.default_rng(seed)
    reduced_obs = smf.ols(f_reduced, data=data).fit()
    full_obs = smf.ols(f_full, data=data).fit()
    f_obs = nested_F(reduced_obs, full_obs)

    resid = reduced_obs.resid.values
    fitted = reduced_obs.fittedvalues.values
    tmp = data.copy()

    f_null = np.empty(n_perm)
    for b in range(n_perm):
        tmp[metric] = fitted + rng.permutation(resid)
        r_perm = smf.ols(f_reduced, data=tmp).fit()
        f_perm = smf.ols(f_full, data=tmp).fit()
        f_null[b] = nested_F(r_perm, f_perm)

    p_perm = (np.sum(f_null >= f_obs) + 1) / (n_perm + 1)
    return f_obs, p_perm
