"""Confound analysis: test whether demographic/clinical-encounter
variables (age, sex, race, ethnicity, marital/employment status,
income, education, word count, cognition) are associated with the
nine LLM-derived metrics, per cohort.

Continuous confounds -> Spearman rho. Binary categorical -> Mann-Whitney
U. Multi-level categorical -> Kruskal-Wallis H. Within each cohort, a
test is skipped (not run, not "not significant") if fewer than two
categories have >=3 participants -- this happens for Race in HC and
Ethnicity in CLBP, where one category dominates the cohort. FDR
(Benjamini-Hochberg) is applied within each cohort separately, across
all tests run for that cohort.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEMOGRAPHICS_CSV, RESULTS_DIR

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

COHORTS = ["HC", "CLBP", "MDD"]
COHORT_COLOR = {"HC": "#70B870", "CLBP": "#E07070", "MDD": "#7090D0"}

LLM_METRICS = [
    "Narrative_Fragmentation", "Agency_Deficit",
    "Physical_Pain", "Emotional_Pain", "Depression",
    "poor_QoL", "Anxiety", "Rumination", "Catastrophizing",
]

INCOME_MAP = {
    "<$20,000": "Low (<$40k)",
    "$20-29,000": "Low (<$40k)",
    "$30-39,000": "Low (<$40k)",
    "$40-49,000": "Lower-middle ($40-99k)",
    "$50-99,000": "Lower-middle ($40-99k)",
    "$100-199,000": "Upper-middle ($100-199k)",
    ">/= $200,000": "High (>=$200k)",
    "Prefer not to say": np.nan,
    "Unknown": np.nan,
}
INCOME_ORDER = ["Low (<$40k)", "Lower-middle ($40-99k)", "Upper-middle ($100-199k)", "High (>=$200k)"]

# continuous -> Spearman
CONTINUOUS = {
    "Age": "Age",
    "Education (yrs)": "Total years of education",
    "Word count": "word_count",
    "TICS (cognition)": "TICS",
}

# binary categorical -> Mann-Whitney U  (col, group_a_label, group_b_label)
BINARY_CAT = {
    "Sex": ("Sex", "Female", "Male"),
}

# multi-level categorical -> Kruskal-Wallis H
MULTI_CAT = {
    "Marital status": "Marital status",
    "Employment status": "Employment status",
    "Income bucket": "Income_bucket",
    "Race": "Race",
    "Ethnicity": "Ethnicity",
}


def load_confound_data() -> pd.DataFrame:
    """Merge LLM scores (common section), word counts, and demographics
    into one DataFrame keyed by study_id."""
    llm_df = load_llm_scores("common")
    llm_df["study_id"] = llm_df["study_id"].astype(int)

    word_counts = pd.read_csv(RESULTS_DIR / "word_counts.csv")
    word_counts["study_id"] = word_counts["study_id"].astype(int)

    demo = pd.read_csv(DEMOGRAPHICS_CSV)
    demo.columns = demo.columns.str.strip()
    demo = demo.rename(columns={"Study ID": "study_id"})
    demo["study_id"] = demo["study_id"].astype(int)
    demo = demo.drop(columns=["Group"], errors="ignore")  # llm_df's dx is the source of truth

    df = llm_df.merge(word_counts[["study_id", "word_count", "char_count"]], on="study_id", how="left")
    df = df.merge(demo, on="study_id", how="left")

    df["Income_bucket"] = df["Total household income last year"].map(INCOME_MAP)
    df["Income_bucket"] = pd.Categorical(df["Income_bucket"], categories=INCOME_ORDER, ordered=True)
    return df


def run_cohort_tests(cohort_df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    """Run every confound test for one cohort. Returns a DataFrame of raw
    results (no FDR correction yet -- see run_all_cohorts)."""
    rows = []

    for conf_label, conf_col in CONTINUOUS.items():
        for metric in LLM_METRICS:
            sub = cohort_df[[conf_col, metric]].dropna()
            if len(sub) < 5:
                continue
            rho, p = stats.spearmanr(sub[conf_col], sub[metric])
            rows.append(dict(Cohort=cohort_name, Test="Spearman ρ", Confound=conf_label,
                              Metric=metric, N=len(sub), Stat=round(rho, 3), p_raw=p))

    for conf_label, (conf_col, grp_a, grp_b) in BINARY_CAT.items():
        for metric in LLM_METRICS:
            sub = cohort_df[[conf_col, metric]].dropna()
            a = sub.loc[sub[conf_col] == grp_a, metric]
            b = sub.loc[sub[conf_col] == grp_b, metric]
            if len(a) < 3 or len(b) < 3:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(dict(Cohort=cohort_name, Test="Mann-Whitney U", Confound=conf_label,
                              Metric=metric, N=len(a) + len(b), Stat=round(u, 1), p_raw=p))

    for conf_label, conf_col in MULTI_CAT.items():
        for metric in LLM_METRICS:
            sub = cohort_df[[conf_col, metric]].dropna()
            groups = [grp[metric].values for _, grp in sub.groupby(conf_col, observed=True) if len(grp) >= 3]
            if len(groups) < 2:
                continue  # not enough category diversity within this cohort to test
            h, p = stats.kruskal(*groups)
            rows.append(dict(Cohort=cohort_name, Test="Kruskal-Wallis H", Confound=conf_label,
                              Metric=metric, N=sum(len(g) for g in groups), Stat=round(h, 2), p_raw=p))

    return pd.DataFrame(rows)


def run_all_cohorts(df: pd.DataFrame, cohorts: list = COHORTS) -> pd.DataFrame:
    """run_cohort_tests for each cohort, with FDR (Benjamini-Hochberg)
    applied within each cohort separately."""
    all_results = []
    for cohort in cohorts:
        cdf = df[df["dx"] == cohort].copy()
        res = run_cohort_tests(cdf, cohort)
        if len(res) > 0:
            _, p_fdr, _, _ = multipletests(res["p_raw"], method="fdr_bh")
            res["p_FDR"] = p_fdr
            res["sig_FDR"] = p_fdr < 0.05
        all_results.append(res)
    return pd.concat(all_results, ignore_index=True)
