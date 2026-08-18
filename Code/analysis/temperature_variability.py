"""Output-variability check: with temperature=0, how much does the LLM's
score for the same transcript change across repeated calls?

Data: `scores_100runs.csv` (see prepare_temperature_variability_scores.py),
filtered to the 104 subjects (of 115 with 100-run data) that are part of
the official 131-subject cohort, 6 metrics: LL_Pain, LL_PPain
(Physical_Pain), LL_EPain (Emotional_Pain), LL_Dep (Depression), LL_QoL
(poor_QoL), LL_Anx (Anxiety).

Failed calls (parse_failed=True rows) are dropped rather than kept as
NaN. ICC uses pingouin's nan_policy="omit", which does listwise deletion
of any subject missing at least one run -- not the same as dropping a
run for everyone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TEMP_VARIABILITY_SCORES_CSV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores

import numpy as np
import pandas as pd
import pingouin as pg

METRIC_COLUMNS = ["LL_Pain", "LL_PPain", "LL_EPain", "LL_Dep", "LL_QoL", "LL_Anx"]


def load_variability_data() -> pd.DataFrame:
    """Long-format scores for subjects in the official 131-subject cohort,
    successful calls only."""
    df = pd.read_csv(TEMP_VARIABILITY_SCORES_CSV)
    df = df[~df["parse_failed"]].copy()
    cohort_ids = set(load_llm_scores("common")["study_id"].astype(int))
    df = df[df["study_id"].isin(cohort_ids)]
    return df


def per_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD, and CV (%) of each metric across a subject's available
    runs. CV is unstable (can be arbitrarily large) for subjects whose
    mean is near zero -- a 0-9 rating scale allows that, unlike e.g. a
    strictly positive lab value, so read per-subject CV cautiously near
    the floor of the scale."""
    agg = df.groupby(["dx", "study_id"])[METRIC_COLUMNS].agg(["mean", "std"])
    rows = []
    for (dx, study_id), row in agg.iterrows():
        for metric in METRIC_COLUMNS:
            mean, std = row[(metric, "mean")], row[(metric, "std")]
            cv = (std / mean * 100) if mean != 0 else np.nan
            rows.append(dict(dx=dx, study_id=study_id, metric=metric, mean=mean, sd=std, cv_pct=cv))
    return pd.DataFrame(rows)


def compute_icc(df: pd.DataFrame) -> pd.DataFrame:
    """ICC(1,1): how much of the total variance is between-subject vs.
    run-to-run noise, treating each run as an interchangeable "rater" of
    the same subject. Subjects missing any run are excluded
    (nan_policy="omit"). One row per metric."""
    rows = []
    for metric in METRIC_COLUMNS:
        icc_table = pg.intraclass_corr(
            data=df, targets="study_id", raters="run", ratings=metric, nan_policy="omit"
        )
        icc1 = icc_table[icc_table["Type"] == "ICC(1,1)"].iloc[0]
        rows.append(dict(
            metric=metric, ICC1=icc1["ICC"], CI95_low=icc1["CI95"][0], CI95_high=icc1["CI95"][1],
            F=icc1["F"], pval=icc1["pval"],
        ))
    return pd.DataFrame(rows)
