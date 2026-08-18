"""Output-variability check: with temperature=0, how much does the LLM's
score for the same transcript change across repeated calls?

Data: `scores_100runs.csv` (see prepare_temperature_variability_scores.py),
115 subjects x 100 runs, 6 metrics (LL_Pain, LL_PPain, LL_EPain, LL_Dep,
LL_QoL, LL_Anx). This predates the paper's final 9-metric prompt and full
131-subject cohort: LL_PPain/LL_EPain/LL_Dep/LL_QoL/LL_Anx correspond to
the current Physical_Pain/Emotional_Pain/Depression/poor_QoL/Anxiety:
LL_Pain has no equivalent in the final 9 metrics, and none of the 4
Prompt-2 metrics (Catastrophizing, Rumination, Narrative_Fragmentation,
Agency_Deficit) were scored in this run. Read this as a check on whether
temperature=0 meaningfully reduces output variance at all, not as a
variability estimate for the paper's final metrics.

Runs 46 and 47 hit a watsonx connection outage affecting many subjects at
once (84 of 11500 runs; every failure falls in these two run indices, no
subject is missing any other run) and are dropped for ALL subjects before
computing statistics, leaving a fully balanced 115 x 98 design instead of
doing per-subject listwise deletion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TEMP_VARIABILITY_SCORES_CSV

import numpy as np
import pandas as pd
import pingouin as pg

METRIC_COLUMNS = ["LL_Pain", "LL_PPain", "LL_EPain", "LL_Dep", "LL_QoL", "LL_Anx"]
FAILED_RUNS = [46, 47]


def load_variability_data() -> pd.DataFrame:
    """Long-format scores with the failed-outage runs dropped for every
    subject, so every subject has the same 98 runs (fully balanced)."""
    df = pd.read_csv(TEMP_VARIABILITY_SCORES_CSV)
    df = df[~df["run"].isin(FAILED_RUNS)].copy()
    assert df["parse_failed"].sum() == 0, "unexpected parse failure outside the known outage runs"
    return df


def per_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD, and CV (%) of each metric across the 98 runs, per subject.
    CV is unstable (can be arbitrarily large) for subjects whose mean is
    near zero -- a 0-9 rating scale allows that, unlike e.g. a strictly
    positive lab value, so read per-subject CV cautiously near the floor
    of the scale."""
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
    run-to-run noise, treating each of the 98 runs as an interchangeable
    "rater" of the same subject. One row per metric."""
    rows = []
    for metric in METRIC_COLUMNS:
        icc_table = pg.intraclass_corr(
            data=df, targets="study_id", raters="run", ratings=metric, nan_policy="raise"
        )
        icc1 = icc_table[icc_table["Type"] == "ICC(1,1)"].iloc[0]
        rows.append(dict(
            metric=metric, ICC1=icc1["ICC"], CI95_low=icc1["CI95"][0], CI95_high=icc1["CI95"][1],
            F=icc1["F"], pval=icc1["pval"],
        ))
    return pd.DataFrame(rows)
