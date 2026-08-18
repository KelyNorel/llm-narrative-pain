"""Reproduce Table S3: does cohort (CLBP/MDD/HC) explain each LLM metric
beyond what age and sex alone explain? For each of the 9 metrics, a
Freedman-Lane permutation test (5,000 permutations, see
permutation_analysis.py) compares the full model (metric ~ cohort + age
+ sex) against the reduced model (metric ~ age + sex), then FDR
(Benjamini-Hochberg) corrects across the 9 metrics.

Metrics are processed in the paper's original computation order (not
its display order) so that each metric's fixed permutation seed
(seed=i, i = index in that order) reproduces the exact F/p values from
the source notebook; the output table is then reindexed to the paper's
display order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from permutation_analysis import freedman_lane, N_PERM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEMOGRAPHICS_CSV, RESULTS_DIR

import pandas as pd
from statsmodels.stats.multitest import multipletests

# original computation order (fixes each metric's permutation seed)
COMPUTATION_ORDER = [
    "Physical_Pain", "Depression", "Anxiety", "Emotional_Pain", "poor_QoL",
    "Agency_Deficit", "Narrative_Fragmentation", "Rumination", "Catastrophizing",
]

# paper's display order (Table S3 row order)
DISPLAY_ORDER = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]

DISPLAY_NAMES = {
    "Physical_Pain": "Physical Pain", "poor_QoL": "poor QoL",
    "Emotional_Pain": "Emotional Pain", "Catastrophizing": "Catastrophizing",
    "Depression": "Depression", "Anxiety": "Anxiety", "Rumination": "Rumination",
    "Agency_Deficit": "Agency Deficit", "Narrative_Fragmentation": "Narrative Fragmentation",
}


def load_data() -> pd.DataFrame:
    scores = load_llm_scores("common").rename(columns={"study_id": "subject_id", "dx": "cohort"})
    demo = pd.read_csv(DEMOGRAPHICS_CSV)[["Study ID", "Age", "Sex"]].rename(
        columns={"Study ID": "subject_id", "Age": "age", "Sex": "sex"})
    return scores.merge(demo, on="subject_id", how="inner")


def make_table() -> pd.DataFrame:
    df = load_data()

    rows = []
    for i, metric in enumerate(COMPUTATION_ORDER):
        f_reduced = f"{metric} ~ age + C(sex)"
        f_full = f"{metric} ~ C(cohort) + age + C(sex)"
        f_obs, p_perm = freedman_lane(df, metric, f_reduced, f_full, n_perm=N_PERM, seed=i)
        rows.append(dict(metric=metric, F=f_obs, p_perm=p_perm))

    res = pd.DataFrame(rows)
    res["p_FDR"] = multipletests(res["p_perm"], method="fdr_bh")[1]
    res["sig_FDR"] = res["p_FDR"] < 0.05

    res = res.set_index("metric").loc[DISPLAY_ORDER].reset_index()
    res["Metric"] = res["metric"].map(DISPLAY_NAMES)
    res = res[["Metric", "F", "p_perm", "p_FDR", "sig_FDR"]]

    output_path = RESULTS_DIR / "permutation_tests" / "table_s3_group_comparisons.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    make_table()
