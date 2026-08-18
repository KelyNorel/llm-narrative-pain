"""Reproduce Table S5: Spearman correlations between each LLM metric and
the continuous confounds (Age, Education, Word count, TICS), per cohort,
using confound_analysis.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confound_analysis import load_confound_data, run_all_cohorts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

import pandas as pd

COHORT_ORDER = ["CLBP", "MDD", "HC"]
CONFOUND_ORDER = ["Age", "Education (yrs)", "Word count", "TICS (cognition)"]
METRIC_ORDER = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
METRIC_DISPLAY_NAMES = {
    "Physical_Pain": "Physical Pain", "poor_QoL": "QoL",
    "Emotional_Pain": "Emotional Pain", "Catastrophizing": "Catastrophizing",
    "Depression": "Depression", "Anxiety": "Anxiety", "Rumination": "Rumination",
    "Agency_Deficit": "Agency Deficit", "Narrative_Fragmentation": "Narrative Fragmentation",
}


def make_table() -> pd.DataFrame:
    df = load_confound_data()
    df_all = run_all_cohorts(df)

    spear = df_all[df_all["Test"] == "Spearman ρ"].copy()
    spear["Cohort"] = pd.Categorical(spear["Cohort"], categories=COHORT_ORDER, ordered=True)
    spear["Confound"] = pd.Categorical(spear["Confound"], categories=CONFOUND_ORDER, ordered=True)
    spear["Metric"] = pd.Categorical(spear["Metric"], categories=METRIC_ORDER, ordered=True)
    spear = spear.sort_values(["Cohort", "Confound", "Metric"])

    table = spear[["Cohort", "Confound", "Metric", "N", "Stat", "p_raw", "p_FDR"]].rename(
        columns={"Stat": "Spearman_r", "p_raw": "p_uncorrected", "p_FDR": "p_FDR"})
    table["Metric"] = table["Metric"].map(METRIC_DISPLAY_NAMES)

    output_path = RESULTS_DIR / "confounds" / "table_s5_confound_correlations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    make_table()
