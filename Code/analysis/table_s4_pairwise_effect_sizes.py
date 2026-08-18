"""Reproduce Table S4: pairwise cohort comparisons (CLBP vs MDD, CLBP vs
HC, MDD vs HC) on each of the nine LLM-derived metrics -- Kruskal-Wallis
omnibus per metric, then Mann-Whitney U for each pair with Cliff's delta
/ rank-biserial r as effect size (see effect_sizes.py). p-values are
FDR-corrected (Benjamini-Hochberg) across all 27 pairwise tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from effect_sizes import cliffs_delta_and_rank_biserial, interpret_cliffs_delta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

METRICS = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
DISPLAY_NAMES = {
    "Physical_Pain": "Physical Pain", "poor_QoL": "poor QoL",
    "Emotional_Pain": "Emotional Pain", "Catastrophizing": "Catastrophizing",
    "Depression": "Depression", "Anxiety": "Anxiety", "Rumination": "Rumination",
    "Agency_Deficit": "Agency Deficit", "Narrative_Fragmentation": "Narrative Fragmentation",
}
COHORTS = ["CLBP", "MDD", "HC"]
COMPARISONS = [("CLBP", "MDD"), ("CLBP", "HC"), ("MDD", "HC")]


def make_table() -> pd.DataFrame:
    df = load_llm_scores("common").rename(columns={"dx": "Dx"})

    rows = []
    for metric in METRICS:
        data_by_group = {g: df[df["Dx"] == g][metric].dropna().values for g in COHORTS}
        _, kw_p = kruskal(*data_by_group.values())
        assert kw_p < 0.05, f"{metric}: Kruskal-Wallis omnibus not significant, unexpected"

        for g1, g2 in COMPARISONS:
            d1, d2 = data_by_group[g1], data_by_group[g2]
            u_stat, p_pair = mannwhitneyu(d1, d2, alternative="two-sided")
            delta, r = cliffs_delta_and_rank_biserial(u_stat, len(d1), len(d2))
            rows.append(dict(
                Metric=DISPLAY_NAMES[metric], Comparison=f"{g1} vs {g2}",
                N1=len(d1), N2=len(d2), U=u_stat, p_value=p_pair,
                Cliffs_delta=delta, rank_biserial_r=r, Magnitude=interpret_cliffs_delta(delta),
            ))

    table = pd.DataFrame(rows)
    table["p_FDR"] = multipletests(table["p_value"], method="fdr_bh")[1]

    output_path = RESULTS_DIR / "effect_sizes" / "table_s4_pairwise_effect_sizes.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    make_table()
