"""Reproduce Table S9: partial correlation coefficients (CLBP, 9 LLM
metrics) for every edge that is FDR-significant at any of the 3 tested
Graphical Lasso alphas (see table_s8_glasso_alpha_sensitivity.py), with
r and p reported at each alpha.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from glasso_correlations import select_alpha_1se_band, alpha_sensitivity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

import numpy as np
import pandas as pd

METRICS = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
METRIC_DISPLAY_NAMES = {
    "Physical_Pain": "Physical Pain", "poor_QoL": "QoL", "Emotional_Pain": "Emotional Pain",
    "Catastrophizing": "Catastrophizing", "Depression": "Depression", "Anxiety": "Anxiety",
    "Rumination": "Rumination", "Agency_Deficit": "Agency Deficit",
    "Narrative_Fragmentation": "Narrative Fragmentation",
}
ALPHA_COLUMN_LABELS = ["alpha=7.69", "alpha=25.11 (CV)", "alpha=53.45"]


def _fmt(r: float, p: float, significant: bool) -> str:
    if not significant:
        return "Not significant"
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"r={r:.3f}, {p_text}"


def make_table() -> pd.DataFrame:
    df = load_llm_scores("common")
    df_clbp = df.loc[df["dx"] == "CLBP", METRICS]

    alpha_low, alpha_optimal, alpha_high = select_alpha_1se_band(df_clbp)
    results = alpha_sensitivity(df_clbp, [alpha_low, alpha_optimal, alpha_high], n_bootstrap=5000)

    triu = np.triu_indices(len(METRICS), k=1)
    any_significant = np.zeros_like(results[0]["reject"], dtype=bool)
    for result in results:
        any_significant |= result["reject"]

    rows = []
    for i, j in zip(*triu):
        if not any_significant[i, j]:
            continue
        edge = f"{METRIC_DISPLAY_NAMES[METRICS[i]]} - {METRIC_DISPLAY_NAMES[METRICS[j]]}"
        row = dict(Edge=edge)
        for result, col_label in zip(results, ALPHA_COLUMN_LABELS):
            row[col_label] = _fmt(result["partial_corr"][i, j], result["p_fdr"][i, j], result["reject"][i, j])
        rows.append(row)

    table = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "glasso" / "table_s9_significant_edges.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    make_table()
