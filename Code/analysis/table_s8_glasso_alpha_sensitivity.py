"""Reproduce Table S8: sensitivity of the CLBP partial-correlation
network (Fig. 4 panel B) to the Graphical Lasso regularization parameter
alpha, using the 1-SE band around the CV-optimal alpha (see
glasso_correlations.select_alpha_1se_band). For each alpha: how many of
the 36 possible edges (9 metrics) are zeroed out (|partial r| < 0.01),
and how many are FDR-significant (bootstrap, 5000 resamples).
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
CORR_THRESHOLD = 0.01
BAND_LABELS = ["Lower bound (1-SE)", "CV-optimal", "Upper bound (1-SE)"]


def make_table() -> pd.DataFrame:
    df = load_llm_scores("common")
    df_clbp = df.loc[df["dx"] == "CLBP", METRICS]

    alpha_low, alpha_optimal, alpha_high = select_alpha_1se_band(df_clbp)
    results = alpha_sensitivity(
        df_clbp, [alpha_low, alpha_optimal, alpha_high], n_bootstrap=5000, corr_threshold=CORR_THRESHOLD)

    rows = []
    for result, band_label in zip(results, BAND_LABELS):
        triu = np.triu_indices_from(result["partial_corr"], k=1)
        n_total = len(triu[0])
        n_zeroed = int((np.abs(result["partial_corr"][triu]) < CORR_THRESHOLD).sum())
        n_sig = int(result["reject"][triu].sum())

        rows.append(dict(
            alpha=round(result["alpha"], 2), Band=band_label,
            zeroed_edges=f"{n_zeroed}/{n_total}", sparsity_pct=round(n_zeroed / n_total * 100, 1),
            fdr_sig_edges=f"{n_sig}/{n_total}",
        ))

    table = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "glasso" / "table_s8_alpha_sensitivity.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    make_table()
