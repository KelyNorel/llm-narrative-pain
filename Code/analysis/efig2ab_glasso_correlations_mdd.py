"""Reproduce eFigure 2 panels A/B: bivariate (Spearman) and GLasso
partial correlation between the nine LLM metrics, in MDD patients
(the MDD equivalent of Fig. 4 panels A/B).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from glasso_correlations import apply_glasso, precision_to_partial_correlation, bootstrap_pvalues, plot_glasso_with_fdr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR

METRICS = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
METRIC_DISPLAY_NAMES = [
    "Physical\nPain", "QoL#", "Emotional Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency Deficit", "Narrative\nFragmentation",
]


def make_figure(n_bootstrap: int = 1000):
    df = load_llm_scores("common")
    df_mdd = df.loc[df["dx"] == "MDD", METRICS]

    precision, covariance, optimal_alpha = apply_glasso(df_mdd, alpha=None)
    partial_corr = precision_to_partial_correlation(precision)
    direct_corr = df_mdd.corr(method="spearman").values

    direct_pvals, partial_pvals = bootstrap_pvalues(df_mdd, optimal_alpha, n_bootstrap=n_bootstrap)

    print(f"Optimal alpha (5-fold CV): {optimal_alpha:.4f}")
    fig, results = plot_glasso_with_fdr(
        direct_corr, partial_corr, direct_pvals, partial_pvals,
        METRIC_DISPLAY_NAMES, fdr_alpha=0.05, corr_threshold=0.01, figsize=(24, 11),
    )
    output_path = FIGURES_DIR / "efig2ab_glasso_correlations_mdd.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    return fig, results


if __name__ == "__main__":
    make_figure()
