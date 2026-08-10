"""Reproduce Fig. 5 panels A-C: LLM-derived metrics compared across
clinical cohorts (condition-specific interview section) and their
matched online communities (r/chronicpain, r/depressed).

Descriptive only — no per-pair p-value brackets except r/chronicpain
vs r/depressed. With ~4,900 and ~2,300 Reddit posts against 67/33
clinical subjects, Mann-Whitney p-values are trivially significant for
almost any difference, so they're not a meaningful filter here; the
one bracket drawn is gated on effect size (min_effect_size=2, i.e.
|mean difference| >= 2 points on the 0-10 scale) rather than p-value,
same mechanism as Fig. 2's brackets, different threshold.

Panel D (schematic summary of cross-context patterns) was built
manually in PowerPoint, not reproduced here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from boxplots import plot_metrics_comparison

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, FIGURES_DIR

import pandas as pd

METRICS_GROUPS = [
    ["Physical_Pain", "poor_QoL"],
    ["Emotional_Pain", "Catastrophizing", "Depression", "Anxiety", "Rumination"],
    ["Agency_Deficit", "Narrative_Fragmentation"],
]
METRIC_NAMES_GROUPS = [
    ["Physical Pain", "QoL#"],
    ["Emotional Pain", "Catastrophizing", "Depression", "Anxiety", "Rumination"],
    ["Agency\nDeficit", "Narrative\nFragmentation"],
]

REDDIT_CHRONICPAIN_CSV = RESULTS_DIR / "reddit" / "r_chronicpain.csv"
REDDIT_DEPRESSED_CSV = RESULTS_DIR / "reddit" / "r_depressed.csv"


def make_figure():
    clinical = load_llm_scores("condition_specific")

    reddit_pain = pd.read_csv(REDDIT_CHRONICPAIN_CSV)
    reddit_pain["dx"] = "r/chronicpain"
    reddit_dep = pd.read_csv(REDDIT_DEPRESSED_CSV)
    reddit_dep["dx"] = "r/depressed"

    df = pd.concat([clinical, reddit_pain, reddit_dep], ignore_index=True)

    fig, caption = plot_metrics_comparison(
        df,
        METRICS_GROUPS,
        METRIC_NAMES_GROUPS,
        group_labels=["CLBP", "r/chronicpain", "MDD", "r/depressed"],
        add_pval_table=False,
        min_effect_size=2,
        layout="vertical",
        pairwise_comparisons=[("r/chronicpain", "r/depressed", "#000000")],
        fn=FIGURES_DIR / "fig5_reddit_comparison.png",
    )
    print(caption)
    print(f"Saved -> {FIGURES_DIR / 'fig5_reddit_comparison.png'}")
    return fig, caption


if __name__ == "__main__":
    make_figure()
