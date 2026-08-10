"""Reproduce Fig. S1: speech duration and word count by cohort.

Uses boxplots.py's plot_metrics_comparison (Kruskal-Wallis omnibus,
then pairwise Mann-Whitney U, same mechanism as Fig. 2). All three
pairwise comparisons (CLBP-MDD, CLBP-HC, MDD-HC) come out p < 0.001
for both metrics, so brackets are omitted from the plot (three
identical *** brackets per panel just add clutter) and the result is
stated once in the figure caption instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boxplots import plot_metrics_comparison

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_AMOUNT_CSV, FIGURES_DIR

import pandas as pd

METRICS_GROUPS = [
    ["speech duration (min)"],
    ["word count"],
]
METRIC_NAMES_GROUPS = METRICS_GROUPS
Y_LABELS = ["Speech Duration (min)", "Word Count"]
Y_LIMITS = [(0, 60), (0, 10000)]


def make_figure():
    df = pd.read_csv(DATA_AMOUNT_CSV, encoding="utf-8-sig")

    fig, caption = plot_metrics_comparison(
        df,
        METRICS_GROUPS,
        METRIC_NAMES_GROUPS,
        main_title="Amount of Data Analyzed",
        group_labels=["CLBP", "MDD", "HC"],
        dx_col="Dx",
        y_labels=Y_LABELS,
        y_limits=Y_LIMITS,
        layout="horizontal",
        add_pval_table=False,
        pairwise_comparisons=None,  # all 3 pairs are p<0.001 for both metrics; see figure caption
        fn=FIGURES_DIR / "figs1_data_amount.png",
    )
    print(caption)
    print(f"Saved -> {FIGURES_DIR / 'figs1_data_amount.png'}")
    return fig, caption


if __name__ == "__main__":
    make_figure()
