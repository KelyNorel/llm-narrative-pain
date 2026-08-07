"""Reproduce Fig. 2: LLM ratings differ significantly across cohorts
(common interview section). Panel A: Physical Pain, QoL. Panel B:
Emotional Pain, Catastrophizing, Depression, Anxiety, Rumination.
Panel C: Agency Deficit, Narrative Fragmentation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from boxplots import plot_metrics_comparison

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR

METRICS_GROUPS = [
    ["Physical_Pain", "poor_QoL"],  # Panel A
    ["Emotional_Pain", "Catastrophizing", "Depression", "Anxiety", "Rumination"],  # Panel B
    ["Agency_Deficit", "Narrative_Fragmentation"],  # Panel C
]

METRIC_NAMES_GROUPS = [
    ["Physical Pain", "QoL#"],
    ["Emotional Pain", "Catastrophizing", "Depression", "Anxiety", "Rumination"],
    ["Agency\nDeficit", "Narrative\nFragmentation"],
]


def make_figure():
    df = load_llm_scores("common")
    fig, caption = plot_metrics_comparison(
        df,
        METRICS_GROUPS,
        METRIC_NAMES_GROUPS,
        group_labels=["CLBP", "MDD", "HC"],
        min_effect_size=0.0,
        add_pval_table=False,
        pairwise_comparisons=[
            ("CLBP", "MDD", "#ffbb78"),  # pastel orange (MDD's color)
            ("CLBP", "HC", "#98df8a"),  # pastel green (HC's color)
        ],
        fn=FIGURES_DIR / "fig2_llm_ratings_by_cohort.png",
        layout="vertical",
    )
    print(caption)
    print(f"Saved -> {FIGURES_DIR / 'fig2_llm_ratings_by_cohort.png'}")
    return fig, caption


if __name__ == "__main__":
    make_figure()
