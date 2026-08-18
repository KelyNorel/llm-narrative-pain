"""Reproduce Fig. S5: same cohort comparison as Fig. 2, but each LLM
metric is normalized by the subject's common-section word count before
plotting -- a check that Fig. 2's cohort differences aren't just an
artifact of interview length (a potential confound, see Table S3). Same
panels, same pairwise brackets, same significance testing as Fig. 2
(boxplots.py's plot_metrics_comparison, re-run on the normalized values).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from word_counts import count_words_and_chars
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

ALL_METRICS = [m for group in METRICS_GROUPS for m in group]
Y_LIMITS = [[0, 0.10]] * len(METRICS_GROUPS)


def make_figure():
    df = load_llm_scores("common")
    word_counts = count_words_and_chars()
    word_counts["study_id"] = word_counts["study_id"].astype(int)

    df = df.merge(word_counts[["study_id", "word_count"]], on="study_id", how="left")
    for metric in ALL_METRICS:
        df[metric] = df[metric] / df["word_count"]

    fig, caption = plot_metrics_comparison(
        df,
        METRICS_GROUPS,
        METRIC_NAMES_GROUPS,
        group_labels=["CLBP", "MDD", "HC"],
        min_effect_size=0.0,
        add_pval_table=False,
        y_limits=Y_LIMITS,
        pairwise_comparisons=[
            ("CLBP", "MDD", "#ffbb78"),
            ("CLBP", "HC", "#98df8a"),
        ],
        fn=FIGURES_DIR / "figs5_word_count_normalized_ratings.png",
        layout="vertical",
    )
    print(caption)
    print(f"Saved -> {FIGURES_DIR / 'figs5_word_count_normalized_ratings.png'}")
    return fig, caption


if __name__ == "__main__":
    make_figure()
