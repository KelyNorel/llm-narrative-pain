"""Reproduce Fig. S4: naive word-counting vs. LLM-derived metrics for
cohort differentiation. Panel A: 2D SVD projection ("MDS", see
mds_analysis.py) of the nine LLM metrics. Panel B: pain-word count vs.
depression-word count per subject's common-section transcript (see
keyword_counts.py). Point positions in panel B are jittered (fixed seed)
purely so overlapping integer counts are visible; the counts themselves
are exact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mds_analysis import compute_mds_projection
from keyword_counts import count_keywords_all_subjects

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

JITTER_RANDOM_STATE = 42


def make_figure():
    mds_df = compute_mds_projection()
    kw_df = count_keywords_all_subjects()

    rng = np.random.default_rng(JITTER_RANDOM_STATE)
    kw_df["pain_count_jitter"] = kw_df["pain_count"] + rng.normal(0, 0.25, len(kw_df))
    kw_df["depression_count_jitter"] = kw_df["depression_count"] + rng.normal(0, 0.15, len(kw_df))

    sns.set_context("talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.scatterplot(data=mds_df, x="MDS1", y="MDS2", hue="dx", s=80, alpha=0.7, ax=axes[0])
    axes[0].set_title("MDS (multidimensional scaling) colored by Cohort", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("MDS1", fontsize=13)
    axes[0].set_ylabel("MDS2", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=11)
    axes[0].text(-0.12, 1.05, "A", transform=axes[0].transAxes, fontsize=16, fontweight="bold")

    sns.scatterplot(data=kw_df, x="pain_count_jitter", y="depression_count_jitter",
                     hue="dx", s=100, alpha=0.7, ax=axes[1])
    axes[1].set_title("Depression Count vs Pain Count by Cohort", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Pain Count", fontsize=13)
    axes[1].set_ylabel("Depression Count", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=11)
    axes[1].text(-0.12, 1.05, "B", transform=axes[1].transAxes, fontsize=16, fontweight="bold")

    plt.tight_layout()
    output_path = FIGURES_DIR / "figs4_mds_vs_keyword_counts.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    return fig


if __name__ == "__main__":
    make_figure()
