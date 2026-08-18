"""Reproduce Table S7: SD, CV, and ICC(1,1) of the LLM's scores across
repeated temperature=0 calls on the same transcript, per metric.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temperature_variability import load_variability_data, per_subject_stats, compute_icc, METRIC_COLUMNS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TEMP_VARIABILITY_DIR, FIGURES_DIR

import matplotlib.pyplot as plt
import seaborn as sns


def make_table():
    df = load_variability_data()
    subject_stats = per_subject_stats(df)
    icc = compute_icc(df)

    summary = subject_stats.groupby("metric").agg(
        n_subjects=("study_id", "nunique"),
        mean_sd=("sd", "mean"),
        mean_cv_pct=("cv_pct", "mean"),
        median_cv_pct=("cv_pct", "median"),
        n_undefined_cv=("cv_pct", lambda s: s.isna().sum()),
    ).reindex(METRIC_COLUMNS).reset_index()
    summary = summary.merge(icc, on="metric")

    subject_stats_path = TEMP_VARIABILITY_DIR / "per_subject_variability.csv"
    summary_path = TEMP_VARIABILITY_DIR / "table_s7_output_variability.csv"
    subject_stats.to_csv(subject_stats_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved -> {subject_stats_path}")
    print(f"Saved -> {summary_path}")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=subject_stats, x="metric", y="sd", ax=axes[0], order=METRIC_COLUMNS)
    axes[0].set_title("Per-subject SD across runs")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=subject_stats, x="metric", y="cv_pct", ax=axes[1], order=METRIC_COLUMNS)
    axes[1].set_title("Per-subject CV (%) across runs\n(undefined/unstable near a mean of 0 -- see docstring)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    output_path = FIGURES_DIR / "table_s7_output_variability.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")

    return summary, subject_stats


if __name__ == "__main__":
    make_table()
