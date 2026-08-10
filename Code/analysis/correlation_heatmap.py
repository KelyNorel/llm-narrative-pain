"""LLM-metric vs. clinical-questionnaire correlation heatmap (Spearman,
FDR-corrected), matching the paper's Fig. 3 layout: cohort1 | median |
cohort2 | median, with a gray separator column before each median and
bold annotations for FDR-significant cells.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def _spearman_with_fdr(df: pd.DataFrame, llm_metrics: list, clinical_scores: list, fdr_alpha: float):
    """Spearman rho for every (LLM metric, clinical score) pair, with
    Benjamini-Hochberg FDR correction applied across the whole matrix."""
    n_llm, n_clinical = len(llm_metrics), len(clinical_scores)
    corr_matrix = np.full((n_llm, n_clinical), np.nan)
    p_values = np.full((n_llm, n_clinical), np.nan)

    for i, llm_metric in enumerate(llm_metrics):
        for j, clinical_score in enumerate(clinical_scores):
            mask = df[[llm_metric, clinical_score]].notna().all(axis=1)
            if mask.sum() > 2:
                corr, pval = spearmanr(df.loc[mask, llm_metric], df.loc[mask, clinical_score])
                corr_matrix[i, j] = corr
                p_values[i, j] = pval

    p_flat = p_values.flatten()
    valid = ~np.isnan(p_flat)
    fdr_significant_flat = np.full(p_flat.shape, False)
    if valid.sum() > 0:
        _, pvals_corrected, _, _ = multipletests(p_flat[valid], alpha=fdr_alpha, method="fdr_bh")
        fdr_significant_flat[valid] = pvals_corrected < fdr_alpha

    return corr_matrix, fdr_significant_flat.reshape((n_llm, n_clinical))


def plot_combined_llm_vs_clinical_correlation(
    df_cohort1: pd.DataFrame,
    df_cohort2: pd.DataFrame,
    llm_metrics: list,
    clinical_scores_cohort1: list,
    clinical_scores_cohort2: list,
    llm_names: list = None,
    clinical_names_cohort1: list = None,
    clinical_names_cohort2: list = None,
    cohort1_label: str = "CLBP",
    cohort2_label: str = "MDD",
    figsize: tuple = (20, 10),
    fdr_alpha: float = 0.05,
    annot_fontsize: int = 10,
    label_fontsize: int = 12,
    fn: Path = None,
):
    """Single heatmap combining both cohorts + a per-cohort median column.

    Layout: cohort1 clinical scores | [gray] | cohort1 median | [white
    gap] | cohort2 clinical scores | [gray] | cohort2 median.

    Returns (combined_df, {"fdr1": bool array, "fdr2": bool array}).
    """
    llm_names = llm_names or llm_metrics
    clinical_names_cohort1 = clinical_names_cohort1 or clinical_scores_cohort1
    clinical_names_cohort2 = clinical_names_cohort2 or clinical_scores_cohort2

    corr1, fdr1 = _spearman_with_fdr(df_cohort1, llm_metrics, clinical_scores_cohort1, fdr_alpha)
    corr2, fdr2 = _spearman_with_fdr(df_cohort2, llm_metrics, clinical_scores_cohort2, fdr_alpha)

    n_llm = len(llm_metrics)
    median_corr1 = np.nanmedian(corr1, axis=1).reshape(-1, 1)
    median_corr2 = np.nanmedian(corr2, axis=1).reshape(-1, 1)

    sep_gray = np.full((n_llm, 1), -0.5)  # neutral value, rendered as a gray patch (see below)
    sep_white = np.full((n_llm, 1), np.nan)  # NaN -> masked out (white gap)

    combined_corr = np.hstack([corr1, sep_gray, median_corr1, sep_white, corr2, sep_gray, median_corr2])

    n_clinical1, n_clinical2 = len(clinical_names_cohort1), len(clinical_names_cohort2)
    combined_fdr = np.zeros(combined_corr.shape, dtype=bool)
    combined_fdr[:, :n_clinical1] = fdr1
    mdd_start = n_clinical1 + 1 + 1 + 1  # skip gray sep, median col, white sep
    combined_fdr[:, mdd_start:mdd_start + n_clinical2] = fdr2

    combined_columns = (
        list(clinical_names_cohort1) + [""] + [f"{cohort1_label}\nMedian"]
        + [""] + list(clinical_names_cohort2) + [""] + [f"{cohort2_label}\nMedian"]
    )

    gray_sep_1 = n_clinical1
    gray_sep_2 = n_clinical1 + 1 + 1 + 1 + n_clinical2

    annot_array = np.empty_like(combined_corr, dtype=object)
    for i in range(n_llm):
        for j in range(combined_corr.shape[1]):
            if j in (gray_sep_1, gray_sep_2) or np.isnan(combined_corr[i, j]):
                annot_array[i, j] = ""
            else:
                val = combined_corr[i, j]
                annot_array[i, j] = f"$\\mathbf{{{val:.2f}}}$" if combined_fdr[i, j] else f"{val:.2f}"

    combined_df = pd.DataFrame(combined_corr, index=llm_names, columns=combined_columns)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.isnan(combined_corr)
    sns.heatmap(
        combined_df, annot=annot_array, fmt="", cmap="RdBu_r", vmin=0, vmax=1, center=0,
        square=False, linewidths=1.5, linecolor="white",
        cbar_kws={"label": "Correlation", "shrink": 0.8, "ticks": [-1, -0.5, 0, 0.5, 1.0]},
        annot_kws={"size": annot_fontsize}, mask=mask, ax=ax,
    )

    cohort1_center = (0 + (n_clinical1 + 2)) / 2 + 0.5
    cohort2_start = n_clinical1 + 3
    cohort2_center = (cohort2_start + (cohort2_start + n_clinical2 + 2)) / 2 - 0.5
    ax.text(cohort1_center, -0.3, cohort1_label, ha="center", va="bottom", fontsize=label_fontsize + 2, fontweight="bold")
    ax.text(cohort2_center, -0.3, cohort2_label, ha="center", va="bottom", fontsize=label_fontsize + 2, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    for i in range(n_llm):
        for sep_idx in (gray_sep_1, gray_sep_2):
            ax.add_patch(plt.Rectangle((sep_idx, i), 1, 1, facecolor="#d3d3d3", edgecolor="white", linewidth=1.5))

    plt.tight_layout()

    if fn:
        fn = Path(fn)
        fn.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.savefig(fn.with_suffix(".tiff"), dpi=600, bbox_inches="tight", format="tiff")

    print(f"{cohort1_label}: {fdr1.sum()} / {(~np.isnan(corr1)).sum()} FDR-significant")
    print(f"{cohort2_label}: {fdr2.sum()} / {(~np.isnan(corr2)).sum()} FDR-significant")

    return combined_df, {"fdr1": fdr1, "fdr2": fdr2}
