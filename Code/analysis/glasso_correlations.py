"""Graphical Lasso bivariate/partial correlation matrices between the
nine LLM metrics, with bootstrap + FDR-corrected significance
(Fig. 4 panels A/B and the MDD supplementary equivalent).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedFormatter
from sklearn.covariance import GraphicalLassoCV, graphical_lasso
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def apply_glasso(data: pd.DataFrame, alpha: float = None, random_state: int = 42):
    """Fit Graphical Lasso on rank-transformed data.

    If alpha is None, selects it via 5-fold GraphicalLassoCV. Returns
    (precision, covariance, alpha_used).
    """
    X_ranked = np.apply_along_axis(rankdata, 0, data.values)

    if alpha is None:
        model = GraphicalLassoCV(cv=5, alphas=20, max_iter=100)
        model.fit(X_ranked)
        return model.precision_, model.covariance_, model.alpha_

    emp_cov = np.cov(X_ranked.T)
    covariance, precision = graphical_lasso(emp_cov=emp_cov, alpha=alpha, max_iter=100)
    return precision, covariance, alpha


def precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1)
    return partial_corr


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(covariance))
    return covariance / np.outer(d, d)


def bootstrap_pvalues(data: pd.DataFrame, alpha: float, n_bootstrap: int = 1000, threshold: float = 0.01):
    """Bootstrap p-values for both direct (GLasso covariance-derived) and
    partial correlations: the fraction of bootstrap resamples where the
    edge's magnitude falls at or below `threshold` (i.e. is effectively zero).

    GLasso occasionally fails to fit a bootstrap resample (ill-conditioned
    covariance); those draws are excluded from the fraction rather than
    counted as zero-correlation evidence, which would inflate p-values.
    """
    n_samples, n_features = data.shape
    direct_boot = []
    partial_boot = []

    rng = np.random.default_rng(42)
    n_failed = 0
    for _ in tqdm(range(n_bootstrap), desc="bootstrap"):
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_data = data.iloc[boot_indices].reset_index(drop=True)
        try:
            precision, covariance, _ = apply_glasso(boot_data, alpha=alpha)
            direct_boot.append(covariance_to_correlation(covariance))
            partial_boot.append(precision_to_partial_correlation(precision))
        except Exception:
            n_failed += 1
            continue

    if n_failed:
        print(f"  {n_failed}/{n_bootstrap} bootstrap resamples failed to converge (excluded, not zero-filled)")
    direct_boot = np.array(direct_boot)
    partial_boot = np.array(partial_boot)

    direct_pvals = np.zeros((n_features, n_features))
    partial_pvals = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            direct_pvals[i, j] = direct_pvals[j, i] = np.mean(np.abs(direct_boot[:, i, j]) <= threshold)
            partial_pvals[i, j] = partial_pvals[j, i] = np.mean(np.abs(partial_boot[:, i, j]) <= threshold)
    np.fill_diagonal(direct_pvals, 1.0)
    np.fill_diagonal(partial_pvals, 1.0)
    return direct_pvals, partial_pvals


def plot_glasso_with_fdr(
    direct_corr: np.ndarray,
    partial_corr: np.ndarray,
    direct_pvals: np.ndarray,
    partial_pvals: np.ndarray,
    feature_names: list,
    fdr_alpha: float = 0.05,
    corr_threshold: float = 0.01,
    figsize: tuple = (24, 11),
):
    """Side-by-side lower-triangle heatmaps: bivariate correlation and
    GLasso partial correlation, edges in bold where FDR-corrected q < fdr_alpha.

    Returns (fig, {"direct_reject": bool matrix, "partial_reject": bool matrix}).
    """
    triu = np.triu_indices_from(direct_corr, k=1)

    direct_reject, _, _, _ = multipletests(direct_pvals[triu], alpha=fdr_alpha, method="fdr_bh")
    partial_reject, _, _, _ = multipletests(partial_pvals[triu], alpha=fdr_alpha, method="fdr_bh")

    direct_reject_matrix = np.zeros_like(direct_pvals, dtype=bool)
    partial_reject_matrix = np.zeros_like(partial_pvals, dtype=bool)
    direct_reject_matrix[triu] = direct_reject_matrix.T[triu] = direct_reject
    partial_reject_matrix[triu] = partial_reject_matrix.T[triu] = partial_reject

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3)
    mask_triangle = np.tril(np.ones_like(direct_corr, dtype=bool))

    def _annotate(corr, reject_matrix, mask_near_zero=False):
        annot = np.empty_like(corr, dtype=object)
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if mask_near_zero and abs(corr[i, j]) < corr_threshold:
                    annot[i, j] = ""
                    continue
                text = f"{corr[i, j]:.2f}"
                annot[i, j] = f"$\\mathbf{{{text}}}$" if reject_matrix[i, j] else text
        return annot

    sns.heatmap(
        direct_corr, mask=mask_triangle, xticklabels=feature_names, yticklabels=feature_names,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, cbar=False,
        ax=ax1, annot=_annotate(direct_corr, direct_reject_matrix), fmt="", annot_kws={"size": 14},
    )
    ax1.yaxis.set_major_formatter(FixedFormatter(feature_names))
    ax1.tick_params(axis="both", labelsize=17)
    ax1.set_title("Correlation", fontsize=17, fontweight="bold", pad=15)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    im = sns.heatmap(
        partial_corr, mask=mask_triangle, xticklabels=feature_names, yticklabels=feature_names,
        cmap="RdBu_r", center=0, vmin=0, vmax=1, square=True, linewidths=0.5, cbar=False,
        ax=ax2, annot=_annotate(partial_corr, partial_reject_matrix, mask_near_zero=True),
        fmt="", annot_kws={"size": 14},
    )
    ax2.yaxis.set_major_formatter(FixedFormatter(feature_names))
    ax2.tick_params(axis="both", labelsize=17)
    ax2.set_title("Partial Correlation", fontsize=17, fontweight="bold", pad=15)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    cbar = fig.colorbar(im.collections[0], ax=(ax1, ax2), location="right", shrink=0.6, pad=0.02)
    cbar.set_label("Correlation", rotation=270, labelpad=25, fontsize=22)

    n_total = len(triu[0])
    print(f"Direct: {direct_reject_matrix[triu].sum()}/{n_total} edges FDR-significant (q<{fdr_alpha})")
    print(f"Partial: {partial_reject_matrix[triu].sum()}/{n_total} edges FDR-significant (q<{fdr_alpha})")

    return fig, {"direct_reject": direct_reject_matrix, "partial_reject": partial_reject_matrix}
