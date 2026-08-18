"""Reproduce the categorical-confounds heatmap (Supplement; exact figure
number TBD -- rename this file once known): for each cohort, Kruskal-Wallis
/ Mann-Whitney p-values of each categorical confound against the nine
LLM metrics.

Gray "n/a" cells mark confounds where the test could not be computed
in that cohort (fewer than two categories had >=3 participants), not
"tested and not significant" -- see confound_analysis.py's docstring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confound_analysis import load_confound_data, run_all_cohorts, COHORTS, COHORT_COLOR, LLM_METRICS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CONFOUND_ORDER = ["Sex", "Marital status", "Employment status", "Income bucket", "Race", "Ethnicity"]


def make_figure():
    df = load_confound_data()
    df_all = run_all_cohorts(df)
    cat_tests = df_all[df_all["Test"].isin(["Mann-Whitney U", "Kruskal-Wallis H"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

    for ax, cohort in zip(axes, COHORTS):
        sub = cat_tests[cat_tests["Cohort"] == cohort]
        piv_p = sub.pivot(index="Confound", columns="Metric", values="p_raw").reindex(index=CONFOUND_ORDER, columns=LLM_METRICS)
        piv_sig = sub.pivot(index="Confound", columns="Metric", values="sig_FDR").reindex(index=CONFOUND_ORDER, columns=LLM_METRICS)
        not_tested = piv_p.isna()  # test skipped, not "p=1.0"

        neg_log_p = -np.log10(piv_p.clip(lower=1e-10))
        vmax = np.nanmax(neg_log_p.values) if np.isfinite(np.nanmax(neg_log_p.values)) else 1
        annot = piv_p.map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")

        im = sns.heatmap(
            neg_log_p, ax=ax, cmap="Reds", vmin=0, vmax=vmax,
            annot=annot, fmt="", linewidths=0.4, annot_kws={"size": 9},
            cbar=(cohort == "MDD"), cbar_kws={"label": "-log10(p_raw)", "shrink": 0.8},
            xticklabels=LLM_METRICS, yticklabels=CONFOUND_ORDER, mask=not_tested,
        )

        for i, conf in enumerate(CONFOUND_ORDER):
            for j, met in enumerate(LLM_METRICS):
                if not_tested.loc[conf, met]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#D3D3D3", edgecolor="white", linewidth=0.4))
                    ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=8, color="dimgray", style="italic")
                # NaN is truthy in Python, so require an actual True, not just any truthy value
                elif pd.notna(piv_sig.loc[conf, met]) and piv_sig.loc[conf, met]:
                    ax.text(j + 0.5, i + 0.82, "*", ha="center", va="center", fontsize=13, color="black", fontweight="bold")

        n = df[df["dx"] == cohort].shape[0]
        ax.set_title(f"{cohort}  (n={n})", fontsize=12, color=COHORT_COLOR[cohort], fontweight="bold", pad=10)
        ax.set_xticklabels(LLM_METRICS, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticklabels(CONFOUND_ORDER, rotation=0, fontsize=9.5)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        "Categorical Confounds vs LLM Metrics -- per Cohort\n"
        "(cell value = p_raw, color = -log10(p); * = FDR q < 0.05, corrected within cohort; "
        "gray/n.a. = too few participants per category to test)",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()

    output_path = FIGURES_DIR / "figs_categorical_confounds_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    return fig, df_all


if __name__ == "__main__":
    make_figure()
