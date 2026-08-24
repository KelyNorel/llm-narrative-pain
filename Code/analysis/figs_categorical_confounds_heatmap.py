"""Reproduce the categorical-confounds heatmap (Supplement; exact figure
number TBD -- rename this file once known): for each cohort, Kruskal-Wallis
/ Mann-Whitney p-values of each categorical confound against the nine
LLM metrics. Panels stacked vertically, CLBP/MDD/HC top to bottom;
columns follow Fig. 2's metric order and display names.

Gray "n/a" cells mark confounds where the test could not be computed
in that cohort (fewer than two categories had >=3 participants), not
"tested and not significant" -- see confound_analysis.py's docstring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confound_analysis import load_confound_data, run_all_cohorts, COHORT_COLOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COHORT_ORDER = ["CLBP", "MDD", "HC"]
CONFOUND_ORDER = ["Sex", "Marital status", "Employment status", "Income bucket", "Race", "Ethnicity"]

# metric order/display names matching Fig. 2 (fig2_llm_ratings_by_cohort.py)
METRIC_ORDER = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
METRIC_DISPLAY_NAMES = [
    "Physical Pain", "QoL#", "Emotional Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency Deficit", "Narrative Fragmentation",
]


def make_figure():
    df = load_confound_data()
    df_all = run_all_cohorts(df)
    cat_tests = df_all[df_all["Test"].isin(["Mann-Whitney U", "Kruskal-Wallis H"])].copy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=True)
    last_im = None

    for row_idx, (ax, cohort) in enumerate(zip(axes, COHORT_ORDER)):
        sub = cat_tests[cat_tests["Cohort"] == cohort]
        piv_p = sub.pivot(index="Confound", columns="Metric", values="p_raw").reindex(index=CONFOUND_ORDER, columns=METRIC_ORDER)
        piv_sig = sub.pivot(index="Confound", columns="Metric", values="sig_FDR").reindex(index=CONFOUND_ORDER, columns=METRIC_ORDER)
        not_tested = piv_p.isna()  # test skipped, not "p=1.0"

        neg_log_p = -np.log10(piv_p.clip(lower=1e-10))
        vmax = np.nanmax(neg_log_p.values) if np.isfinite(np.nanmax(neg_log_p.values)) else 1
        annot = piv_p.map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")

        im = sns.heatmap(
            neg_log_p, ax=ax, cmap="Reds", vmin=0, vmax=vmax,
            annot=annot, fmt="", linewidths=0.6, annot_kws={"size": 21},
            cbar=False,
            xticklabels=METRIC_DISPLAY_NAMES, yticklabels=CONFOUND_ORDER, mask=not_tested,
        )
        if cohort == "HC":
            last_im = im

        for i, conf in enumerate(CONFOUND_ORDER):
            for j, met in enumerate(METRIC_ORDER):
                if not_tested.loc[conf, met]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#D3D3D3", edgecolor="white", linewidth=0.4))
                    ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=18, color="dimgray", style="italic")
                # NaN is truthy in Python, so require an actual True, not just any truthy value
                elif pd.notna(piv_sig.loc[conf, met]) and piv_sig.loc[conf, met]:
                    ax.text(j + 0.5, i + 0.82, "*", ha="center", va="center", fontsize=28, color="black", fontweight="bold")

        n = df[df["dx"] == cohort].shape[0]
        ax.set_title(f"{cohort}  (n={n})", fontsize=30, color=COHORT_COLOR[cohort], fontweight="bold", pad=16)
        ax.set_yticklabels(CONFOUND_ORDER, rotation=0, fontsize=22)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if row_idx == len(COHORT_ORDER) - 1:
            ax.set_xticklabels(METRIC_DISPLAY_NAMES, rotation=35, ha="right", fontsize=22)

    fig.suptitle(
        "Categorical Confounds vs LLM Metrics -- per Cohort\n"
        "(cell value = p_raw, color = -log10(p), scale per cohort; * = FDR q < 0.05, corrected within cohort; "
        "gray/n.a. = too few participants per category to test)",
        fontsize=18, y=1.01,
    )
    plt.tight_layout()

    # dedicated colorbar axis to the right of all three panels (HC's scale,
    # shown for reference -- each panel's own vmax differs), so it doesn't
    # shrink that panel's width relative to the other two
    hc_pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([hc_pos.x1 + 0.03, hc_pos.y0, 0.02, hc_pos.height])
    cbar = fig.colorbar(last_im.collections[0], cax=cbar_ax)
    cbar.set_label("-log10(p_raw)", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    output_path = FIGURES_DIR / "figs_categorical_confounds_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    return fig, df_all


if __name__ == "__main__":
    make_figure()
