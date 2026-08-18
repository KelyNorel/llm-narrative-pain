"""Reproduce Fig. S3: confusion matrices for Random Forest classifiers
that predict cohort (CLBP/HC/MDD) from pairs of LLM-derived metrics --
Agency Deficit & Rumination, and Agency Deficit & Narrative Fragmentation.
Several other feature pairs and a Logistic Regression baseline were also
tested (see classification_analysis.py); these two Random Forest models
were selected for the supplement as the best 2-feature pairs among the
narrative-structure metrics (Narrative_Fragmentation, Agency_Deficit,
Rumination, Catastrophizing).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classification_analysis import load_classification_data, run_feature_set, CLASS_ORDER

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_SETS = [
    (["Agency_Deficit", "Rumination"], "Agency Deficit & Rumination"),
    (["Narrative_Fragmentation", "Agency_Deficit"], "Agency Deficit & Narrative Fragmentation"),
]


def _annotate(cm: np.ndarray) -> np.ndarray:
    """Cell text: count only off-diagonal, count + row-wise recall % on
    the diagonal (e.g. "61\n(91%)")."""
    row_sums = cm.sum(axis=1)
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                pct = cm[i, j] / row_sums[i] * 100
                annot[i, j] = f"{cm[i, j]}\n({pct:.0f}%)"
            else:
                annot[i, j] = f"{cm[i, j]}"
    return annot


def make_figure():
    df = load_classification_data()

    fig, axes = plt.subplots(1, len(FEATURE_SETS), figsize=(12, 6))
    results = []

    for idx, (ax, (features, display_name)) in enumerate(zip(axes, FEATURE_SETS)):
        result = run_feature_set(df, features)
        results.append(result)
        cm = result["confusion_matrix"]

        sns.heatmap(
            cm, annot=_annotate(cm), fmt="", cmap="Blues",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            cbar=(idx == len(FEATURE_SETS) - 1), ax=ax,
            cbar_kws={"label": ""}, annot_kws={"size": 13, "weight": "bold"},
        )
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title(
            f"Random Forest\n{display_name}\nF1 Macro = {result['f1_macro']:.3f}",
            fontsize=12, fontweight="bold", pad=15,
        )
        ax.set_xlabel("Predicted Label", fontsize=16)
        ax.set_ylabel("True Label" if idx == 0 else "", fontsize=16)

    plt.tight_layout()

    output_path = FIGURES_DIR / "figs3_confusion_matrices_classification.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")

    for r in results:
        print(f"\n{r['features']}: F1 Macro = {r['f1_macro']:.4f}")
        print(f"  best_params = {r['best_params']}")
        print(f"  confusion_matrix ({CLASS_ORDER}):\n{r['confusion_matrix']}")

    return fig, results


if __name__ == "__main__":
    make_figure()
