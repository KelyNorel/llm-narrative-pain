"""Combine eFigure 2 panels A/B and C into the final composite figure
(MDD). Panel C here is black/white — the published figure's colored
"Emotional"/"Cognitive" sectors were added manually afterward; run
this first, then color panel C, then rerun to composite the colored
version if needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from join_panels import join_ab_c

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR


def make_figure():
    output_path = FIGURES_DIR / "efig2_combined_mdd.png"
    join_ab_c(
        FIGURES_DIR / "efig2ab_glasso_correlations_mdd.png",
        FIGURES_DIR / "efig2c_circular_dendrogram_mdd.png",
        output_path,
    )
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    make_figure()
