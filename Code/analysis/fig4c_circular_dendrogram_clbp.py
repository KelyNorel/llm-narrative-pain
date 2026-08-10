"""Reproduce Fig. 4 panel C: circular dendrogram (Ward linkage) of the
GLasso partial correlation matrix between the nine LLM metrics, in
CLBP patients.

GLasso can be refit live (see fig4ab_glasso_correlations_clbp.py,
which does exactly that for panels A/B) and gives nearly identical
partial correlations. But this dendrogram's cluster topology is
sensitive even to that small a difference — a slightly different
alpha (GraphicalLassoCV drifts across scikit-learn versions) can flip
which side of the threshold a borderline edge like QoL#-Agency Deficit
falls on, changing where a leaf ends up in the tree. That's not a bug
to paper over: this script uses the precomputed partial correlation
matrix (Results/glasso/) so the dendrogram matches the published
figure exactly, and says so here rather than silently recomputing.

Renders in black/white — the published figure's colored "Emotional" /
"Cognitive" sector labels were added manually afterward (cluster
boundaries were chosen by eye from the leaf angles, per the paper's
Methods), not reproduced here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circular_dendrogram import compute_circular_coords, plot_circular_dendrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PARTIAL_CORR_CLBP_CSV, FIGURES_DIR

import pandas as pd


def make_figure():
    partial_corr_df = pd.read_csv(PARTIAL_CORR_CLBP_CSV, index_col=0)
    labels = [c.replace("poor_QoL", "QoL#").replace("_", "\n") for c in partial_corr_df.columns]

    icoord_rad, radial_dist, leaf_angles, ordered_labels = compute_circular_coords(partial_corr_df.values, labels)

    output_path = FIGURES_DIR / "fig4c_circular_dendrogram_clbp.png"
    fig = plot_circular_dendrogram(icoord_rad, radial_dist, leaf_angles, ordered_labels, fn=output_path)
    print(f"Saved -> {output_path}")
    return fig


if __name__ == "__main__":
    make_figure()
