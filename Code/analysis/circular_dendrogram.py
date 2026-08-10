"""Circular (radial) dendrogram of the GLasso partial correlation
matrix (Fig. 4 / eFigure 2 panel C).

Renders in black/white; the published figure's colored "Emotional" /
"Cognitive" cluster sectors and labels were added manually afterward
(not part of this code — cluster boundaries were chosen by eye from
the leaf angles, per the paper's Methods).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

PARTIAL_CORR_THRESHOLD = 0.1
LINKAGE_METHOD = "ward"
BRANCH_COLOR = "#1F3FBF"
LABEL_FONTSIZE = 24
ANGULAR_SPAN_FRACTION = 0.85  # leaves a gap in the circle


def compute_circular_coords(partial_corr: np.ndarray, labels: list, threshold: float = PARTIAL_CORR_THRESHOLD, method: str = LINKAGE_METHOD):
    """Hierarchical-cluster the (thresholded) partial correlation matrix
    and convert its dendrogram to polar coordinates.

    Returns (icoord_rad, radial_dist, leaf_angles, ordered_labels).
    """
    n = len(labels)
    corr_matrix = partial_corr.copy()
    np.fill_diagonal(corr_matrix, 1)
    corr_matrix[np.abs(corr_matrix) < threshold] = 0

    dist_matrix = 1 - corr_matrix
    Z = linkage(squareform(dist_matrix, checks=False), method=method)
    dn = dendrogram(Z, labels=labels, no_plot=True)

    icoord = np.array(dn["icoord"])
    dcoord = np.array(dn["dcoord"])
    ordered_labels = dn["ivl"]

    imin, imax = icoord.min(), icoord.max()
    angular_span = 2 * np.pi * ANGULAR_SPAN_FRACTION
    icoord_rad = (icoord - imin) / (imax - imin) * angular_span

    max_dist = dcoord.max()
    radial_dist = max_dist - dcoord

    leaf_positions = []
    for i in range(len(icoord)):
        if dcoord[i, 0] == 0:
            leaf_positions.append((icoord_rad[i, 0], icoord[i, 0]))
        if dcoord[i, 3] == 0:
            leaf_positions.append((icoord_rad[i, 3], icoord[i, 3]))
    leaf_positions = sorted(set(leaf_positions), key=lambda x: x[1])
    leaf_angles = [angle for angle, _ in leaf_positions]

    return icoord_rad, radial_dist, leaf_angles, ordered_labels


def plot_circular_dendrogram(icoord_rad: np.ndarray, radial_dist: np.ndarray, leaf_angles: list, labels: list, title: str = None, fn: Path = None):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for xs, ys in zip(icoord_rad, radial_dist):
        ax.plot([xs[0], xs[1]], [ys[0], ys[1]], color=BRANCH_COLOR, lw=3.0, zorder=2)
        theta_arc = np.linspace(xs[1], xs[2], 30)
        ax.plot(theta_arc, np.ones(30) * ys[1], color=BRANCH_COLOR, lw=3.0, zorder=2)
        ax.plot([xs[2], xs[3]], [ys[2], ys[3]], color=BRANCH_COLOR, lw=3.0, zorder=2)

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)

    max_dist = radial_dist.max()
    label_radius = max_dist + 0.4
    for angle, label in zip(leaf_angles, labels):
        # Rotate each label to point radially outward, like spokes, rather
        # than keeping text horizontal. Horizontal labels only avoid
        # colliding with a neighbor if the angular gap between leaves is
        # wide enough for their *unrotated* pixel width — with evenly
        # spaced leaves, a long label (e.g. single-word "Catastrophizing"
        # next to a two-line name) can still collide with its neighbor
        # even though the angular gap is the same as every other pair.
        # Radial text instead only needs room along the arc at its own
        # angle, which scales with label_radius independent of neighbors.
        screen_deg = (np.rad2deg(angle) * -1 + 90) % 360  # after theta_direction/theta_offset
        if 90 < screen_deg < 270:
            rotation = screen_deg + 180
            ha = "right"
        else:
            rotation = screen_deg
            ha = "left"
        ax.plot([angle, angle], [max_dist, label_radius - 0.05], color="gray", lw=1.5, alpha=0.4)
        ax.text(
            angle, label_radius, label, fontsize=LABEL_FONTSIZE, fontweight="600",
            ha=ha, va="center", rotation=rotation, rotation_mode="anchor",
        )

    ax.set_ylim(0, max_dist + 0.95)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    if title:
        plt.title(title, fontsize=15, pad=40, fontweight="bold")

    plt.tight_layout()
    if fn:
        fn = Path(fn)
        fn.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fn, dpi=300, bbox_inches="tight", facecolor="white")
    return fig
