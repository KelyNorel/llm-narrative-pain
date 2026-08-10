"""Boxplot-by-cohort figures with Kruskal-Wallis + pairwise Mann-Whitney U
significance brackets, matching the paper's figures (e.g. Fig. 2).

`plot_metrics_comparison` is generic over any set of metrics grouped
into panels — it is the one function behind every boxplot figure in
the paper, not just Fig. 2. For each metric: Kruskal-Wallis omnibus
test first; pairwise Mann-Whitney U (two-sided) is only drawn for
pairs listed in `pairwise_comparisons` when the omnibus test is
significant, matching the paper's stats.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

PASTEL_COLORS = {
    "CLBP": "#aec7e8",
    "MDD": "#ffbb78",
    "HC": "#98df8a",
    "r/chronicpain": "#aec7e8",
    "r/depressed": "#ffbb78",
}


def _remove_outliers(data):
    """Boxplots are drawn without outliers (whiskers/box only); individual
    points and statistics still use the original data."""
    if len(data) < 4:
        return data
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    return data[(data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)]


def create_significance_caption(all_significant_results: list, group_sizes: dict) -> str:
    """Build a figure-caption string listing sample sizes and every
    pairwise comparison that passed both significance and min_effect_size."""
    n_parts = [f"{group} (N={n})" for group, n in group_sizes.items()]
    caption_text = "; ".join(n_parts) + ". "

    if not all_significant_results:
        return caption_text + "No significant pairwise differences detected."

    caption_lines = []
    for result in all_significant_results:
        p_val = result["p_value"]
        if p_val < 0.001:
            sig_symbol, p_text = "***", "p<0.001"
        elif p_val < 0.01:
            sig_symbol, p_text = "**", f"p={p_val:.3f}"
        elif p_val < 0.05:
            sig_symbol, p_text = "*", f"p={p_val:.3f}"
        else:
            continue
        caption_lines.append(
            f"{result['metric']}: {result['group1']} vs {result['group2']} ({p_text}, {sig_symbol})"
        )

    if not caption_lines:
        return caption_text + "No significant pairwise differences detected."

    caption_text += "Significant pairwise comparisons (Mann-Whitney U test): " + "; ".join(caption_lines)
    caption_text += ". * p < 0.05, ** p < 0.01, *** p < 0.001"
    return caption_text


def plot_metrics_comparison(
    df,
    metrics_groups: list,
    metric_names_groups: list,
    main_title: str = "",
    y_limits: list = None,
    y_labels: list = None,
    layout: str = "horizontal",
    group_labels: list = None,
    min_effect_size: float = 0.5,
    add_pval_table: bool = True,
    group_mapping: dict = None,
    pairwise_comparisons: list = None,
    dx_col: str = "dx",
    *,
    fn=None,
):
    """Multi-panel boxplot figure with independent y-axis scales per panel.

    - metrics_groups: list of panels; each panel is a list of metric column names.
    - metric_names_groups: matching list of display names for each metric.
    - group_labels: cohort order, e.g. ["CLBP", "MDD", "HC"].
    - pairwise_comparisons: list of (group_a, group_b, bracket_color) to draw
      when a metric's Kruskal-Wallis omnibus test is significant (p < 0.05).
    - min_effect_size: minimum |mean difference| for a pair to be listed in
      the returned caption (does not affect what's drawn on the plot).
    - fn: optional path to save the figure to.

    Returns (fig, caption_text).
    """
    sns.set_context("talk", font_scale=1.5)
    SCALE = 1.25

    n_plot_subplots = len(metrics_groups)

    if layout == "horizontal" and n_plot_subplots > 1:
        if add_pval_table:
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.3)
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        else:
            fig, axes = plt.subplots(1, n_plot_subplots, figsize=(10 * n_plot_subplots, 8))
    else:
        if add_pval_table:
            fig = plt.figure(figsize=(20, 32))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.4)
            axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        else:
            fig, axes = plt.subplots(n_plot_subplots, 1, figsize=(20, 20))

    if isinstance(axes, np.ndarray):
        axes = axes.tolist()
    elif not isinstance(axes, list):
        axes = [axes]

    fig.suptitle(main_title, fontsize=28 * SCALE, fontweight="bold", y=0.998)

    if group_labels is None:
        group_labels = ["CLBP", "MDD", "HC"]
    if group_mapping is None:
        group_mapping = {label: label for label in group_labels}
    n_groups = len(group_labels)

    group_sizes = {}
    first_metric = metrics_groups[0][0]
    for group_label in group_labels:
        data_label = group_mapping[group_label]
        group_sizes[group_label] = len(df[df[dx_col] == data_label][first_metric].dropna())

    all_significant_results = []

    for subplot_idx, (metrics, metric_names) in enumerate(zip(metrics_groups, metric_names_groups)):
        ax = axes[subplot_idx]

        all_data = []
        all_positions = []
        position = 0
        for metric, metric_name in zip(metrics, metric_names):
            for group_label in group_labels:
                data_label = group_mapping[group_label]
                data = df[df[dx_col] == data_label][metric].dropna()
                all_data.append(data)
                position += 1
                all_positions.append(position)
            position += 1  # gap between metrics

        all_positions = all_positions[: len(all_data)]

        data_no_outliers = [_remove_outliers(d) if len(d) > 0 else d for d in all_data]

        bp = ax.boxplot(
            data_no_outliers, positions=all_positions,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=4),
            boxprops=dict(linewidth=3),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            widths=0.75,
        )

        # Color each box by cohort; hatch large-N boxes (N >= 100)
        for i, patch in enumerate(bp["boxes"]):
            group_label = group_labels[i % n_groups]
            N = len(all_data[i])
            patch.set_facecolor(PASTEL_COLORS.get(group_label, "#cccccc"))
            patch.set_alpha(0.7)
            if N >= 100:
                patch.set_hatch("///")
                patch.set_edgecolor("black")
                patch.set_linewidth(2)

        # Draw a flat median line for near-zero-IQR (collapsed) boxes
        for data_orig, pos in zip(all_data, all_positions):
            if len(data_orig) > 3:
                Q1, Q3 = np.percentile(data_orig, [25, 75])
                if (Q3 - Q1) < 0.1:
                    median = np.median(data_orig)
                    ax.plot(
                        [pos - 0.35, pos + 0.35], [median, median],
                        color="black", linewidth=8, zorder=4, solid_capstyle="butt",
                    )

        # Individual jittered points, only when N < 100 (avoids overplotting)
        for data, pos in zip(data_no_outliers, all_positions):
            if 0 < len(data) < 100:
                x = np.random.normal(pos, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.4, color="black", s=15, zorder=3)

        subplot_data = np.concatenate([d for d in all_data if len(d) > 0])
        if len(subplot_data) == 0:
            continue
        data_max = np.max(subplot_data)

        if y_limits and subplot_idx < len(y_limits) and y_limits[subplot_idx] != "auto":
            ax.set_ylim(y_limits[subplot_idx])
        else:
            y_min = 0
            if data_max <= 10:
                y_max = 12
            else:
                y_max = data_max + 0.2 * (data_max - np.min(subplot_data))
            ax.set_ylim(y_min, y_max)
            if data_max <= 10 and y_max == 12:
                yticks = ax.get_yticks()
                ax.set_yticks(yticks)
                ax.set_yticklabels([str(int(y)) if y <= 10 else "" for y in yticks])

        # Kruskal-Wallis omnibus per metric, then pairwise Mann-Whitney U
        # brackets for the requested pairs only if the omnibus test is significant.
        for metric_idx, metric_name in enumerate(metric_names):
            metric_positions = [all_positions[metric_idx * n_groups + i] for i in range(n_groups)]
            metric_data = [all_data[metric_idx * n_groups + i] for i in range(n_groups)]

            valid_metric_data = [d for d in metric_data if len(d) > 0]
            if len(valid_metric_data) < 2:
                continue
            _, kw_pval = kruskal(*valid_metric_data)
            if kw_pval >= 0.05 or pairwise_comparisons is None:
                continue

            line_offset = 0
            for comp_group1, comp_group2, line_color in pairwise_comparisons:
                if comp_group1 not in group_labels or comp_group2 not in group_labels:
                    continue
                idx1, idx2 = group_labels.index(comp_group1), group_labels.index(comp_group2)
                if len(metric_data[idx1]) == 0 or len(metric_data[idx2]) == 0:
                    continue

                _, p_pair = mannwhitneyu(metric_data[idx1], metric_data[idx2], alternative="two-sided")
                stars = "***" if p_pair < 0.001 else "**" if p_pair < 0.01 else "*" if p_pair < 0.05 else ""

                mean_diff = abs(np.mean(metric_data[idx1]) - np.mean(metric_data[idx2]))
                if mean_diff >= min_effect_size and p_pair < 0.05:
                    all_significant_results.append({
                        "metric": metric_name, "group1": comp_group1, "group2": comp_group2,
                        "p_value": p_pair, "effect_size": mean_diff,
                    })

                if not stars:
                    continue
                pos1, pos2 = metric_positions[idx1], metric_positions[idx2]
                current_ylim = ax.get_ylim()
                y_range = current_ylim[1] - current_ylim[0]
                line_y = current_ylim[1] - (0.05 + line_offset * 0.08) * y_range
                mid_pos = (pos1 + pos2) / 2
                gap = 0.3
                ax.plot([pos1, mid_pos - gap], [line_y, line_y], color=line_color, linewidth=3.5, solid_capstyle="butt")
                ax.plot([mid_pos + gap, pos2], [line_y, line_y], color=line_color, linewidth=3.5, solid_capstyle="butt")
                tick_height = 0.015 * y_range
                ax.plot([pos1, pos1], [line_y, line_y - tick_height], color=line_color, linewidth=3.5)
                ax.plot([pos2, pos2], [line_y, line_y - tick_height], color=line_color, linewidth=3.5)
                ax.text(mid_pos, line_y, stars, ha="center", va="center", fontsize=22, fontweight="bold", color="black")
                line_offset += 1

        # Metric title(s) above the panel
        if len(metrics) == 1 and not y_labels:
            ax.set_title(metric_names[0], fontsize=28 * SCALE, fontweight="bold", pad=30)
        elif len(metrics) > 1:
            saved_ylim = ax.get_ylim()
            y_range = saved_ylim[1] - saved_ylim[0]
            title_height = saved_ylim[1] + 0.08 * y_range
            for i in range(len(metrics)):
                start_idx, end_idx = i * n_groups, i * n_groups + n_groups - 1
                center_pos = (all_positions[start_idx] + all_positions[end_idx]) / 2
                ax.text(center_pos, title_height, metric_names[i], ha="center", va="bottom",
                        fontsize=24 * SCALE, fontweight="normal", clip_on=False)
            ax.set_ylim(saved_ylim)

        # Cohort labels below the last panel (vertical) or every panel (horizontal)
        show_labels = True if layout == "horizontal" else (subplot_idx == n_plot_subplots - 1)
        if show_labels:
            saved_ylim = ax.get_ylim()
            for i in range(len(metrics)):
                for j in range(n_groups):
                    pos = all_positions[i * n_groups + j]
                    ax.text(pos, saved_ylim[0] - 0.08 * (saved_ylim[1] - saved_ylim[0]), group_labels[j],
                            ha="right", va="top", fontsize=24 * SCALE, rotation=60, clip_on=False)
            ax.set_ylim(saved_ylim)

        # Dashed separators between metrics within a panel
        if len(metrics) > 1:
            for i in range(1, len(metrics)):
                prev_end = (i - 1) * n_groups + n_groups - 1
                curr_start = i * n_groups
                x_pos = (all_positions[prev_end] + all_positions[curr_start]) / 2
                ax.axvline(x=x_pos, color="gray", linestyle="--", alpha=0.5, linewidth=2)

        ax.set_ylabel(y_labels[subplot_idx] if y_labels and subplot_idx < len(y_labels) else "Score", fontsize=26 * SCALE)
        ax.set_xticklabels([])
        ax.set_xlim(0, max(all_positions) + 1)
        ax.tick_params(axis="y", labelsize=22 * SCALE)

        panel_label = chr(65 + subplot_idx)  # A, B, C, ...
        # A small y-bump (in addition to the x-offset) clears wide y-tick
        # labels (e.g. "8000") without reaching high enough to collide with
        # a figure-level main_title, which sits close to the figure top.
        ax.text(-0.04, 1.08, panel_label, transform=ax.transAxes, fontsize=36 * SCALE, fontweight="bold", va="top", ha="left")

    caption_text = create_significance_caption(all_significant_results, group_sizes)

    if layout == "horizontal":
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, wspace=0.3)
    else:
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.96, hspace=0.4)

    if fn:
        fn.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    sns.set_context("notebook")
    return fig, caption_text.replace("\n", " ")
