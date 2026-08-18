"""Cliff's delta and rank-biserial r from a Mann-Whitney U statistic --
non-parametric effect sizes for the pairwise cohort comparisons behind
Table S4. Both are derived directly from U (mathematically equivalent,
opposite sign), not computed independently.
"""


def cliffs_delta_and_rank_biserial(u_stat: float, n1: int, n2: int) -> tuple:
    """`u_stat` is scipy's Mann-Whitney U for (group1, group2), group1
    passed first. Convention: positive delta => group1 tends toward
    higher values than group2."""
    delta = (2 * u_stat) / (n1 * n2) - 1
    r = -delta
    return delta, r


def interpret_cliffs_delta(delta: float) -> str:
    """Magnitude thresholds from Romano et al. (2006)."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"
