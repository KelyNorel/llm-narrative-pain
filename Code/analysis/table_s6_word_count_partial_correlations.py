"""Reproduce Table S6: partial correlation between word count and each
LLM-derived metric, per cohort, controlling for the other eight metrics.
Graphical Lasso (glasso_correlations.py) is fit on all 9 metrics + word
count together per cohort; the word-count row of the resulting partial
correlation matrix, with bootstrap p-values (1000 resamples), is
reported for each metric. p-values are FDR-corrected (Benjamini-Hochberg)
across all 27 (metric, cohort) pairs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from word_counts import count_words_and_chars
from glasso_correlations import apply_glasso, precision_to_partial_correlation, bootstrap_pvalues

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

import pandas as pd
from statsmodels.stats.multitest import multipletests

COHORTS = ["CLBP", "MDD", "HC"]
METRICS = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
METRIC_DISPLAY_NAMES = {
    "Physical_Pain": "Physical Pain", "poor_QoL": "QoL",
    "Emotional_Pain": "Emotional Pain", "Catastrophizing": "Catastrophizing",
    "Depression": "Depression", "Anxiety": "Anxiety", "Rumination": "Rumination",
    "Agency_Deficit": "Agency Deficit", "Narrative_Fragmentation": "Narrative Fragmentation",
}
NETWORK_COLUMNS = METRICS + ["word_count"]


def load_data() -> pd.DataFrame:
    scores = load_llm_scores("common").rename(columns={"study_id": "Study ID", "dx": "Dx"})
    word_counts = count_words_and_chars().rename(columns={"study_id": "Study ID"})
    word_counts["Study ID"] = word_counts["Study ID"].astype(int)
    return scores.merge(word_counts[["Study ID", "word_count"]], on="Study ID", how="inner")


def make_table() -> pd.DataFrame:
    df = load_data()

    rows = []
    for cohort in COHORTS:
        sub_df = df[df["Dx"] == cohort][NETWORK_COLUMNS].reset_index(drop=True)
        precision, _, alpha = apply_glasso(sub_df, alpha=None)
        partial_corr = precision_to_partial_correlation(precision)
        _, partial_pvals = bootstrap_pvalues(sub_df, alpha, n_bootstrap=1000)

        wc_idx = NETWORK_COLUMNS.index("word_count")
        for metric in METRICS:
            m_idx = NETWORK_COLUMNS.index(metric)
            rows.append(dict(
                Group=cohort, Metric=METRIC_DISPLAY_NAMES[metric], N=len(sub_df), alpha=round(alpha, 3),
                partial_r=partial_corr[wc_idx, m_idx], p_uncorrected=partial_pvals[wc_idx, m_idx],
            ))

    table = pd.DataFrame(rows)
    table["p_FDR"] = multipletests(table["p_uncorrected"], alpha=0.05, method="fdr_bh")[1]

    output_path = RESULTS_DIR / "confounds" / "table_s6_word_count_partial_correlations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    make_table()
