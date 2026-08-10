"""Reproduce Fig. 3: Spearman correlations between LLM-derived metrics
and clinical assessment scores, by cohort (CLBP, MDD), FDR-corrected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores
from correlation_heatmap import plot_combined_llm_vs_clinical_correlation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLINICAL_CLBP_CSV, CLINICAL_MDD_CSV, FIGURES_DIR

import pandas as pd

LLM_METRICS = [
    "Physical_Pain", "poor_QoL", "Emotional_Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency_Deficit", "Narrative_Fragmentation",
]
LLM_DISPLAY_NAMES = [
    "Physical Pain", "QoL#", "Emotional Pain", "Catastrophizing",
    "Depression", "Anxiety", "Rumination", "Agency Deficit", "Narrative Fragmentation",
]

CLBP_CLINICAL_SCORES = [
    "NRS Pain Score", "VAS Pain Score", "MPQ Sensory", "PDQ", "CSI Score",
    "MPQ Affective", "PCS", "Pain Interference", "RMDQ",
    "VAS Depression Score", "MADRS", "HADS (Depression)", "HADS (Anxiety)",
]
MDD_CLINICAL_SCORES = ["CSI Score", "VAS Depression Score", "MADRS", "HADS (Depression)", "HADS (Anxiety)"]


def make_figure():
    llm_scores = load_llm_scores("common")
    llm_scores = llm_scores.rename(columns={"study_id": "Study ID"})

    df_clbp = pd.merge(llm_scores, pd.read_csv(CLINICAL_CLBP_CSV, encoding="utf-8-sig"), on="Study ID")
    df_mdd = pd.merge(llm_scores, pd.read_csv(CLINICAL_MDD_CSV, encoding="utf-8-sig"), on="Study ID")

    combined_df, fdr_results = plot_combined_llm_vs_clinical_correlation(
        df_clbp, df_mdd,
        llm_metrics=LLM_METRICS,
        clinical_scores_cohort1=CLBP_CLINICAL_SCORES,
        clinical_scores_cohort2=MDD_CLINICAL_SCORES,
        llm_names=LLM_DISPLAY_NAMES,
        cohort1_label="CLBP",
        cohort2_label="MDD",
        figsize=(20, 10),
        fdr_alpha=0.05,
        annot_fontsize=11,
        label_fontsize=14,
        fn=FIGURES_DIR / "fig3_llm_clinical_correlations.png",
    )
    print(f"Saved -> {FIGURES_DIR / 'fig3_llm_clinical_correlations.png'}")
    return combined_df, fdr_results


if __name__ == "__main__":
    make_figure()
