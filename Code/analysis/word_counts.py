"""Word/character counts per transcript, compared across cohorts.

Counts are computed on the common interview section, matching how the
paper always derives LLM-based metrics (see README) and reports
narrative length (fig. S1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRANSCRIPTS_DIR, RESULTS_DIR, FIGURES_DIR

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COHORT_ORDER = ["CLBP", "MDD", "HC"]
COHORT_COLORS = {"CLBP": "tab:blue", "MDD": "tab:orange", "HC": "tab:green"}


def count_words_and_chars(transcripts_dir: Path = TRANSCRIPTS_DIR / "common") -> pd.DataFrame:
    """Count words and characters in every transcript in transcripts_dir."""
    rows = []
    for txt_path in sorted(transcripts_dir.glob("*.txt")):
        content = txt_path.read_text(encoding="utf-8")
        study_id, dx = txt_path.stem.split("_")
        rows.append({
            "study_id": study_id,
            "dx": dx,
            "word_count": len(content.split()),
            "char_count": len(content),
        })
    return pd.DataFrame(rows)


def plot_word_count_boxplot(
    df: pd.DataFrame, output_path: Path = FIGURES_DIR / "word_count_boxplot.png"
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=df, x="dx", y="word_count", hue="dx", order=COHORT_ORDER,
        palette=COHORT_COLORS, legend=False, ax=ax,
    )
    sns.stripplot(
        data=df, x="dx", y="word_count", order=COHORT_ORDER,
        color="black", size=4, alpha=0.6, jitter=True, ax=ax,
    )
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Word count")
    ax.set_title("Interview word count by cohort (common section)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def main():
    df = count_words_and_chars()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "word_counts.csv", index=False)
    plot_word_count_boxplot(df)
    print(f"{len(df)} transcripts processed -> {RESULTS_DIR / 'word_counts.csv'}")


if __name__ == "__main__":
    main()
