"""Naive word-counting baseline for Fig. S4 panel B: how many times do
pain- and depression-related word stems appear in each subject's common-
section transcript, as a comparison point against the LLM-derived metrics.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRANSCRIPTS_COMMON_DIR

import pandas as pd

DEPRESSION_WORDS = ["depression", "depress", "depressed", "depressing", "depressive", "depresses"]
PAIN_WORDS = [
    "pain", "painful", "paining", "pains",
    "ache", "aches", "aching", "achy",
    "hurt", "hurts", "hurting",
    "sore", "sores", "soreness",
]


def count_words(text: str, word_list: list) -> int:
    """Whole-word, case-insensitive count of any stem in `word_list`
    (matches suffixes too, e.g. "pain" also matches "pains", "painful")."""
    count = 0
    for word in word_list:
        pattern = r"\b" + re.escape(word) + r"\w*\b"
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def count_keywords_all_subjects() -> pd.DataFrame:
    """Pain-word and depression-word counts for every subject's common-
    section transcript."""
    rows = []
    for f in sorted(TRANSCRIPTS_COMMON_DIR.glob("*.txt")):
        study_id, dx = f.stem.split("_")
        text = f.read_text(encoding="utf-8")
        rows.append(dict(
            study_id=int(study_id), dx=dx,
            pain_count=count_words(text, PAIN_WORDS),
            depression_count=count_words(text, DEPRESSION_WORDS),
        ))
    return pd.DataFrame(rows)
