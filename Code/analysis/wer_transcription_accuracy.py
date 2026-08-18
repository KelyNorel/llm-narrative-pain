"""Word Error Rate (WER) of the automatic (Whisper) transcripts against
15 manually-transcribed reference interviews (Data/transcripts/human/).

The automatic transcript compared against is common/ alone, not
common/ + condition_specific/ concatenated: empirically, common/ alone
gives a much lower (sensible) WER for every subject where a fair
comparison is possible, confirming the human reference transcripts
cover the common section only, matching HC (which has no
condition_specific/ section at all).

A WORD_COUNT_RATIO_THRESHOLD flags subjects whose automatic transcript
is implausibly short relative to the human reference (e.g. 75 words
against a 711-word human reference) as `flagged` rather than silently
averaging them into the summary stats. For all 3 flagged subjects
(1221, 1241, 1406), the automatic transcript's start and end match the
human reference closely, but roughly the middle two of four
common-section topics (what the subject enjoys doing; a recent dream)
are missing entirely -- not present in condition_specific/ either, so
the content wasn't misplaced, it's just gone, and each drop is marked
with a literal "..." in the text. This points to the earlier LLM-based
transcript-splitting step dropping a chunk of the source transcript
for these 3 subjects specifically, not a transcription (Whisper) error.

Their full-reference WER is therefore mostly deletions of content the
automatic pipeline never had a chance to get right or wrong, and isn't
comparable to the rest. To still give a rough read on transcription
quality itself (what a reviewer asking "how good is the transcription"
actually wants), `wer_first_n_words` scores each automatic transcript
against just the first `automatic_words` words of the human reference
-- an approximation, not a true aligned match to the covered span
(the automatic transcript is a start+end splice, not a strict prefix),
but close enough to be informative and clearly labeled as such rather
than silently presented as equivalent to the full-reference WER.

The human reference has "[uncomprehensive word]"/"[uncomprehensive
phrase]" markers (audio the transcriber couldn't make out) stripped
during prepare_human_transcripts.py, rather than excluded from scoring
position-by-position. With only 17 such markers across all 15
transcripts, this is a documented limitation, not a live concern: if
the automatic transcript produced a real word at that position (it did,
in at least one observed case, where Whisper transcribed audio the
human transcriber could not), it is counted as an insertion error even
though the automatic transcript was arguably correct there.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRANSCRIPTS_COMMON_DIR, TRANSCRIPTS_HUMAN_DIR, WER_BY_SUBJECT_CSV

import jiwer
import pandas as pd

WORD_COUNT_RATIO_THRESHOLD = 0.5  # flag if automatic/human word-count ratio is outside [threshold, 1/threshold]


def compute_wer() -> pd.DataFrame:
    rows = []
    for human_path in sorted(TRANSCRIPTS_HUMAN_DIR.glob("*.txt")):
        study_id, dx = human_path.stem.split("_")
        common_path = TRANSCRIPTS_COMMON_DIR / f"{study_id}_{dx}.txt"
        if not common_path.exists():
            print(f"skipping {study_id}_{dx}: no automatic transcript in {TRANSCRIPTS_COMMON_DIR}")
            continue

        reference = human_path.read_text(encoding="utf-8")
        hypothesis = common_path.read_text(encoding="utf-8")
        reference_words = reference.split()
        human_words, automatic_words = len(reference_words), len(hypothesis.split())
        word_ratio = automatic_words / human_words

        reference_first_n = " ".join(reference_words[:automatic_words])

        rows.append({
            "study_id": study_id,
            "dx": dx,
            "wer": jiwer.wer(reference, hypothesis),
            "wer_first_n_words": jiwer.wer(reference_first_n, hypothesis),
            "human_words": human_words,
            "automatic_words": automatic_words,
            "flagged": not (WORD_COUNT_RATIO_THRESHOLD <= word_ratio <= 1 / WORD_COUNT_RATIO_THRESHOLD),
        })

    df = pd.DataFrame(rows)
    WER_BY_SUBJECT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WER_BY_SUBJECT_CSV, index=False)

    clean = df[~df["flagged"]]
    flagged = df[df["flagged"]]
    print(df.to_string(index=False))
    print(f"\nClean subjects (n={len(clean)}): Mean WER: {clean['wer'].mean():.3f}  |  Median WER: {clean['wer'].median():.3f}")
    if len(flagged):
        print(f"\n{len(flagged)} subject(s) flagged (automatic transcript missing content, see module docstring): {flagged['study_id'].tolist()}")
        print(f"Their wer_first_n_words (approximate, see docstring): {dict(zip(flagged['study_id'], flagged['wer_first_n_words'].round(3)))}")
    print(f"\nSaved -> {WER_BY_SUBJECT_CSV}")
    return df


if __name__ == "__main__":
    compute_wer()
