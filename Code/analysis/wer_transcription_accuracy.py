"""Word Error Rate (WER) of the automatic (Whisper) transcripts against
15 manually-transcribed reference interviews (Data/transcripts/human/).

`substitutions`/`deletions`/`insertions`/`hits` (from
jiwer.process_words) break WER down by error type: substitutions are
genuine misheard words; deletions are words in the human reference
that the automatic transcript is missing (this is what dominates for
the 3 flagged subjects below, quantifying the content-loss story);
insertions are words the automatic transcript produced with no
reference counterpart -- usually genuine ASR errors (hallucinated or
repeated words), but not always: in at least one observed case an
insertion was Whisper correctly transcribing audio the human
transcriber marked "[uncomprehensive word]" (could not make out).
Insertions alone don't distinguish these two cases; that takes reading
the specific transcript.

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
with a literal "..." in the text.

This is a real defect in the earlier LLM-based transcript-splitting
step for these 3 subjects specifically (it dropped a chunk of the
source transcript instead of faithfully extracting it), not a
transcription (Whisper) error, and not something this repo's authors
caught by manually reviewing the split output against the source audio
at the time. It's surfaced here, after the fact, only because these 3
happen to be among the 15 with an independent human reference to catch
it against; the same failure mode could be present, undetected, in
other subjects' common/condition_specific files.

Their full-reference WER is therefore mostly deletions of content the
automatic pipeline never had a chance to get right or wrong, and isn't
comparable to the rest. `wer_matched_span` instead scores each against
a reconstructed reference: the human text up to where the automatic
transcript's first segment ends, spliced with the human text spanning
its second segment -- both boundaries located by anchor phrases (see
MANUALLY_VERIFIED_SPANS) identified by manually reading each pair of
transcripts side by side. This isolates transcription quality on the
content the automatic pipeline actually attempted, separate from the
splitting step's data loss, and comes out much closer to the rest of
the cohort (0.30-0.44) than the naive full-reference WER (0.92-0.97)
suggested.
MANUALLY_VERIFIED_SPANS is specific to these 3 subjects' known failure
pattern (one dropped middle section) and won't generalize to a
differently-broken transcript.

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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRANSCRIPTS_COMMON_DIR, TRANSCRIPTS_HUMAN_DIR, WER_BY_SUBJECT_CSV

import jiwer
import pandas as pd

WORD_COUNT_RATIO_THRESHOLD = 0.5  # flag if automatic/human word-count ratio is outside [threshold, 1/threshold]

# For the 3 flagged subjects: phrases (found verbatim, case/punctuation-insensitive,
# in both the automatic and human transcripts) marking where the automatic
# transcript's two segments start/end within the *human* reference.
MANUALLY_VERIFIED_SPANS = {
    "1221_CLBP": {
        "start_anchor": "what would be the other part",
        "end_start_anchor": "recent event the most recent event",
        "end_tail_anchor": "homeschooled our two daughters",
    },
    "1241_CLBP": {
        "start_anchor": "two adult children no grandchildren yet",
        "end_start_anchor": "so it was it was a it's always a fun time",
        "end_tail_anchor": "getting together with my wife's family",
    },
    "1406_MDD": {
        "start_anchor": "investigations and did executive security protection",
        "end_start_anchor": "the recent event i would probably say on friday",
        "end_tail_anchor": "do that every friday",
    },
}


def _normalize_word(word: str) -> str:
    word = word.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", word)


def _find_word_seq(words_norm: list, anchor_norm: list, start_from: int = 0) -> int:
    n = len(anchor_norm)
    for i in range(start_from, len(words_norm) - n + 1):
        if words_norm[i:i + n] == anchor_norm:
            return i
    return -1


def build_matched_span_reference(human_text: str, spans: dict) -> str:
    """Splice the human reference to match the automatic transcript's two
    (start, end) segments, per MANUALLY_VERIFIED_SPANS anchors."""
    words = human_text.split()
    words_norm = [_normalize_word(w) for w in words]

    start_idx = _find_word_seq(words_norm, spans["start_anchor"].split())
    start_end = start_idx + len(spans["start_anchor"].split())

    end_start = _find_word_seq(words_norm, spans["end_start_anchor"].split(), start_from=start_end)
    tail_words = spans["end_tail_anchor"].split()
    tail_idx = _find_word_seq(words_norm, tail_words, start_from=end_start)
    end_end = tail_idx + len(tail_words)

    return " ".join(words[0:start_end] + words[end_start:end_end])


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
        human_words, automatic_words = len(reference.split()), len(hypothesis.split())
        word_ratio = automatic_words / human_words
        flagged = not (WORD_COUNT_RATIO_THRESHOLD <= word_ratio <= 1 / WORD_COUNT_RATIO_THRESHOLD)

        out = jiwer.process_words(reference, hypothesis)

        wer_matched_span = None
        if flagged and f"{study_id}_{dx}" in MANUALLY_VERIFIED_SPANS:
            matched_ref = build_matched_span_reference(reference, MANUALLY_VERIFIED_SPANS[f"{study_id}_{dx}"])
            wer_matched_span = jiwer.wer(matched_ref, hypothesis)

        rows.append({
            "study_id": study_id,
            "dx": dx,
            "wer": round(out.wer, 3),
            "substitutions": out.substitutions,
            "deletions": out.deletions,
            "insertions": out.insertions,
            "hits": out.hits,
            "wer_matched_span": round(wer_matched_span, 3) if wer_matched_span is not None else None,
            "human_words": human_words,
            "automatic_words": automatic_words,
            "flagged": flagged,
        })

    df = pd.DataFrame(rows)
    WER_BY_SUBJECT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WER_BY_SUBJECT_CSV, index=False)

    clean = df[~df["flagged"]]
    flagged = df[df["flagged"]]
    print(df.to_string(index=False))
    print(f"\nClean subjects (n={len(clean)}): Mean WER: {clean['wer'].mean():.3f}  |  Median WER: {clean['wer'].median():.3f}")
    if len(flagged):
        print(f"\n{len(flagged)} subject(s) flagged (transcript-splitting step dropped content, see module docstring): {flagged['study_id'].tolist()}")
        print(f"Their wer_matched_span (isolates transcription quality from the data loss): {dict(zip(flagged['study_id'], flagged['wer_matched_span'].round(3)))}")
    print(f"\nSaved -> {WER_BY_SUBJECT_CSV}")
    return df


if __name__ == "__main__":
    compute_wer()
