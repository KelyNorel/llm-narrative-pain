"""Split whole-interview transcripts into common vs condition-specific
sections via an LLM call (same model/credentials as llm_scoring.py).

Input: Data/transcripts/raw/ (whole interview, from Whisper).
Output: Data/transcripts/common/ (all participants) and
Data/transcripts/condition_specific/ (CLBP and MDD only; HC has no
condition-specific section, so part2_content is expected to be empty
for HC transcripts).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import call_llm, parse_llm_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRANSCRIPTS_RAW_DIR, TRANSCRIPTS_COMMON_DIR, TRANSCRIPTS_CONDITION_DIR

SPLIT_PROMPT = """You are tasked with splitting an interview transcript into two parts based on the interview structure.

**INTERVIEW STRUCTURE:**
- **Part 1 (Generic questions)**: Lines 1 to S - Non-pain/depression specific questions like:
  - "Tell me a little bit about yourself"
  - "Tell me about things you enjoy doing"
  - "Tell me about a recent dream that you can describe vividly"
  - "Please describe a recent event you had with family and/or friends"

- **Part 2 (Condition-specific questions)**: Lines S+1 to N - Pain/depression related questions like:
  - "What is your pain story?" / "What is your depression story?"
  - "What do you think is causing your back-pain?" / "What do you think is causing your depression?"
  - "What are things that affect your pain/depression?"
  - "How does your pain/depression affect your life?"
  - "How does pain/depression affect your concentration?"
  - "How does pain/depression affect your mood?"
  - "What treatment(s) have you tried for your pain/depression?"
  - Questions about comparison with other people with similar conditions
  - Questions about alternative therapies
  - Questions about relationships and how condition affects them

**YOUR TASK:**
1. Identify where the interview transitions from generic questions to condition-specific questions
2. Split the transcript at that transition point
3. Provide both parts separately

**IMPORTANT NOTES:**
- The interviewer uses the question list as a guide but may adapt based on participant responses
- Some participants may mention their condition early (e.g., when asked "tell me about your life"), but still split based on the question structure, not the content
- Look for the structural transition in the interview flow, not just content mentions

**Response Format:**
```json
{
  "part1_content": "[Full text of Part 1 - generic questions]",
  "part2_content": "[Full text of Part 2 - condition-specific questions]",
  "split_point_line": [line number],
  "split_reasoning": "[brief explanation]"
}
```

Analyze the interview structure carefully and provide the split."""


def split_transcript(transcript_path: Path) -> dict:
    transcript_text = transcript_path.read_text(encoding="utf-8")
    raw = call_llm(SPLIT_PROMPT, transcript_text)
    parsed = parse_llm_json(raw)
    if not parsed or "part1_content" not in parsed or "part2_content" not in parsed:
        raise ValueError(f"Could not parse split for {transcript_path.name}: {raw!r}")
    return parsed


def split_all_transcripts(
    raw_dir: Path = TRANSCRIPTS_RAW_DIR,
    common_dir: Path = TRANSCRIPTS_COMMON_DIR,
    condition_dir: Path = TRANSCRIPTS_CONDITION_DIR,
) -> None:
    common_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.txt"))
    for i, txt_path in enumerate(raw_files):
        common_path = common_dir / txt_path.name
        if common_path.exists():
            print(f"{i} skipping {txt_path.name} (already split)")
            continue

        print(f"{i} splitting {txt_path.name}...")
        try:
            parts = split_transcript(txt_path)
        except Exception as e:
            print(f"  failed: {e}")
            continue

        common_path.write_text(parts["part1_content"], encoding="utf-8")
        if parts["part2_content"].strip():  # HC participants have no condition-specific section
            (condition_dir / txt_path.name).write_text(parts["part2_content"], encoding="utf-8")

    print(f"Done. Split {len(raw_files)} transcripts -> {common_dir}, {condition_dir}")


if __name__ == "__main__":
    split_all_transcripts()
