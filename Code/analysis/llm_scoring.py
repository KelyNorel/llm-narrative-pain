"""LLM-based scoring of interview transcripts (Llama-3-405B-Instruct via IBM watsonx).

Each transcript is scored with two prompts that together produce the
paper's nine metrics: Prompt 1 (Physical_Pain, Emotional_Pain,
Depression, poor_QoL, Anxiety) and Prompt 2 (Catastrophizing,
Rumination, Narrative_Fragmentation, Agency_Deficit).

Credentials are never hardcoded: set WATSONX_APIKEY, WATSONX_URL, and
WATSONX_PROJECT_ID in a `.env` file at the project root (copy
`.env.example`, see README's Analysis section). Requires access to an
IBM watsonx project with the Llama-3-405B-Instruct model deployed.
"""
import sys
import os
import json
import re
import time
import random
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    TRANSCRIPTS_COMMON_DIR,
    TRANSCRIPTS_CONDITION_DIR,
    LLM_SCORES_COMMON_CSV,
    LLM_SCORES_CONDITION_CSV,
)

import litellm

MODEL_ID = "meta-llama/llama-3-405b-instruct"
LITELLM_MODEL_ID = f"watsonx_text/{MODEL_ID}"

# temperature=0: deterministic scoring, matching the paper. See
# Code/rebuttal/ for the empirical characterization of residual
# variability that remains even at temperature=0.
PARAMETERS = {
    "max_tokens": 4095,
    "temperature": 0,
}

SYSTEM_PROMPT = """You are an expert assistant that always responds in valid JSON format when requested.
Do not include any explanations, markdown formatting, or text outside of the JSON structure when JSON output is requested.
When performing evaluations or assessments, provide your analysis in the exact format requested. Be concise and precise with your responses."""

PROMPT_1 = """You are a knowledgeable psychiatrist tasked with evaluating the emotional state of a patient based on an interview transcript. Carefully analyze the transcript and provide the following scores as decimal values (to one decimal place) on a scale from 0.0 to 10.0, formatted exactly as specified:
1. Physical_Pain: Rate the level of physical pain separately on a scale from 0.0 to 10.0, where 0.0 indicates no physical pain and 10.0 indicates the most severe level of physical pain.
2.- Emotional_Pain: Rate the level of emotional pain separately on a scale from 0.0 to 10.0, where 0.0 indicates no emotional pain and 10.0 indicates the most severe emotional pain possible.
3.Depression: Rate the patient's level of depression on a scale from 0.0 to 10.0, where 0.0 indicates no depression and 10.0 indicates the most severe depression possible.
4. QoL: Rate the patient's quality of life on a scale from 0.0 to 10.0, where 0.0 indicates the best quality of life and 10.0 indicates the poorest quality of life.
5. Anxiety: Rate the patient's anxiety on a scale from 0.0 to 10.0, where 0.0 indicates no anxiety and 10.0 indicates the most severe anxiety possible.
Response Format: { "Physical_Pain": [decimal value], "Emotional_Pain": [decimal value], "Depression": [decimal value], "poor_QoL": [decimal value], "Anxiety": [decimal value] }
Example response: { "Physical_Pain": 6.2, "Emotional_Pain": 9.1, "Depression": 3.7, "poor_QoL": 5.8, "Anxiety": 1.2 }
Provide only a valid JSON response with decimal values to one decimal place. The JSON should follow this exact structure. Think deeply."""

PROMPT_2 = """You are a knowledgeable psychiatrist tasked with evaluating the emotional state of a patient based on an interview transcript. Carefully analyze the transcript and provide the following scores as decimal values (to one decimal place) on a scale from 0.0 to 10.0, formatted exactly as specified:
6. Catastrophizing: Rate the participant's catastrophic thinking patterns on a scale from 0.0 to 10.0, where 0.0 indicates no catastrophizing (realistic assessment of potential outcomes, balanced perspective on challenges, appropriate evaluation of risks) and 10.0 indicates maximum catastrophizing (consistently predicting worst-case scenarios, dramatic overestimation of negative possibilities, persistent expectation of disaster across multiple situations, inability to consider moderate or positive outcomes).
7. Rumination: Rate the participant's level of rumination on a scale from 0.0 to 10.0, where 0.0 indicates no rumination (fluid, forward-moving thought patterns, varied topics, solution-focused thinking) and 10.0 indicates maximum rumination (persistent repetitive thinking about negative experiences, circular thought patterns, inability to move past specific concerns, excessive focus on problems without progress toward resolution, repetitive analysis of past events or worries).
8. Narrative_Fragmentation: Rate the participant's overall narrative coherence on a scale from 0.0 to 10.0, where 0.0 indicates highly coherent narrative structure (clear chronological flow, appropriate time anchoring, consistent timeline, logical cause-and-effect reasoning, appropriate causal connections) and 10.0 indicates severely fragmented narrative (no clear sequence, major temporal gaps, contradictory timeline, no logical causal connections, magical thinking).
9. Agency_Deficit: Rate the participant's sense of personal agency and control on a scale from 0.0 to 10.0, where 0.0 indicates strong sense of personal agency (balanced internal-external attribution, recognizes personal responsibility and ability to influence outcomes, appropriate internal locus of control) and 10.0 indicates complete powerlessness (exclusively external attribution, victim stance, everything controlled by fate/others/circumstances, no sense of personal influence).
Provide only a valid JSON response with decimal values to one decimal place. The JSON should follow this exact structure. Response Format:
{"Catastrophizing": [decimal value], "Rumination": [decimal value], "Narrative_Fragmentation": [decimal value], "Agency_Deficit": [decimal value]} {"Catastrophizing": 6.3, "Rumination": 3.8, "Narrative_Fragmentation": 5.3, "Agency_Deficit": 2.2} Provide only a valid JSON response with decimal values to one decimal place. The JSON should follow this exact structure. Think deeply."""

PROMPTS = (PROMPT_1, PROMPT_2)


def _require_watsonx_credentials() -> None:
    required = ("WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Set them before running (see README's Analysis section)."
        )


def call_llm(prompt: str, transcript_text: str, run_id: int | None = None, max_retries: int = 5) -> str:
    """Send prompt+transcript to the LLM, retrying with exponential backoff on rate limits."""
    _require_watsonx_credentials()
    params = PARAMETERS.copy()
    if run_id is not None:
        params["seed"] = run_id  # varies GPU-batch non-determinism across repeated runs

    question = f"{prompt} Response: {transcript_text}"
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=LITELLM_MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                **params,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_message = str(e)
            if "429" in error_message and attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            elif attempt == max_retries - 1:
                raise
    raise RuntimeError("Failed after multiple retries due to rate limiting")


def parse_llm_json(llm_output: str) -> dict | None:
    """Extract a JSON object from an LLM response, tolerating markdown fences or extra text."""
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_output):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    for candidate in re.findall(r"({[\s\S]*?})", llm_output):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def score_transcript(transcript_path: Path, run_id: int | None = None) -> dict:
    """Score one transcript with both prompts, returning the merged 9-metric dict."""
    transcript_text = transcript_path.read_text(encoding="utf-8")
    scores = {}
    for prompt in PROMPTS:
        raw = call_llm(prompt, transcript_text, run_id=run_id)
        parsed = parse_llm_json(raw)
        if parsed is None:
            raise ValueError(f"Could not parse LLM JSON output for {transcript_path.name}: {raw!r}")
        scores.update(parsed)
    return scores


# Section name -> (transcripts folder, output CSV). Filenames match the
# precomputed results already in the repo (Results/llm_scores/), so a
# fresh run overwrites them in place with the same names.
SECTIONS = {
    "common": (TRANSCRIPTS_COMMON_DIR, LLM_SCORES_COMMON_CSV),
    "condition_specific": (TRANSCRIPTS_CONDITION_DIR, LLM_SCORES_CONDITION_CSV),
}


def score_all_transcripts(section: str = "common") -> "pd.DataFrame":
    """Score every transcript in the given interview section, caching one JSON file per subject."""
    import pandas as pd

    transcripts_dir, csv_path = SECTIONS[section]
    output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for txt_path in sorted(transcripts_dir.glob("*.txt")):
        study_id, dx = txt_path.stem.split("_")
        json_path = output_dir / f"{txt_path.stem}.json"

        if json_path.exists():
            scores = json.loads(json_path.read_text())
        else:
            print(f"Scoring {txt_path.name}...")
            try:
                scores = score_transcript(txt_path)
            except Exception:
                print(f"  failed on {txt_path.name}")
                traceback.print_exc()
                continue
            json_path.write_text(json.dumps(scores, indent=2))

        rows.append({"study_id": study_id, "dx": dx, **scores})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"{len(df)} transcripts scored -> {csv_path}")
    return df


def load_llm_scores(section: str = "common") -> "pd.DataFrame":
    """Load precomputed LLM scores for the given interview section.

    Use this in downstream analysis/stats scripts instead of hardcoding
    a path to Results/llm_scores/llm_scores_common.csv or
    llm_scores_condition_specific.csv.
    """
    import pandas as pd

    _, csv_path = SECTIONS[section]
    df = pd.read_csv(csv_path)
    return df.rename(columns={"Study ID": "study_id", "Dx": "dx"})


if __name__ == "__main__":
    score_all_transcripts("common")
    score_all_transcripts("condition_specific")
