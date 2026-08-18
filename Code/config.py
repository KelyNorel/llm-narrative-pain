"""Shared project paths and credentials.

Every other script/notebook in this repo imports its paths from here
instead of hardcoding them, so the codebase can run on any machine.
The project root is resolved from this file's own location on disk;
set the PAIN_PROJECT_ROOT environment variable to override it (e.g.
to keep Data/ on an external drive while the code stays in the repo).

Secrets (e.g. WATSONX_APIKEY) are never hardcoded: they are loaded
from a git-ignored `.env` file at the project root. Copy `.env.example`
to `.env` and fill in your own values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(os.environ.get("PAIN_PROJECT_ROOT", _DEFAULT_ROOT)).resolve()

load_dotenv(PROJECT_ROOT / ".env")

CODE_DIR = PROJECT_ROOT / "Code"
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
FIGURES_DIR = PROJECT_ROOT / "Figures"

# Preprocessing pipeline stages:
# audio_raw -> audio_cut -> transcripts/raw -> transcripts/{common,condition_specific}
AUDIO_RAW_DIR = DATA_DIR / "audio_raw"
AUDIO_CUT_DIR = DATA_DIR / "audio_cut"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSCRIPTS_RAW_DIR = TRANSCRIPTS_DIR / "raw"
TRANSCRIPTS_COMMON_DIR = TRANSCRIPTS_DIR / "common"
TRANSCRIPTS_CONDITION_DIR = TRANSCRIPTS_DIR / "condition_specific"
TRANSCRIPTS_HUMAN_DIR = TRANSCRIPTS_DIR / "human"

# LLM scoring outputs (9 metrics per subject, one CSV per interview section)
LLM_SCORES_DIR = RESULTS_DIR / "llm_scores"
LLM_SCORES_COMMON_CSV = LLM_SCORES_DIR / "llm_scores_common.csv"
LLM_SCORES_CONDITION_CSV = LLM_SCORES_DIR / "llm_scores_condition_specific.csv"

# Clinical questionnaire scores (validated instruments), one CSV per cohort
CLINICAL_DIR = DATA_DIR / "clinical"
CLINICAL_CLBP_CSV = CLINICAL_DIR / "clbp_clinical.csv"
CLINICAL_MDD_CSV = CLINICAL_DIR / "mdd_clinical.csv"

# GLasso partial correlation matrices (point estimates behind Fig. 4 / eFigure 2
# panel C's dendrogram), one CSV per cohort
GLASSO_DIR = RESULTS_DIR / "glasso"
PARTIAL_CORR_CLBP_CSV = GLASSO_DIR / "partial_corr_clbp.csv"
PARTIAL_CORR_MDD_CSV = GLASSO_DIR / "partial_corr_mdd.csv"

# Per-subject speech duration + word count (Fig. S1)
DATA_AMOUNT_CSV = RESULTS_DIR / "data_amount.csv"

# WER: automatic transcripts (common + condition_specific) vs. 15 manual
# reference transcripts in TRANSCRIPTS_HUMAN_DIR
WER_DIR = RESULTS_DIR / "wer"
WER_BY_SUBJECT_CSV = WER_DIR / "wer_by_subject.csv"

# Participant demographics (age, sex, race, ethnicity, income, etc.), all
# cohorts in one file, for confound analyses
DEMOGRAPHICS_CSV = DATA_DIR / "demographics" / "demographics.csv"
