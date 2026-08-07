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
