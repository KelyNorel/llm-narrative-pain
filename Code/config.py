"""Shared project paths.

Every other script/notebook in this repo imports its paths from here
instead of hardcoding them, so the codebase can run on any machine.
The project root is resolved from this file's own location on disk;
set the PAIN_PROJECT_ROOT environment variable to override it (e.g.
to keep Data/ on an external drive while the code stays in the repo).
"""
import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(os.environ.get("PAIN_PROJECT_ROOT", _DEFAULT_ROOT)).resolve()

CODE_DIR = PROJECT_ROOT / "Code"
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
FIGURES_DIR = PROJECT_ROOT / "Figures"

# Preprocessing pipeline stages: audio_raw -> audio_cut -> transcripts
AUDIO_RAW_DIR = DATA_DIR / "audio_raw"
AUDIO_CUT_DIR = DATA_DIR / "audio_cut"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
