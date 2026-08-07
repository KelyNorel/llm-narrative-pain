# LLM Narrative Pain

Code to reproduce the analyses from *"Naturalistic Narrative Analysis Captures Validated, Novel Psychological Constructs in Chronic Pain"* (Norel, Zhang, Gewandter, Naddour, Abdallah, Duan, Cecchi, Geha).

Chronic low-back pain (CLBP), major depressive disorder (MDD), and pain-free healthy control (HC) participants completed semi-structured interviews. Interviews were transcribed and scored by an LLM (Llama-3-405B-Instruct, accessed through IBM watsonx) on nine clinical/psychological metrics, which were then compared against validated questionnaires.

The repo is organized by pipeline stage, matching the folders under `Code/`:

| Folder | Purpose |
|---|---|
| `Code/preprocessing/` | Audio cleanup + transcription (interview recordings -> text) |
| `Code/analysis/` | LLM-based scoring + statistical analyses reported in the paper |
| `Code/rebuttal/` | Additional analyses added during peer review |

`Data/`, `Results/`, and `Figures/` mirror these stages as the corresponding code is added.

## Setup

```bash
pyenv virtualenv 3.11 clbp_mdd
pyenv activate clbp_mdd
pip install -r requirements.txt
python -m ipykernel install --user --name clbp_mdd --display-name "clbp_mdd"
```

The last line registers the venv as a Jupyter kernel named `clbp_mdd`, so it shows up in the kernel picker (VS Code / JupyterLab) instead of only the default `Python 3 (ipykernel)`.

All stages share one virtualenv, so dependencies live in a single `requirements.txt` at the repo root, extended as each stage is documented here.

Paths are never hardcoded. `Code/config.py` resolves the project root from its own location on disk; every script imports its paths from there. Set the `PAIN_PROJECT_ROOT` environment variable to override it (e.g. to keep `Data/` on an external drive while the code stays in the repo).

## Data

Raw interview recordings and transcripts are **not included** in this repository — they are identifiable patient data (PHI) collected under IRB approval (URMC IRB Study 00007633) and cannot be shared. The folders below are kept as empty placeholders (`.gitkeep`) so the pipeline runs once you supply your own data in the same layout:

```
Data/
├── audio_raw/       # original .m4a interview recordings (not included)
├── audio_cut/       # .wav after m4a conversion + long-silence removal (generated)
└── transcripts/     # .txt transcripts from Whisper (generated)
```

## Preprocessing (`Code/preprocessing/`)

Recordings were captured in Zoom with each speaker on a separate audio channel, so no speaker diarization is needed.

1. **`audio_preprocessing.py`** — converts `.m4a` recordings under `Data/audio_raw/` (searched recursively, so cohort subfolders are optional) to `.wav`, then removes long silences (pydub, silence threshold -50 dB, min length 5 s), writing the result to `Data/audio_cut/`.
2. **`transcribe.py`** — transcribes each `.wav` in `Data/audio_cut/` with OpenAI's Whisper **base** model (run locally, matching the paper), writing `.txt` files to `Data/transcripts/`.

Run both from `run_preprocessing.ipynb`, or directly:

```bash
cd Code/preprocessing
python audio_preprocessing.py
python transcribe.py
```

**System dependency:** [`ffmpeg`](https://ffmpeg.org/) must be installed and on `PATH` (used by both `pydub` and Whisper). On macOS: `brew install ffmpeg`.

Note: this consolidates the original exploratory notebooks (`whisper.ipynb`, `cut_pauses_run.ipynb`) into reusable functions with the same parameters (Whisper `base`, `condition_on_previous_text=False`, `hallucination_silence_threshold=2`; silence threshold -50 dB / 5000 ms). The originals also included one-off cells for specific subjects/cohorts and a manual Whisper weight-caching step, which are dropped here in favor of Whisper's own model cache (`~/.cache/whisper`).

---

**Author:** Raquel (Kely) Norel, PhD
**Domain:** Computational Psychiatry / NLP / LLM-Based Clinical Assessment
**Status:** 🚧 In progress. Preprocessing (audio cleanup + Whisper transcription) is done and reproducible from raw recordings. Remaining: `Code/analysis/` (LLM scoring via IBM watsonx + statistical analyses) and `Code/rebuttal/`.
