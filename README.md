# LLM Narrative Pain

Code to reproduce the analyses from *"Naturalistic Narrative Analysis Captures Validated, Novel Psychological Constructs in Chronic Pain"* (Norel, Zhang, Gewandter, Naddour, Abdallah, Duan, Cecchi, Geha).

Chronic low-back pain (CLBP), major depressive disorder (MDD), and pain-free healthy control (HC) participants completed semi-structured interviews. Interviews were transcribed and scored by an LLM (Llama-3-405B-Instruct, accessed through IBM watsonx) on nine clinical/psychological metrics, which were then compared against validated questionnaires.

The repo is organized by pipeline stage, matching the folders under `Code/`:

| Folder | Purpose |
|---|---|
| `Code/preprocessing/` | Audio cleanup + transcription (interview recordings -> text) |
| `Code/analysis/` | LLM-based scoring + all statistical analyses reported in the paper |

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

Raw interview **recordings** are **not included** in this repository — they are identifiable patient data (PHI) collected under IRB approval (URMC IRB Study 00007633) and cannot be shared. `audio_raw/` and `audio_cut/` are kept as empty placeholders (`.gitkeep`) so the preprocessing pipeline runs once you supply your own recordings in the same layout.

Interview **transcripts** are de-identified and are included, since sharing them is covered by the study's IRB approval.

```
Data/
├── audio_raw/                    # original .m4a interview recordings (not included)
├── audio_cut/                    # .wav after m4a conversion + long-silence removal (not included)
└── transcripts/
    ├── raw/                      # whole-interview transcript from Whisper (not included)
    ├── common/                   # common section (all participants: CLBP, MDD, HC)
    └── condition_specific/       # condition-specific section (CLBP and MDD only)
```

Each transcript is named `<Study ID>_<Dx>.txt` (e.g. `1202_CLBP.txt`), `Dx` ∈ {`CLBP`, `MDD`, `HC`}. `raw/` is split into `common/` + `condition_specific/` by an LLM call (`Code/analysis/transcript_splitter.py`), not by an audio-level segmentation. LLM-derived metrics reported in the main analysis are always computed from `common/`, except for the Reddit external-validation comparison (Fig. 5), which uses `condition_specific/`.

## Preprocessing (`Code/preprocessing/`)

Recordings were captured in Zoom with each speaker on a separate audio channel, so no speaker diarization is needed.

1. **`audio_preprocessing.py`** — converts `.m4a` recordings under `Data/audio_raw/` (searched recursively, so cohort subfolders are optional) to `.wav`, then removes long silences (pydub, silence threshold -50 dB, min length 5 s), writing the result to `Data/audio_cut/`.
2. **`transcribe.py`** — transcribes each `.wav` in `Data/audio_cut/` with OpenAI's Whisper **base** model (run locally, matching the paper), writing whole-interview `.txt` files to `Data/transcripts/raw/`. Splitting into `common/`/`condition_specific/` happens later, in `Code/analysis/transcript_splitter.py` (LLM-based, not audio-based).

Run both from `run_preprocessing.ipynb`, or directly:

```bash
cd Code/preprocessing
python audio_preprocessing.py
python transcribe.py
```

**System dependency:** [`ffmpeg`](https://ffmpeg.org/) must be installed and on `PATH` (used by both `pydub` and Whisper). On macOS: `brew install ffmpeg`.

## Analysis (`Code/analysis/`)

### Credentials

`llm_scoring.py` and `transcript_splitter.py` call Llama-3-405B-Instruct through IBM watsonx (via [litellm](https://github.com/BerriAI/litellm)). Credentials are never hardcoded — copy `.env.example` to `.env` at the repo root and fill in your own `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID`. `.env` is git-ignored. `Code/config.py` loads it automatically for every script.

These two scripts require a live watsonx project with that model deployed; they cannot be run/tested without one.

### Scripts

1. **`transcript_splitter.py`** — splits each whole-interview transcript in `Data/transcripts/raw/` into `common/` + `condition_specific/`, via an LLM prompt that identifies the structural transition between the generic and condition-specific parts of the interview (HC transcripts have no condition-specific section).
2. **`llm_scoring.py`** — scores each transcript with two prompts (`PROMPT_1`: Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety; `PROMPT_2`: Catastrophizing, Rumination, Narrative_Fragmentation, Agency_Deficit — the paper's nine metrics), `temperature=0` for determinism. `score_all_transcripts(section)` (`section` = `"common"` or `"condition_specific"`) caches one JSON per subject and writes `Results/llm_scores/llm_scores_common.csv` / `llm_scores_condition_specific.csv`. `load_llm_scores(section)` reads those CSVs back — use it (not a hardcoded path) in any downstream analysis script.
3. **`word_counts.py`** — counts words/characters per transcript in `Data/transcripts/common/`, saves `Results/word_counts.csv`, and plots word count by cohort (`Figures/word_count_boxplot.png`).
4. **`boxplots.py`** — `plot_metrics_comparison(...)`, the shared plotting function behind the paper's boxplot figures: per-metric Kruskal-Wallis omnibus test, then pairwise Mann-Whitney U (two-sided) brackets only for pairs listed in `pairwise_comparisons` when the omnibus test is significant; pastel cohort colors, hatching for N ≥ 100, jittered points for N < 100. Generic over any metrics/panels, not specific to one figure.
5. **`fig2_llm_ratings_by_cohort.py`** — reproduces Fig. 2 (Physical Pain/QoL, five negative-affect metrics, Agency Deficit/Narrative Fragmentation, by cohort) using `boxplots.py` + `load_llm_scores("common")`, saving `Figures/fig2_llm_ratings_by_cohort.png`.

Run from `run_llm_scoring.ipynb` / `run_word_counts.ipynb`, or directly:

```bash
cd Code/analysis
python transcript_splitter.py
python llm_scoring.py
python word_counts.py
python fig2_llm_ratings_by_cohort.py
```

---

**Author:** Raquel (Kely) Norel, PhD
**Domain:** Computational Psychiatry / NLP / LLM-Based Clinical Assessment
**Status:** 🚧 In progress. Preprocessing, the LLM-scoring pipeline (transcript splitting + 9-metric scoring + word counts), precomputed LLM scores for all 131 subjects, and Fig. 2 (LLM ratings by cohort) are in place. Remaining: statistical analyses (Kruskal-Wallis, Spearman/FDR, Graphical Lasso, classification), Reddit external validation, and the 100-run determinism check (SD/CV/ICC1).
