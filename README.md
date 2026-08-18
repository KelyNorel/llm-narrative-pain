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
    ├── condition_specific/       # condition-specific section (CLBP and MDD only)
    └── human/                    # 15 manually-transcribed reference interviews (WER validation)
```

Each transcript is named `<Study ID>_<Dx>.txt` (e.g. `1202_CLBP.txt`), `Dx` ∈ {`CLBP`, `MDD`, `HC`}. `raw/` is split into `common/` + `condition_specific/` by an LLM call (`Code/analysis/transcript_splitter.py`), not by an audio-level segmentation. LLM-derived metrics reported in the main analysis are always computed from `common/`, except for the Reddit external-validation comparison (Fig. 5), which uses `condition_specific/`.

`transcripts/human/` holds 5 manually-transcribed reference interviews per cohort, produced by `Code/analysis/prepare_human_transcripts.py` from a source Word document (not redistributed; only its per-subject output is, since that's covered by the same IRB approval). They cover the common section only, for all 15 subjects — see `wer_transcription_accuracy.py`.

`Data/clinical/clbp_clinical.csv` / `mdd_clinical.csv` hold the validated clinical questionnaire scores (NRS, VAS, MPQ, PDQ, CSI, PCS, RMDQ, MADRS, HADS, etc.) per `Study ID`, one file per cohort — also de-identified and included.

`Data/demographics/demographics.csv` holds one row per subject (all three cohorts) with age, sex, race, ethnicity, marital/employment status, income, education, and cognition score (TICS) — also de-identified and included.

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
2. **`llm_scoring.py`** — scores each transcript with two prompts (`PROMPT_1`: Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety; `PROMPT_2`: Catastrophizing, Rumination, Narrative_Fragmentation, Agency_Deficit — the paper's nine metrics), `temperature=0` to reduce output variance (LLMs are not fully deterministic even at `temperature=0`). `score_all_transcripts(section)` (`section` = `"common"` or `"condition_specific"`) caches one JSON per subject and writes `Results/llm_scores/llm_scores_common.csv` / `llm_scores_condition_specific.csv`. `load_llm_scores(section)` reads those CSVs back — use it (not a hardcoded path) in any downstream analysis script.
3. **`word_counts.py`** — counts words/characters per transcript in `Data/transcripts/common/`, saves `Results/word_counts.csv`, and plots word count by cohort (`Figures/word_count_boxplot.png`).
4. **`boxplots.py`** — `plot_metrics_comparison(...)`, the shared plotting function behind the paper's boxplot figures: per-metric Kruskal-Wallis omnibus test, then pairwise Mann-Whitney U (two-sided) brackets only for pairs listed in `pairwise_comparisons` when the omnibus test is significant; pastel cohort colors, hatching for N ≥ 100, jittered points for N < 100. Generic over any metrics/panels, not specific to one figure.
5. **`fig2_llm_ratings_by_cohort.py`** — reproduces Fig. 2 (Physical Pain/QoL, five negative-affect metrics, Agency Deficit/Narrative Fragmentation, by cohort) using `boxplots.py` + `load_llm_scores("common")`, saving `Figures/fig2_llm_ratings_by_cohort.png`.
6. **`correlation_heatmap.py`** — `plot_combined_llm_vs_clinical_correlation(...)`: Spearman correlation between LLM metrics and clinical scores, FDR-corrected (Benjamini-Hochberg) per cohort, combined into one heatmap with a per-cohort median column.
7. **`fig3_llm_clinical_correlations.py`** — reproduces Fig. 3 (LLM metrics vs. clinical scores, CLBP + MDD) using `correlation_heatmap.py` + `Data/clinical/`, saving `Figures/fig3_llm_clinical_correlations.png`.
8. **`glasso_correlations.py`** — fits Graphical Lasso (5-fold CV alpha) on the nine LLM metrics and bootstraps FDR-corrected significance for both the bivariate (Spearman) and GLasso partial correlation networks.
9. **`fig4ab_glasso_correlations_clbp.py`** / **`efig2ab_glasso_correlations_mdd.py`** — reproduce Fig. 4 / eFigure 2 panels A/B using `glasso_correlations.py`, saving `Figures/fig4ab_glasso_correlations_clbp.png` / `efig2ab_glasso_correlations_mdd.png`.
10. **`circular_dendrogram.py`** — Ward-linkage circular (radial) dendrogram of a (thresholded) partial correlation matrix, labels rotated to point outward. Renders black/white; the published figure's colored "Emotional"/"Cognitive" cluster sectors were added manually afterward (cluster boundaries chosen by eye from the leaf angles, per the paper's Methods).
11. **`fig4c_circular_dendrogram_clbp.py`** / **`efig2c_circular_dendrogram_mdd.py`** — reproduce Fig. 4 / eFigure 2 panel C using `circular_dendrogram.py` and the precomputed partial correlation matrices in `Results/glasso/` (not a live GLasso refit — the dendrogram's topology is sensitive to small alpha differences that drift across scikit-learn versions, unlike panels A/B's significance counts, which reproduce exactly either way). Saves `Figures/fig4c_circular_dendrogram_clbp.png` / `efig2c_circular_dendrogram_mdd.png`.
12. **`join_panels.py`** — `join_ab_c(...)`: stacks a panel-A/B image above a panel-C image with "A"/"B"/"C" labels, matching the paper's layout.
13. **`fig4_join_panels_clbp.py`** / **`efig2_join_panels_mdd.py`** — composite the full Fig. 4 / eFigure 2 (panels A+B+C) using `join_panels.py`, saving `Figures/fig4_combined_clbp.png` / `efig2_combined_mdd.png`.
14. **`fig5_reddit_comparison.py`** — reproduces Fig. 5 panels A-C (LLM metrics for the condition-specific section, clinical cohorts vs. matched Reddit communities `r/chronicpain` / `r/depressed`) using `boxplots.py` + `Results/reddit/`. No p-value brackets except `r/chronicpain` vs. `r/depressed`, gated on effect size (`min_effect_size=2`) rather than p-value, since Mann-Whitney p-values are trivially significant at Reddit's N (~4,900 / ~2,300) against clinical N (67/33). Panel D (schematic, built in PowerPoint) is not reproduced. Saves `Figures/fig5_reddit_comparison.png`.
15. **`figs1_data_amount.py`** — reproduces Fig. S1 (speech duration + word count by cohort) using `boxplots.py` + `Results/data_amount.csv`. All three pairwise comparisons are p < 0.001 for both metrics, so brackets are omitted from the plot (`pairwise_comparisons=None`) and stated once in the figure caption instead. Saves `Figures/figs1_data_amount.png`.
16. **`prepare_human_transcripts.py`** — one-time conversion of the 15 manually-transcribed reference interviews (source Word doc, not redistributed) into `Data/transcripts/human/`. Not part of the regular pipeline; its *output* is what's versioned.
17. **`wer_transcription_accuracy.py`** — Word Error Rate of the automatic transcripts (`common/`) against `Data/transcripts/human/`, via `jiwer`. Across the 12 subjects with a usable automatic transcript: median WER 10.4%, mean 19.4% (mean pulled up by a 5-subject cluster at 32-42% vs. the rest at 2-12% — no subject in between; worth a closer look if audio ever becomes available again, since it isn't now). 3 subjects (1221, 1241, 1406) are flagged and excluded from that summary: their automatic `common/` transcript matches the human reference at the start and end but is missing two of the four common-section topics entirely (not misplaced into `condition_specific/` either) — a real defect in the earlier LLM-based transcript-splitting step for these 3 (it dropped a chunk instead of extracting it faithfully, and it wasn't caught by manual review against the source audio at the time), not a transcription (Whisper) error. Their full-reference WER (0.92-0.97) is therefore mostly deletions unrelated to transcription quality. `wer_matched_span` reconstructs a fair reference by splicing the human text to the two segments the automatic transcript actually covers (boundaries manually identified, `MANUALLY_VERIFIED_SPANS`), landing at 0.30-0.44 — much closer to the rest of the cohort, confirming the inflated raw WER was mostly the splitting bug, not worse transcription. The same undetected failure mode could exist in other subjects' `common/`/`condition_specific/` files outside this 15-subject validation set. Saves `Results/wer/wer_by_subject.csv`.
18. **`confound_analysis.py`** — tests whether demographic/clinical-encounter variables (age, sex, race, ethnicity, marital/employment status, income, education, word count, TICS) are associated with the nine LLM metrics, per cohort (Spearman for continuous, Mann-Whitney U for binary, Kruskal-Wallis H for multi-level categorical; FDR within each cohort). A test is skipped, not "not significant", when fewer than two categories have ≥3 participants in that cohort (Race in HC: 28 White/Caucasian vs. 2 Black or African American vs. 1 Asian; Ethnicity in CLBP: 65 non-Hispanic vs. 2 Hispanic or Latino) — `run_cohort_tests` simply produces no row for that (confound, metric) pair rather than a fabricated p-value.
19. **`figs_categorical_confounds_heatmap.py`** — reproduces the categorical-confounds heatmap (Supplement; exact figure number TBD, rename this file once known) using `confound_analysis.py`. Skipped tests are rendered as gray "n/a" cells, not left to fall through to the significance-marker logic: the original version checked `if piv_sig.loc[conf, met]:` directly on a pivoted column that's `NaN` for skipped tests, and `bool(float('nan'))` is `True` in Python — so untested cells were silently drawn with a `*` as if FDR-significant. Saves `Figures/figs_categorical_confounds_heatmap.png`.

Run from `run_llm_scoring.ipynb` / `run_word_counts.ipynb`, or directly:

```bash
cd Code/analysis
python transcript_splitter.py
python llm_scoring.py
python word_counts.py
python fig2_llm_ratings_by_cohort.py
python fig3_llm_clinical_correlations.py
python fig4ab_glasso_correlations_clbp.py
python efig2ab_glasso_correlations_mdd.py
python fig4c_circular_dendrogram_clbp.py
python efig2c_circular_dendrogram_mdd.py
python fig4_join_panels_clbp.py
python efig2_join_panels_mdd.py
python fig5_reddit_comparison.py
python figs1_data_amount.py
python prepare_human_transcripts.py
python wer_transcription_accuracy.py
python figs_categorical_confounds_heatmap.py
```

---

**Author:** Raquel (Kely) Norel, PhD
**Domain:** Computational Psychiatry / NLP / LLM-Based Clinical Assessment
**Status:** 🚧 In progress. Preprocessing, the LLM-scoring pipeline, precomputed LLM scores and clinical/demographic data, Fig. 2, Fig. 3, Fig. 4 / eFigure 2 (all panels), Fig. 5 (panels A-C), Fig. S1, WER validation against 15 manual reference transcripts, and the categorical-confounds heatmap are in place. Remaining: classification analysis and the 100-run output variability check at `temperature=0` (SD/CV/ICC1).
