"""One-time conversion: parse the 100 repeated temperature=0 scoring runs
(one raw JSON per run per subject) into a single long CSV.

Not part of the regular reproducible pipeline: the source JSONs live
outside the repo and are not redistributed (they predate the paper's
final 9-metric prompt and full 131-subject cohort -- see the module
docstring in temperature_variability.py for what they actually cover).
Rerun this only if the source folder changes, with SOURCE_DIR updated
for your machine.

Filename pattern: "{run}_{dx}_{study_id}.json", run in 0-99. A run that
failed to score (LLM connection error, not a parsing bug) is saved as
"{run}_{dx}_{study_id}.json.raw" instead of valid JSON; those are kept
as rows with all metrics NaN and parse_failed=True, not silently
dropped, so anyone computing variability stats can see how much data
was lost to failed calls (84 of 11500 runs here) rather than assuming
every subject has a full 100 runs.
"""
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TEMP_VARIABILITY_SCORES_CSV

import pandas as pd

SOURCE_DIR = Path("/Users/rnorel/Documents/Neuro/Pain/LLaMA_Results/LL6scores_100runs")

FILENAME_RE = re.compile(r"^(\d+)_([A-Za-z]+)_(\d+)\.json(\.raw)?$")
METRIC_COLUMNS = ["LL_Pain", "LL_PPain", "LL_EPain", "LL_Dep", "LL_QoL", "LL_Anx"]


def parse_source_dir(source_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(source_dir.iterdir()):
        m = FILENAME_RE.match(f.name)
        if not m:
            continue
        run, dx, study_id, is_raw = m.groups()
        row = dict(run=int(run), dx=dx, study_id=int(study_id), parse_failed=bool(is_raw))
        if is_raw:
            row.update({c: None for c in METRIC_COLUMNS})
        else:
            scores = json.loads(f.read_text())
            row.update({c: scores.get(c) for c in METRIC_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dx", "study_id", "run"]).reset_index(drop=True)


def main():
    df = parse_source_dir(SOURCE_DIR)
    TEMP_VARIABILITY_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEMP_VARIABILITY_SCORES_CSV, index=False)
    print(f"Saved -> {TEMP_VARIABILITY_SCORES_CSV}")
    print(f"{df['study_id'].nunique()} subjects, {len(df)} rows, "
          f"{df['parse_failed'].sum()} failed runs (LLM connection error)")


if __name__ == "__main__":
    main()
