"""2D projection (labeled "MDS" in the paper) of the nine LLM metrics for
Fig. S4 panel A. Implemented as StandardScaler + TruncatedSVD on the
z-scored metrics: for Euclidean distances, classical MDS is equivalent to
PCA/SVD on the centered data, so this is the same projection under a
different name/implementation, not an approximation of it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

LLM_METRICS = [
    "Narrative_Fragmentation", "Agency_Deficit", "Rumination", "Catastrophizing",
    "Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL", "Anxiety",
]
RANDOM_STATE = 42


def compute_mds_projection() -> pd.DataFrame:
    df = load_llm_scores("common")
    scaled = StandardScaler().fit_transform(df[LLM_METRICS])
    n_components = min(10, min(scaled.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    transformed = svd.fit_transform(scaled)

    result = pd.DataFrame(transformed[:, :2], columns=["MDS1", "MDS2"], index=df.index)
    result["study_id"] = df["study_id"]
    result["dx"] = df["dx"]
    return result
