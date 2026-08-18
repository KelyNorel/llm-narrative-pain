"""Classification analysis: can pairs of LLM-derived metrics distinguish
the three cohorts (CLBP, HC, MDD)?

For a given pair of metrics, a Random Forest's hyperparameters are
grid-searched (5-fold stratified CV, scoring=f1_macro); the reported
F1 Macro is the mean of that model's f1_macro across the 5 folds (same
number GridSearchCV/cross_validate report). The confusion matrix is
built separately, from out-of-fold predictions collected by refitting
the same (best-hyperparameter) pipeline within each of the 5 folds --
so every subject is scored only by folds that did not train on it, and
each subject appears in the matrix exactly once. Because the grid
search and the confusion-matrix folds are the same StratifiedKFold
instance/seed, this is not a fully nested CV: hyperparameters are
picked using the same folds the confusion matrix reports on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_scoring import load_llm_scores

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
CV_FOLDS = 5
CLASS_ORDER = ["CLBP", "HC", "MDD"]  # alphabetical, matches sorted(Dx.unique())

RF_PARAM_GRID = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}


def load_classification_data() -> pd.DataFrame:
    """LLM scores for all 131 subjects (common section) -- the nine
    metrics are the candidate classification features, Dx is the 3-way
    cohort label."""
    df = load_llm_scores("common")
    return df.rename(columns={"study_id": "Study ID", "dx": "Dx"})


def run_feature_set(df: pd.DataFrame, features: list) -> dict:
    """Grid-search a Random Forest on `features`, then build its
    out-of-fold confusion matrix. Returns best hyperparameters, the
    cross-validated F1 macro (mean across folds), and the confusion
    matrix (rows/cols ordered as CLASS_ORDER)."""
    X = df[features]
    y = df["Dx"]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    pipeline = Pipeline([("classifier", RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE))])
    grid_search = GridSearchCV(
        pipeline, RF_PARAM_GRID, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(X, y)
    best_pipeline = grid_search.best_estimator_

    cv_results = cross_validate(best_pipeline, X, y, cv=cv, scoring="f1_macro")
    f1_macro_mean = cv_results["test_score"].mean()

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_pipeline.fit(X_train, y_train)
        y_pred_all.extend(best_pipeline.predict(X_test))
        y_true_all.extend(y_test)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASS_ORDER)

    return dict(
        features=features,
        best_params=grid_search.best_params_,
        f1_macro=f1_macro_mean,
        confusion_matrix=cm,
    )
