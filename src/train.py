from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .config import ARTIFACT_DIR, DATA_PATH, PLOT_DIR, RANDOM_STATE
from .explainability import make_shap_summary_plot
from .modeling import build_pipeline, evaluate_with_cv, get_models, train_test_data, tune_model
from .plots import save_feature_importance, save_metric_bar, save_roc_curve
from .utils import clean_dataframe, infer_target_column, load_raw_data, normalize_target, save_json, save_joblib


def _safe_positive_proba(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return np.max(proba, axis=1)


def _fit_and_score(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    test_proba = _safe_positive_proba(pipe, X_test)
    test_pred = (test_proba >= 0.5).astype(int)
    return {
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_pred": test_pred,
        "test_proba": test_proba,
    }


def run_training(data_path: str | Path = DATA_PATH) -> Dict[str, Any]:
    raw = load_raw_data(data_path)
    raw = clean_dataframe(raw)
    target_col = infer_target_column(raw)

    y = normalize_target(raw[target_col])
    X = raw.drop(columns=[target_col])

    leakage = [c for c in ["Attrition Reason", "AttritionDate", "IsVoluntary", "Employee Name", "Manager Name", "Join Date", "Employee ID", "EmployeeNumber"] if c in X.columns]
    X = X.drop(columns=leakage, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_data(X, y)

    results: List[Dict[str, Any]] = []
    fitted_defaults: Dict[str, Any] = {}

    # 1) Benchmark 5 algorithms with cross validation using default parameters
    for name, (estimator, _, scale_numeric) in get_models().items():
        pipe = build_pipeline(X_train, estimator, scale_numeric=scale_numeric)
        cv_scores = evaluate_with_cv(pipe, X_train, y_train)
        default_score = _fit_and_score(pipe, X_train, y_train, X_test, y_test)

        row = {
            "model": name,
            "cv_roc_auc": cv_scores["test_roc_auc"],
            "cv_accuracy": cv_scores["test_accuracy"],
            "cv_precision": cv_scores["test_precision"],
            "cv_recall": cv_scores["test_recall"],
            "cv_f1": cv_scores["test_f1"],
            "test_roc_auc_default": default_score["test_roc_auc"],
        }
        results.append(row)
        fitted_defaults[name] = pipe

    results_df = pd.DataFrame(results).sort_values(["cv_roc_auc", "test_roc_auc_default"], ascending=False).reset_index(drop=True)

    # 2) Tune only the top 2 candidates to keep the workflow production-practical
    top_candidates = results_df["model"].head(1).tolist()
    tuned_results: List[Dict[str, Any]] = []
    best_name = None
    best_search = None
    best_score = -1.0

    model_map = {name: spec for name, spec in get_models().items()}
    for name in top_candidates:
        estimator, params, scale_numeric = model_map[name]
        pipe = build_pipeline(X_train, estimator, scale_numeric=scale_numeric)
        search = tune_model(pipe, params, X_train, y_train)
        tuned_pipe = search.best_estimator_
        tuned_test_proba = _safe_positive_proba(tuned_pipe, X_test)
        tuned_test_pred = (tuned_test_proba >= 0.5).astype(int)
        tuned_roc = float(roc_auc_score(y_test, tuned_test_proba))
        row = {
            "model": name,
            "best_params": search.best_params_,
            "tuned_cv_roc_auc": float(search.best_score_),
            "tuned_test_roc_auc": tuned_roc,
        }
        tuned_results.append(row)
        if search.best_score_ > best_score:
            best_score = float(search.best_score_)
            best_name = name
            best_search = search

    tuned_df = pd.DataFrame(tuned_results).sort_values(["tuned_cv_roc_auc", "tuned_test_roc_auc"], ascending=False)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    best_model = best_search.best_estimator_
    test_proba = _safe_positive_proba(best_model, X_test)
    test_pred = (test_proba >= 0.5).astype(int)

    metrics = {
        "best_model": best_name,
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "classification_report": classification_report(y_test, test_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "selected_from_top_candidates": top_candidates,
    }

    save_json(results_df.to_dict(orient="records"), ARTIFACT_DIR / "model_benchmark.json")
    save_json(tuned_df.to_dict(orient="records"), ARTIFACT_DIR / "tuned_results.json")
    save_json(metrics, ARTIFACT_DIR / "metrics.json")
    save_joblib({
        "model": best_model,
        "target_col": target_col,
        "feature_columns": X.columns.tolist(),
        "threshold": 0.5,
        "positive_class": 1,
        "benchmark": results_df.to_dict(orient="records"),
        "tuned": tuned_df.to_dict(orient="records"),
    }, ARTIFACT_DIR / "best_model.pkl")

    save_metric_bar(results_df[["model", "cv_roc_auc"]].rename(columns={"cv_roc_auc": "roc_auc"}), PLOT_DIR / "model_comparison.png")
    save_roc_curve(y_test, test_proba, f"ROC Curve - {best_name}", PLOT_DIR / "roc_curve.png")

    perm = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1, scoring="roc_auc")
    save_feature_importance(X_test.columns.tolist(), perm.importances_mean, PLOT_DIR / "feature_importance.png")

    sample = X_train.sample(min(150, len(X_train)), random_state=RANDOM_STATE)
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    make_shap_summary_plot(best_model, sample, feature_names, PLOT_DIR / "shap_summary.png")

    return {
        "results": results_df,
        "tuned_results": tuned_df,
        "metrics": metrics,
        "best_model_path": str(ARTIFACT_DIR / "best_model.pkl"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    args = parser.parse_args()
    out = run_training(args.data)
    print("Benchmark results:\n", out["results"].to_string(index=False))
    print("\nTuned results:\n", out["tuned_results"].to_string(index=False))
    print("\nSaved to:", out["best_model_path"])
