from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CV_FOLDS, N_ITER_SEARCH, RANDOM_STATE, TEST_SIZE


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    from sklearn.pipeline import Pipeline as SkPipeline
    numeric_transformer = SkPipeline(steps=numeric_steps)
    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_models() -> Dict[str, Tuple[object, dict, bool]]:
    # model -> (estimator, param_distributions, scale_numeric)
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "model__C": np.logspace(-2, 2, 20),
                "model__solver": ["lbfgs", "liblinear"],
            },
            True,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 6, 10, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            False,
        ),
        "extra_trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "model__n_estimators": [250, 400, 600],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            False,
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            False,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [None, 4, 8, 12],
                "model__max_iter": [100, 200, 300],
                "model__min_samples_leaf": [10, 20, 30],
            },
            False,
        ),
    }


def build_pipeline(X: pd.DataFrame, model, scale_numeric: bool = False) -> Pipeline:
    preprocessor = make_preprocessor(X, scale_numeric=scale_numeric)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_with_cv(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
    return {k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}


def tune_model(pipe: Pipeline, param_distributions: dict, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=0,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def train_test_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
