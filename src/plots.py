from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def save_roc_curve(y_true, y_prob, title: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_feature_importance(features: pd.Index | list, importances, out_path: str | Path, top_n: int = 20) -> None:
    df = pd.DataFrame({"feature": list(features), "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"], df["importance"])
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metric_bar(metrics_df: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(x="model", y="roc_auc", kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison by ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
