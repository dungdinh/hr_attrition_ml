from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import POSITIVE_LABEL, TARGET_COL, LEAKAGE_COLUMNS, ID_COLUMNS


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, sep=None, engine="python")


def normalize_target(y: pd.Series) -> pd.Series:
    """Normalize a binary target column to 0/1 robustly.

    Supports common HR attrition labels such as:
    - Yes / No
    - Y / N
    - True / False
    - 1 / 0 (as strings or numbers)

    If values are not recognized, raise a clear error so the dataset can be inspected.
    """
    y_str = y.astype("string").str.strip().str.lower()

    positive = {"yes", "y", "true", "1", "1.0", "left", "attrited", "attrition"}
    negative = {"no", "n", "false", "0", "0.0", "stayed", "remain", "remained"}

    mapped = y_str.map(lambda v: 1 if v in positive else 0 if v in negative else pd.NA)

    if mapped.isna().any():
        unknown = sorted(set(y_str[mapped.isna()].dropna().unique().tolist()))
        raise ValueError(
            "Target column contains unsupported labels: " + ", ".join(unknown) +
            ". Expected a binary target such as Yes/No or 1/0."
        )

    return mapped.astype(int)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop leakage / identifier columns if present
    drop_cols = [c for c in list(LEAKAGE_COLUMNS | ID_COLUMNS) if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Remove duplicated rows
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def infer_target_column(df: pd.DataFrame) -> str:
    if TARGET_COL in df.columns:
        return TARGET_COL
    candidates = [c for c in df.columns if c.lower() in {"attrition", "leftcompany", "left", "exited", "turnover"}]
    if not candidates:
        raise ValueError("Could not infer target column. Expected a binary attrition target.")
    return candidates[0]


def _json_default(obj):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_joblib(obj: Any, path: str | Path) -> None:
    joblib.dump(obj, path)
