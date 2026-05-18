from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_shap_summary_plot(pipeline, X_sample: pd.DataFrame, feature_names, out_path: str | Path) -> None:
    """Create a SHAP summary plot when enabled.

    By default this function writes a lightweight placeholder so the training
    pipeline stays fast and stable. Set ENABLE_SHAP=1 to compute a real SHAP plot.
    """
    if os.getenv("ENABLE_SHAP", "0") != "1":
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.axis("off")
        ax.text(0.01, 0.62, "SHAP is disabled by default for fast training.", fontsize=12)
        ax.text(0.01, 0.28, "Set ENABLE_SHAP=1 to generate a full SHAP summary plot.", fontsize=10)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    # SHAP can be sensitive to environment versions. We try to patch a known
    # compatibility issue with newer coverage releases before importing shap.
    try:
        import coverage  # type: ignore
        if hasattr(coverage, "types") and not hasattr(coverage.types, "Tracer") and hasattr(coverage.types, "TracerCore"):
            coverage.types.Tracer = coverage.types.TracerCore  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import shap  # type: ignore

        X_trans = pipeline.named_steps["preprocessor"].transform(X_sample)
        model = pipeline.named_steps["model"]

        explainer = shap.Explainer(model, X_trans)
        shap_values = explainer(X_trans)
        plt.figure(figsize=(11, 7))
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.axis("off")
        ax.text(0.01, 0.62, "SHAP summary plot could not be generated in this environment.", fontsize=12)
        ax.text(0.01, 0.28, f"Fallback note: {exc}", fontsize=9, wrap=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
