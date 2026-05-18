from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "best_model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
RESULTS_PATH = BASE_DIR / "artifacts" / "model_benchmark.json"
ROC_PLOT = BASE_DIR / "plots" / "roc_curve.png"
IMP_PLOT = BASE_DIR / "plots" / "feature_importance.png"
SHAP_PLOT = BASE_DIR / "plots" / "shap_summary.png"
COMP_PLOT = BASE_DIR / "plots" / "model_comparison.png"
DATA_PATH = BASE_DIR / "data" / "hr_attrition.csv"

st.set_page_config(page_title="HR Attrition Risk Scoring", layout="wide")

st.title("HR Attrition Risk Scoring Dashboard")
st.caption("Upload data, score attrition risk, and inspect model artifacts.")

if not ARTIFACT_PATH.exists():
    st.error("No model artifact found. Run training first: python -m src.train --data data/hr_attrition.csv")
    st.stop()

bundle = joblib.load(ARTIFACT_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
threshold = bundle.get("threshold", 0.5)

left, right = st.columns([1.1, 1])

with left:
    st.subheader("Batch scoring")
    uploaded = st.file_uploader("Upload a CSV with the same feature columns", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict_proba(df[feature_columns])[:, 1]
            scored = df.copy()
            scored["attrition_probability"] = preds
            scored["risk_segment"] = pd.cut(
                scored["attrition_probability"],
                bins=[-0.01, 0.4, 0.7, 1.0],
                labels=["low", "medium", "high"],
            )
            st.dataframe(scored.sort_values("attrition_probability", ascending=False), use_container_width=True)
            st.download_button(
                "Download scored file",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="attrition_scored.csv",
                mime="text/csv",
            )

with right:
    st.subheader("Model assets")
    cols = st.columns(2)
    for c, p, title in [(cols[0], COMP_PLOT, "Model comparison"), (cols[1], ROC_PLOT, "ROC curve")]:
        if p.exists():
            c.image(str(p), caption=title, use_container_width=True)
    cols2 = st.columns(2)
    for c, p, title in [(cols2[0], IMP_PLOT, "Feature importance"), (cols2[1], SHAP_PLOT, "SHAP summary")]:
        if p.exists():
            c.image(str(p), caption=title, use_container_width=True)

st.divider()
st.subheader("Manual single-employee scoring")
raw = pd.read_csv(DATA_PATH, sep=None, engine="python")
feature_df = raw.drop(columns=[c for c in ["Attrition", "Attrition Reason", "AttritionDate", "IsVoluntary", "Employee Name", "Manager Name", "Join Date", "Employee ID", "EmployeeNumber"] if c in raw.columns], errors="ignore")

with st.form("manual_input"):
    inputs = {}
    col1, col2, col3 = st.columns(3)
    cols_ui = [col1, col2, col3]
    all_cols = list(feature_columns)
    for idx, col in enumerate(all_cols):
        ui = cols_ui[idx % 3]
        series = feature_df[col]
        if series.dtype.kind in "biufc":
            minv = float(np.nanmin(series))
            maxv = float(np.nanmax(series))
            val = float(np.nanmedian(series))
            inputs[col] = ui.number_input(col, value=val, min_value=minv, max_value=maxv)
        else:
            options = [str(x) for x in sorted(series.dropna().astype(str).unique().tolist())]
            inputs[col] = ui.selectbox(col, options)
    submitted = st.form_submit_button("Score employee")

if submitted:
    df_in = pd.DataFrame([inputs])
    prob = float(model.predict_proba(df_in[feature_columns])[:, 1][0])
    risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"
    st.metric("Attrition probability", f"{prob:.2%}")
    st.write(f"Risk segment: **{risk}**")
    st.progress(min(prob, 1.0))
