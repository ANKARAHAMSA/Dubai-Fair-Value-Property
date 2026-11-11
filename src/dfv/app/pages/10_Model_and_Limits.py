"""Page 10: Model Performance, Error Analysis ('What the Model Gets Wrong'), Sources & Limits."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model & Limits — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    get_cached_model,
    load_analysis_bundle,
    render_sidebar_controls,
)
from dfv.config import PATHS  # noqa: E402

mode, data_dir, model_dir, _ = render_sidebar_controls()
bundle, _ = get_cached_model(mode, str(model_dir))
analyses = load_analysis_bundle(mode, str(data_dir))
bundle_metrics = bundle.metadata.get("metrics", {})
if mode == "sample":
    arch_name = str(bundle.metadata.get("model_version", "ridge_residual_lightgbm")).split("-cqr-")[
        0
    ]
    metrics = {
        "model_architecture": arch_name,
        "test_metrics": bundle_metrics,
        "baselines": (
            [
                {
                    "name": bundle_metrics["best_baseline_name"],
                    "mdape": bundle_metrics["best_baseline_mdape"],
                }
            ]
            if bundle_metrics.get("best_baseline_name")
            else []
        ),
        "rolling_cv": [],
    }
else:
    metrics = analyses.get("metrics") or {
        "model_architecture": "lightgbm_tuned",
        "test_metrics": bundle_metrics,
    }

st.title("Model Card, Honest Error Analysis & Data Limits")

tm = metrics.get("test_metrics", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Architecture", str(metrics.get("model_architecture", "n/a")))
c2.metric(
    "Out-of-Time Test MdAPE",
    f"{float(tm.get('model_mdape', 0.0)) * 100:.2f}%"
    if tm.get("model_mdape") is not None
    else "n/a",
)
c3.metric(
    "Nominal 90% Coverage",
    f"{float(tm.get('interval_coverage', 0.0)) * 100:.2f}%"
    if tm.get("interval_coverage") is not None
    else "n/a",
)
c4.metric(
    "Spatial Unseen-Project MdAPE",
    f"{float(tm.get('spatial_mdape', 0.0)) * 100:.2f}%"
    if tm.get("spatial_mdape") is not None
    else "n/a",
)

# Baseline comparison chart
baselines = list(metrics.get("baselines", []))
if tm.get("model_mdape") is not None:
    comp_rows = [
        {"Model": str(b["name"]), "Test MdAPE (%)": round(float(b["mdape"]) * 100.0, 2)}
        for b in baselines
    ]
    comp_rows.append(
        {
            "Model": f"{metrics.get('model_architecture', 'primary')} (Primary)",
            "Test MdAPE (%)": round(float(tm["model_mdape"]) * 100.0, 2),
        }
    )
    comp_df = pd.DataFrame(comp_rows).sort_values("Test MdAPE (%)", ascending=True)
    fig_b = px.bar(
        comp_df,
        x="Test MdAPE (%)",
        y="Model",
        orientation="h",
        title="Out-of-Time Test Set Comparison vs Baselines (Lower MdAPE is Better)",
        color="Test MdAPE (%)",
        color_continuous_scale="Teal_r",
    )
    st.plotly_chart(fig_b, width="stretch")

rolling = metrics.get("rolling_cv", [])
if rolling:
    cv_df = pd.DataFrame(rolling)
    cv_df["mdape_pct"] = pd.to_numeric(cv_df["mdape"], errors="coerce") * 100.0
    fig_cv = px.line(
        cv_df,
        x="fold_start",
        y="mdape_pct",
        markers=True,
        title="Rolling-Origin Cross-Validation MdAPE Across Market Windows",
        labels={"fold_start": "6-Month Test Fold Start", "mdape_pct": "Test MdAPE (%)"},
    )
    st.plotly_chart(fig_cv, width="stretch")

st.divider()
st.subheader("SHAP Interpretability Artifacts")
if mode == "sample":
    st.caption(
        "Note: Metrics above reflect the active Sample CI model; the SHAP & segment error plots below "
        "were exported from the full processed DLD training pipeline."
    )
fig_paths = [
    ("Global SHAP Importance", PATHS.figures / "shap_global_importance.png"),
    ("SHAP Dependence: Metro Distance", PATHS.figures / "shap_dep_metro.png"),
    ("SHAP Dependence: Unit Size (sqm)", PATHS.figures / "shap_dep_area.png"),
    ("SHAP Dependence: Off-Plan Status", PATHS.figures / "shap_dep_offplan.png"),
]
f_cols = st.columns(2)
for idx, (cap, p) in enumerate(fig_paths):
    if p.exists():
        f_cols[idx % 2].image(str(p), caption=cap)

st.divider()
st.subheader("What the Model Gets Wrong (Honest Error Analysis)")
if mode == "sample":
    st.caption(
        "Note: The segment error breakdown below comes from the full processed DLD hold-out evaluation "
        "(`reports/error_analysis.md`), while top-of-page KPIs show the Sample CI model."
    )
err_md = analyses.get("error_analysis_md", "")
if err_md:
    st.markdown(err_md)
else:
    st.info("Run `make train` to generate detailed segment error breakdowns.")

n_active_stations = len(analyses.get("metro_stations", []))
data_end = str(bundle.metadata.get("training_window", {}).get("test_end", "2024-08-22"))

st.divider()
st.subheader("Data Sources, Attribution & Known Gaps")
st.markdown(
    f"""
- **Primary Target Source**: Dubai Land Department (DLD) residential sales data (`{bundle.metadata.get("dataset_mode", mode)}`, data through `{data_end}`). Processed mode uses the verified public extract fallback (`viewit-ai/full-dubai-pulse`) where permission-gated Dubai Pulse downloads are pending.
- **Transit Data**: Active dataset contains `{n_active_stations}` metro stations (processed GTFS `mdb-904` + Route 2020 opening dates in `data/reference/stations_opening.csv`; 14-station core corridor subset in committed `data/sample/`).
- **Known Limitations**:
  1. **No interior / floor / view attributes**: Sea views, Burj Khalifa views, high floors, corner layouts, and developer snagging/payment plans are unrecorded in DLD transaction registries.
  2. **Area-centroid fallback coordinates**: Without unit-level or geocoded building polygons (`dld_projects-open`), unseen projects degrade on spatial hold-out (see Spatial Hold-out MdAPE above) and within-area sales metro premiums cannot be isolated on centroid fallback data.
  3. **Associational, not causal**: Fixed-effects hedonic and ITS event analyses quantify empirical associations; concurrent macro cycles and unobserved amenities prevent pure causal claims unless quasi-experimental assumptions hold.
"""
)
