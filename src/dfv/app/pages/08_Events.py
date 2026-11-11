"""Page 8: Market Cycles, Official DLD Index Sanity Check & Event Studies."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Market Cycles & Events — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    load_analysis_bundle,
    render_sidebar_controls,
)

mode, data_dir, _model_dir, _unit = render_sidebar_controls()
analyses = load_analysis_bundle(mode, str(data_dir))
ev = analyses.get("events_study", {})
metro_res = analyses.get("metro_effect", {})

st.title("Market Cycles, Official Index Cross-Check & Event Studies")
if not ev:
    st.info("Event-study results are not yet available.")
    st.stop()

st.info(str(ev.get("caveat", "")))
chk = ev.get("official_index_sanity_check", {})
if chk.get("available"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Overlap Months Checked", str(chk["overlap_months"]))
    c2.metric("Pearson Correlation vs DLD/Fallback Index", f"{chk['pearson_corr']:.4f}")
    c3.metric("Mean Abs Index Gap (points)", f"{chk['mean_abs_index_gap']:.2f}")

m_df = pd.DataFrame(ev.get("monthly_series", []))
if not m_df.empty:
    m_df["m_dt"] = pd.to_datetime(m_df["month"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=m_df["m_dt"],
            y=m_df["computed_index"],
            mode="lines",
            name="Mix-Stratified Index (2019=100)",
            line={"color": "#0F766E", "width": 2.5},
        )
    )
    if "official_dld_index" in m_df.columns and m_df["official_dld_index"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=m_df["m_dt"],
                y=m_df["official_dld_index"],
                mode="lines",
                name="Official / Fallback DLD Index",
                line={"color": "#1D4ED8", "width": 2.0, "dash": "dash"},
            )
        )
    for item in ev.get("events_analysis", []):
        dt_ev = pd.Timestamp(item["start_date"])
        if dt_ev >= m_df["m_dt"].min() and dt_ev <= m_df["m_dt"].max():
            fig.add_vline(x=dt_ev, line_dash="dot", line_color="#DC2626", opacity=0.55)
            fig.add_annotation(
                x=dt_ev,
                y=float(m_df["computed_index"].max()) * 0.96,
                text=str(item["event_id"]).replace("_", " ")[:18],
                showarrow=False,
                textangle=-90,
                font={"size": 9, "color": "#DC2626"},
            )
    fig.update_layout(
        title="Dubai Monthly Residential Index (2019 = 100) with Verified Events",
        yaxis_title="Index (2019 Mean = 100)",
        height=440,
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("Interrupted Time Series (HC1 OLS) & -6 Month Pre-Event Placebo Checks")
rows = []
for item in ev.get("events_analysis", []):
    its = item.get("its") or {}
    pl = item.get("placebo_minus_6m") or {}
    rows.append(
        {
            "Event": item["title"],
            "Start Date": item["start_date"],
            "ITS Step (%)": its.get("level_step_pct"),
            "ITS Step p-val": its.get("level_step_p_value"),
            "Monthly Slope Δ (%)": its.get("monthly_slope_change_pct"),
            "Placebo -6m Step (%)": pl.get("level_step_pct"),
            "Placebo Step p-val": pl.get("level_step_p_value"),
            "Source URL": item["source_url"],
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

r2020 = metro_res.get("route_2020_event_study", {})
if r2020.get("available"):
    st.divider()
    st.subheader("Route 2020 Metro Opening — Two-Way FE Dynamic Event Study")
    curve = pd.DataFrame(r2020["event_study_curve"])
    fig_r2 = go.Figure()
    fig_r2.add_trace(
        go.Scatter(
            x=curve["rel_quarter"],
            y=curve["pct_effect"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (curve["ci_high_pct"] - curve["pct_effect"]).tolist(),
                "arrayminus": (curve["pct_effect"] - curve["ci_low_pct"]).tolist(),
            },
            mode="lines+markers",
            name="Point Estimate & 95% CI",
            line={"color": "#059669"},
        )
    )
    fig_r2.add_vline(x=-0.5, line_dash="dash", line_color="#DC2626")
    fig_r2.add_hline(y=0.0, line_color="#475569")
    fig_r2.update_layout(
        title="Estimated Price/sqm Differential Around Route 2020 Opening (Ref: Quarter -1)",
        xaxis_title="Quarters Relative to Station Opening",
        yaxis_title="% Effect vs Control Communities",
        height=380,
    )
    st.plotly_chart(fig_r2, width="stretch")
    st.caption(
        "Caveat: Two-way FE DiD compares treated communities near Route 2020 stations (opening 2021-01 / 2021-06) "
        "against outer/core control areas; results reflect post-COVID/Expo structural shifts and rely on area-centroid "
        "coordinates when project-level `dld_projects-open` coordinates are absent."
    )
