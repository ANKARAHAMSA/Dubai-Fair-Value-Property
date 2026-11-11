"""Page 5: Rental Yield Heatmap, Risk-Adjusted Ranking & Top-10 Budget Tool."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Yield & Risk — DubaiFairValue", layout="wide")

from dfv.analysis.yield_score import YIELD_CAVEAT, recommend_areas_for_budget  # noqa: E402
from dfv.app.components.data_loader import (  # noqa: E402
    load_analysis_bundle,
    render_sidebar_controls,
)

mode, data_dir, _model_dir, _unit = render_sidebar_controls()
analyses = load_analysis_bundle(mode, str(data_dir))
yr = analyses.get("yield_risk", {})

st.title("Rental Yield & Risk-Adjusted Area Ranking")
st.warning(yr.get("caveat", YIELD_CAVEAT))

area_df = pd.DataFrame(yr.get("area_ranking", []))
seg_list = yr.get("segment_ranking", [])

if area_df.empty:
    st.info("Yield analysis results not yet generated.")
    st.stop()

col_a, col_b = st.columns([3, 2])
with col_a:
    fig_sc = px.scatter(
        area_df,
        x="risk_score",
        y="gross_yield_pct",
        size="sales_12m_count",
        color="yield_per_risk_score",
        hover_name="area_name",
        title="Community Gross Yield (%) vs Composite Risk Score",
        labels={
            "risk_score": "Risk Score (lower = steadier & more liquid)",
            "gross_yield_pct": "Trailing 12m Gross Yield (%)",
            "yield_per_risk_score": "Yield / Risk",
        },
        color_continuous_scale="Tealgrn",
    )
    st.plotly_chart(fig_sc, width="stretch")

with col_b:
    st.subheader("Top Ranked Communities (Overall)")
    st.dataframe(
        area_df[
            [
                "area_name",
                "gross_yield_pct",
                "risk_score",
                "yield_per_risk_score",
                "median_price_aed",
                "confidence",
            ]
        ].head(12),
        width="stretch",
        hide_index=True,
    )

if seg_list:
    seg_df = pd.DataFrame(seg_list)
    top_areas_hm = area_df["area_name"].head(12).tolist()
    hm_piv = (
        seg_df.loc[seg_df["area_name"].isin(top_areas_hm) & (seg_df["rooms"] <= 4)]
        .groupby(["area_name", "rooms"], as_index=False)["gross_yield_pct"]
        .mean()
        .pivot(index="area_name", columns="rooms", values="gross_yield_pct")
    )
    if not hm_piv.empty:
        fig_hm = px.imshow(
            hm_piv,
            text_auto=".1f",
            aspect="auto",
            title="Gross Yield (%) Heatmap by Community & Bedrooms (Top 12 Areas)",
            labels={"x": "Bedrooms (0=Studio)", "y": "Community", "color": "Gross Yield %"},
            color_continuous_scale="YlGnBu",
        )
        st.plotly_chart(fig_hm, width="stretch")

st.divider()
st.subheader("Top 10 Areas for Your Budget")
b1, b2, b3 = st.columns(3)
with b1:
    budget_aed = st.slider(
        "Maximum Target Price (AED)",
        min_value=400_000,
        max_value=10_000_000,
        value=1_500_000,
        step=100_000,
    )
with b2:
    rooms_sel = st.selectbox("Bedrooms Filter", options=["Any", 0, 1, 2, 3, 4], index=0)
with b3:
    type_sel = st.selectbox("Property Type", options=["Any", "Unit", "Villa"], index=0)

recs = recommend_areas_for_budget(
    seg_list,
    budget_aed=float(budget_aed),
    rooms=None if rooms_sel == "Any" else int(str(rooms_sel)),
    property_type=None if type_sel == "Any" else str(type_sel),
    top_k=10,
)
if not recs:
    st.warning(
        "No matching segments fell under your budget with sufficient sample size. Increase the budget slider."
    )
else:
    st.dataframe(
        pd.DataFrame(recs)[
            [
                "area_name",
                "property_type",
                "rooms",
                "median_price_aed",
                "gross_yield_pct",
                "risk_score",
                "yield_per_risk_score",
                "confidence",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
