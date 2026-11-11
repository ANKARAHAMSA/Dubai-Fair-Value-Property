"""Page 6: Off-Plan vs Ready 'Developer Premium' Hedonic Estimates & Time-to-Catch-Up."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Developer Premium — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    load_analysis_bundle,
    render_sidebar_controls,
)

mode, data_dir, _model_dir, _unit = render_sidebar_controls()
analyses = load_analysis_bundle(mode, str(data_dir))
dev = analyses.get("developer_premium", {})

st.title("Off-Plan vs Ready ('Developer Premium') Analysis")
if not dev:
    st.info("Developer premium analysis results are not yet available.")
    st.stop()

st.info(str(dev.get("caveat", "")))

aq = dev.get("overall_area_quarter_fe", {})
pq = dev.get("overall_project_quarter_fe", {})
mt = dev.get("matched_comparison", {})

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Area x Quarter FE Premium",
    f"{aq.get('pct_effect', 'n/a'):+.2f}%" if aq else "n/a",
    delta=f"95% CI [{aq.get('ci_low_pct', 0):.1f}%, {aq.get('ci_high_pct', 0):.1f}%]"
    if aq
    else None,
)
k2.metric(
    "Project x Quarter FE Premium",
    f"{pq.get('pct_effect', 'n/a'):+.2f}%" if pq else "n/a",
    delta=f"95% CI [{pq.get('ci_low_pct', 0):.1f}%, {pq.get('ci_high_pct', 0):.1f}%]"
    if pq
    else None,
)
k3.metric(
    "Matched Exact-Cell Premium",
    f"{mt.get('matched_pct_premium', 'n/a'):+.2f}%"
    if mt.get("matched_pct_premium") is not None
    else "n/a",
    delta=f"{mt.get('matched_cells', 0)} matched cells",
)
k4.metric("Unknown Developer Share", f"{dev.get('unknown_developer_pct', 0.0):.1f}%")

by_area = pd.DataFrame(dev.get("by_area", []))
if not by_area.empty:
    top_bot = pd.concat([by_area.head(12), by_area.tail(8)]).drop_duplicates("area_name")
    top_bot = top_bot.sort_values("pct_premium")
    fig_a = px.bar(
        top_bot,
        x="pct_premium",
        y="area_name",
        orientation="h",
        color="pct_premium",
        color_continuous_scale="RdYlGn",
        hover_data=["n_offplan", "n_ready", "ci_low_pct", "ci_high_pct"],
        title="Within-Area Off-Plan vs Ready Association (% vs Ready, Year-Quarter FE)",
        labels={"pct_premium": "% Price/sqm Association", "area_name": "Community"},
    )
    st.plotly_chart(fig_a, width="stretch")

by_dev = pd.DataFrame(dev.get("by_developer", []))
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Named Developer Breakdown")
    if by_dev.empty:
        st.warning(
            "Named developer breakdown is empty in this dataset because `developer_name == 'Unknown'` "
            "for ~100% of records until `dld_projects-open` metadata is linked."
        )
    else:
        st.dataframe(by_dev, width="stretch", hide_index=True)

with col_r:
    st.subheader("Time-to-Catch-Up (Ready Resales Reaching Off-Plan Entry Price)")
    cu = pd.DataFrame(dev.get("time_to_catch_up", []))
    if cu.empty:
        st.info("Insufficient cohort overlap to estimate time-to-catch-up.")
    else:
        st.dataframe(cu.head(15), width="stretch", hide_index=True)
