"""Page 7: Time-Travel Animated Price Growth Map (2015 -> Latest)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Time-Travel Growth — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    load_analysis_bundle,
    ppsqm_to_display,
    render_sidebar_controls,
    unit_suffix,
)
from dfv.config import PATHS  # noqa: E402

mode, data_dir, _, unit = render_sidebar_controls()
analyses = load_analysis_bundle(mode, str(data_dir))
gq = analyses.get("growth_quarterly")
u_lbl = unit_suffix(unit)

st.title("Time-Travel: Dubai Community Price Growth (2015 → Latest)")
st.markdown(
    "Press Play or drag the quarterly slider to animate residential price-per-"
    + u_lbl
    + " evolution across Dubai's most liquid communities."
)

if gq is None or gq.empty:
    st.info("Quarterly growth table is empty.")
    st.stop()

work = gq.loc[gq["lat"].between(24.80, 25.35) & gq["lon"].between(55.02, 55.45)].copy()
top_areas = work.groupby("area_name")["tx_count"].sum().nlargest(25).index.tolist()
work = work.loc[work["area_name"].isin(top_areas)].sort_values("quarter_date")
work["disp_ppsqm"] = work["median_ppsqm"].map(lambda v: round(ppsqm_to_display(v, unit), 0))

c_min = float(work["disp_ppsqm"].quantile(0.05))
c_max = float(work["disp_ppsqm"].quantile(0.95))

fig_anim = px.scatter_map(
    work,
    lat="lat",
    lon="lon",
    size="tx_count",
    color="disp_ppsqm",
    hover_name="area_name",
    hover_data=["tx_count", "cumulative_growth_pct", "yoy_growth_pct"],
    animation_frame="year_quarter",
    color_continuous_scale="YlOrRd",
    range_color=[c_min, c_max],
    size_max=34,
    zoom=9.8,
    height=560,
    title=f"Animated Quarterly Price/Size Bubble Map (AED / {u_lbl})",
)
fig_anim.update_layout(map_style="carto-positron", margin={"l": 0, "r": 0, "t": 45, "b": 0})
st.plotly_chart(fig_anim, width="stretch")

latest_q = str(work["year_quarter"].max())
st.subheader(f"Latest Quarter Snapshot ({latest_q})")
latest_tbl = (
    work.loc[work["year_quarter"] == latest_q]
    .sort_values("cumulative_growth_pct", ascending=False)[
        ["area_name", "disp_ppsqm", "tx_count", "yoy_growth_pct", "cumulative_growth_pct"]
    ]
    .rename(columns={"disp_ppsqm": f"Median AED/{u_lbl}"})
)
st.dataframe(latest_tbl, width="stretch", hide_index=True)

gif_path = PATHS.figures / "price_growth_timelapse.gif"
if gif_path.exists():
    with st.expander("Pre-rendered Static Timelapse GIF (for README / LinkedIn)"):
        st.image(str(gif_path), caption="Dubai Residential Price Growth Timelapse")
