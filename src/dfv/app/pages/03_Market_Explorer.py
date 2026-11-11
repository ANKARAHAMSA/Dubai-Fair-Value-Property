"""Page 3: Interactive Market Explorer with community, type, rooms, year, and unit filters."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Market Explorer — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    load_transactions_view,
    ppsqm_to_display,
    render_sidebar_controls,
    unit_suffix,
)

mode, data_dir, _, unit = render_sidebar_controls()
tx_df = load_transactions_view(mode, str(data_dir))
u_lbl = unit_suffix(unit)

st.title("Market Explorer")
st.markdown(
    "Filter official DLD / sample residential transactions by community, size, status, and period."
)

all_areas = sorted(tx_df["area_name"].unique().tolist())
default_areas = [
    a
    for a in ("Downtown Dubai", "Dubai Marina", "Business Bay", "Jumeirah Village Circle")
    if a in all_areas
] or all_areas[:3]

min_yr = int(tx_df["year"].min()) if not tx_df.empty else 2015
max_yr = int(tx_df["year"].max()) if not tx_df.empty else 2024

f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
with f1:
    sel_areas = st.multiselect(
        "Communities (leave empty for all)", options=all_areas, default=default_areas
    )
with f2:
    sel_types = st.multiselect(
        "Property Type", options=["Unit", "Villa"], default=["Unit", "Villa"]
    )
with f3:
    status_opt = st.selectbox("Sale Type", options=["All", "Ready Only", "Off-Plan Only"], index=0)
with f4:
    yr_range = st.slider(
        "Year Range", min_value=min_yr, max_value=max_yr, value=(max(min_yr, 2018), max_yr)
    )

room_opts = sorted(tx_df["rooms"].unique().tolist())
sel_rooms = st.multiselect("Bedrooms (0 = Studio)", options=room_opts, default=room_opts[:5])

sub = tx_df.loc[
    tx_df["year"].between(yr_range[0], yr_range[1])
    & (tx_df["property_type"].isin(sel_types) if sel_types else True)
    & (tx_df["rooms"].isin(sel_rooms) if sel_rooms else True)
].copy()
if sel_areas:
    sub = sub.loc[sub["area_name"].isin(sel_areas)]
if status_opt == "Ready Only":
    sub = sub.loc[sub["is_offplan"] == 0]
elif status_opt == "Off-Plan Only":
    sub = sub.loc[sub["is_offplan"] == 1]

if sub.empty:
    st.warning(
        "No transactions match the selected filters. Widen your year range, areas, or bedroom selection."
    )
    st.stop()

sub["disp_ppsqm"] = sub["price_per_sqm"].map(lambda v: ppsqm_to_display(v, unit))
sub["quarter"] = pd.to_datetime(sub["instance_date"]).dt.to_period("Q").astype(str)

m1, m2, m3 = st.columns(3)
m1.metric("Filtered Transactions", f"{len(sub):,}")
m2.metric(f"Median Price / {u_lbl}", f"AED {sub['disp_ppsqm'].median():,.0f}")
m3.metric("Median Total Price", f"AED {sub['actual_worth'].median():,.0f}")

trend = (
    sub.groupby(["quarter", "area_name"], as_index=False)
    .agg(median_price=("disp_ppsqm", "median"), tx_count=("actual_worth", "count"))
    .sort_values("quarter")
)

c_left, c_right = st.columns(2)
with c_left:
    # Limit legend lines to top 8 areas by count for clean readability
    top8 = sub["area_name"].value_counts().head(8).index.tolist()
    fig_tr = px.line(
        trend.loc[trend["area_name"].isin(top8)],
        x="quarter",
        y="median_price",
        color="area_name",
        title=f"Quarterly Median Price Trajectory (AED / {u_lbl})",
        labels={"median_price": f"AED / {u_lbl}", "quarter": "Quarter", "area_name": "Community"},
    )
    st.plotly_chart(fig_tr, width="stretch")

with c_right:
    vol = sub.groupby(["quarter", "is_offplan"], as_index=False).agg(
        tx_count=("actual_worth", "count")
    )
    vol["Status"] = vol["is_offplan"].map({1: "Off-plan", 0: "Ready"})
    fig_vol = px.bar(
        vol,
        x="quarter",
        y="tx_count",
        color="Status",
        title="Quarterly Transaction Volume (Ready vs Off-Plan)",
        labels={"tx_count": "Transactions", "quarter": "Quarter"},
    )
    st.plotly_chart(fig_vol, width="stretch")

# Downsample histogram safely on huge subsets for snappy browser rendering
hist_df = sub.sample(min(len(sub), 25_000), random_state=42)
fig_hist = px.histogram(
    hist_df,
    x="disp_ppsqm",
    color="property_type",
    nbins=50,
    title=f"Distribution of Transaction Prices (AED / {u_lbl})",
    labels={"disp_ppsqm": f"Price (AED / {u_lbl})", "property_type": "Type"},
)
st.plotly_chart(fig_hist, width="stretch")
