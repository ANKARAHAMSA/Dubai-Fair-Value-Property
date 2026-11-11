"""Page 4: Interactive PyDeck Map of Communities, Metro Stations & Metro-Proximity Overlay."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Spatial & Metro Map — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    load_analysis_bundle,
    load_transactions_view,
    ppsqm_to_display,
    render_sidebar_controls,
    unit_suffix,
)

mode, data_dir, _, unit = render_sidebar_controls()
tx_df = load_transactions_view(mode, str(data_dir))
analyses = load_analysis_bundle(mode, str(data_dir))
stations_df = analyses["metro_stations"]
metro_res = analyses["metro_effect"]
u_lbl = unit_suffix(unit)

st.title("Dubai Spatial & Metro-Proximity Map")
st.markdown(
    "Explore Dubai community price levels alongside RTA Metro stations and inspect how metro "
    "proximity relates to sale and rental prices."
)

max_dist = st.slider(
    "Maximum Distance to Nearest Metro Station (m)",
    min_value=300,
    max_value=10_000,
    value=3000,
    step=100,
)

area_agg = (
    tx_df.groupby("area_name", as_index=False)
    .agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        median_ppsqm=("price_per_sqm", "median"),
        dist_nearest_metro_m=("dist_nearest_metro_m", "median"),
        tx_count=("actual_worth", "count"),
    )
    .dropna(subset=["lat", "lon"])
)
area_filtered = area_agg.loc[area_agg["dist_nearest_metro_m"] <= float(max_dist)].copy()

if area_filtered.empty:
    st.warning(
        f"No communities fall within {max_dist:,} m of a Metro station. Increase the slider."
    )
else:
    area_filtered["disp_ppsqm"] = area_filtered["median_ppsqm"].map(
        lambda v: round(ppsqm_to_display(v, unit), 0)
    )
    area_filtered["radius_m"] = (
        area_filtered["tx_count"]
        .clip(lower=10, upper=8000)
        .map(lambda c: float(max(220.0, min(1100.0, (float(c) ** 0.4) * 45.0))))
    )
    p_min = float(area_filtered["disp_ppsqm"].min())
    p_max = float(max(area_filtered["disp_ppsqm"].max(), p_min + 1.0))

    def _color_row(v: float) -> list[int]:
        t = max(0.0, min(1.0, (v - p_min) / (p_max - p_min)))
        return [int(30 + 210 * t), int(160 - 110 * t), int(180 - 140 * t), 190]

    area_records = [
        {
            "hover_title": str(r["area_name"]),
            "hover_detail": f"Median AED/{u_lbl}: {int(r['disp_ppsqm']):,} | Metro dist: {int(round(float(r['dist_nearest_metro_m']))):,}m",
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "radius_m": float(r["radius_m"]),
            "fill_color": _color_row(float(r["disp_ppsqm"])),
        }
        for _, r in area_filtered.iterrows()
    ]

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=area_records,
            get_position=["lon", "lat"],
            get_radius="radius_m",
            get_fill_color="fill_color",
            pickable=True,
            auto_highlight=True,
        )
    ]

    if not stations_df.empty:
        st_records = [
            {
                "hover_title": f"Metro: {r['station_name']}",
                "hover_detail": f"Line: {r.get('line', 'Metro')}",
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
            for _, r in stations_df.dropna(subset=["lat", "lon"]).iterrows()
        ]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=st_records,
                get_position=["lon", "lat"],
                get_radius=170,
                get_fill_color=[29, 78, 216, 220],
                pickable=True,
            )
        )

    view = pdk.ViewState(latitude=25.12, longitude=55.20, zoom=10.3, pitch=0)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{hover_title}\n{hover_detail}"},
    )
    st.pydeck_chart(deck, width="stretch")

st.divider()
st.subheader("Metro Proximity Hedonic Overlay & Route 2020 Robustness Check")
if metro_res:
    st.info(str(metro_res.get("caveat", "")))
    sales_b = metro_res.get("sales_distance_bands") or metro_res.get("sales_cross_area_bands") or {}
    rent_b = metro_res.get("rents_distance_bands") or {}
    col1, col2 = st.columns(2)
    with col1:
        if sales_b:
            s_df = pd.DataFrame(sales_b.values())
            title_s = (
                "Sales Metro Distance Bands (Within-Area x Quarter FE)"
                if metro_res.get("within_area_identifiable", True)
                else "Sales Metro Distance Bands (Cross-Area Caveat — Centroids Only)"
            )
            fig_sb = px.bar(
                s_df,
                x="term",
                y="pct_effect",
                error_y=s_df["ci_high_pct"] - s_df["pct_effect"],
                title=title_s,
                labels={"pct_effect": "% vs >= 2 km", "term": "Distance Band"},
            )
            st.plotly_chart(fig_sb, width="stretch")
    with col2:
        if rent_b:
            r_df = pd.DataFrame(rent_b.values())
            fig_rb = px.bar(
                r_df,
                x="term",
                y="pct_effect",
                error_y=r_df["ci_high_pct"] - r_df["pct_effect"],
                title="Rent Metro Distance Bands (Area x Quarter FE)",
                labels={"pct_effect": "% Rent vs >= 2 km", "term": "Distance Band"},
            )
            st.plotly_chart(fig_rb, width="stretch")
