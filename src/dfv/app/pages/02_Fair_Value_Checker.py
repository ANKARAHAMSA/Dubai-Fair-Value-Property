"""Page 2: Single-property Fair-Value Checker, 90% CQR interval gauge, SHAP waterfall, and comps."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fair-Value Checker — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    area_sqm_to_display,
    get_cached_model,
    load_transactions_view,
    ppsqm_to_display,
    render_sidebar_controls,
    unit_suffix,
)
from dfv.config import SQFT_PER_SQM  # noqa: E402
from dfv.models.predict import predict_single_property  # noqa: E402

mode, data_dir, model_dir, unit = render_sidebar_controls()
bundle, explainer = get_cached_model(mode, str(model_dir))
tx_df = load_transactions_view(mode, str(data_dir))

st.title("Fair-Value Checker")
st.markdown(
    "Enter a Dubai property's specifications and asking price (AED) to obtain a point estimate, "
    "conformal 90% prediction interval, verdict, and plain-English SHAP cost drivers."
)

area_options = sorted(bundle.area_defaults.keys())
default_idx = area_options.index("Dubai Marina") if "Dubai Marina" in area_options else 0

with st.form("fair_value_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        area_name = st.selectbox("Community / Area", options=area_options, index=default_idx)
        property_type = st.selectbox("Property Type", options=["Unit", "Villa"], index=0)
        is_offplan = st.checkbox("Off-plan sale", value=False)
    with c2:
        rooms = st.number_input("Bedrooms (0 = Studio)", min_value=0, max_value=8, value=2, step=1)
        if unit == "sqft":
            size_input = st.number_input(
                "Internal Area (sqft)", min_value=170.0, max_value=50_000.0, value=1200.0, step=50.0
            )
            area_sqm = float(size_input) / SQFT_PER_SQM
        else:
            size_input = st.number_input(
                "Internal Area (sqm)", min_value=15.0, max_value=5000.0, value=112.0, step=5.0
            )
            area_sqm = float(size_input)
        has_parking = st.checkbox("Parking Included", value=True)
    with c3:
        asking_price = st.number_input(
            "Asking Price (AED)",
            min_value=100_000.0,
            max_value=250_000_000.0,
            value=2_350_000.0,
            step=50_000.0,
        )
        default_metro = float(bundle.area_defaults[area_name]["dist_nearest_metro_m"])
        dist_metro = st.slider(
            "Distance to Nearest Metro (m)",
            min_value=100,
            max_value=10_000,
            value=int(min(max(default_metro, 100), 10_000)),
            step=50,
        )

    submitted = st.form_submit_button("Evaluate Fair Value", type="primary")

if submitted or "last_fv_result" not in st.session_state:
    try:
        res = predict_single_property(
            {
                "area_name": area_name,
                "property_type": property_type,
                "rooms": int(rooms),
                "area_sqm": float(area_sqm),
                "asking_price_aed": float(asking_price),
                "is_offplan": bool(is_offplan),
                "has_parking": bool(has_parking),
                "dist_nearest_metro_m": float(dist_metro),
            },
            bundle=bundle,
            explainer_or_model=explainer,
        )
        st.session_state["last_fv_result"] = res
        st.session_state["last_fv_ctx"] = {
            "area_name": area_name,
            "property_type": property_type,
            "rooms": int(rooms),
            "area_sqm": float(area_sqm),
            "asking_price_aed": float(asking_price),
        }
    except Exception as exc:
        st.error(f"Could not compute estimate safely: {exc}")
        st.stop()

res = st.session_state["last_fv_result"]
ctx = st.session_state["last_fv_ctx"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Fair Value", f"AED {res['estimate']:,.0f}")
k2.metric("90% Confidence Range", f"AED {res['low']:,.0f} – {res['high']:,.0f}")
k3.metric("Asking Gap vs Median", f"{res['gap_pct']:+.1f}%")
k4.metric("Confidence Label", str(res["confidence"]))

verdict_str = str(res["verdict"])
if verdict_str == "Underpriced":
    st.success(f"**Verdict: UNDERPRICED** — {res['reason']}")
elif verdict_str == "Fair":
    st.info(f"**Verdict: FAIR VALUE** — {res['reason']}")
elif verdict_str == "Overpriced":
    st.warning(f"**Verdict: OVERPRICED** — {res['reason']}")
else:
    st.warning(f"**Verdict: INSUFFICIENT DATA** — {res['reason']}")

col_l, col_r = st.columns(2)

with col_l:
    # Horizontal range bullet/gauge comparing asking price vs [low, med, high]
    fig_rng = go.Figure()
    fig_rng.add_trace(
        go.Bar(
            x=[res["high"] - res["low"]],
            y=["Valuation"],
            base=[res["low"]],
            orientation="h",
            name="Calibrated 90% Interval",
            marker={"color": "rgba(15, 118, 110, 0.25)"},
            hovertemplate="90% Range: AED %{base:,.0f} - %{x:,.0f}<extra></extra>",
        )
    )
    fig_rng.add_trace(
        go.Scatter(
            x=[res["estimate"]],
            y=["Valuation"],
            mode="markers+text",
            text=[f"Median AED {res['estimate'] / 1e6:.2f}M"],
            textposition="top center",
            name="Fair Value Median",
            marker={"size": 14, "color": "#0F766E", "symbol": "diamond"},
        )
    )
    fig_rng.add_trace(
        go.Scatter(
            x=[ctx["asking_price_aed"]],
            y=["Valuation"],
            mode="markers+text",
            text=[f"Asking AED {ctx['asking_price_aed'] / 1e6:.2f}M"],
            textposition="bottom center",
            name="Asking Price",
            marker={"size": 14, "color": "#DC2626", "symbol": "x"},
        )
    )
    fig_rng.update_layout(
        title="Asking Price vs Calibrated 90% Fair-Value Band (AED)",
        height=280,
        margin={"l": 20, "r": 20, "t": 45, "b": 25},
        xaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig_rng, width="stretch")

with col_r:
    drivers = res.get("drivers", [])
    if drivers:
        fig_wf = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative"] * len(drivers) + ["total"],
                x=[d["feature"] for d in drivers] + ["Total Net Delta"],
                y=[d["aed_impact"] for d in drivers] + [0.0],
                text=[d["text"].split(":")[0] for d in drivers] + ["Net"],
                textposition="outside",
                increasing={"marker": {"color": "#0F766E"}},
                decreasing={"marker": {"color": "#DC2626"}},
                totals={"marker": {"color": "#1D4ED8"}},
            )
        )
        fig_wf.update_layout(
            title="Top SHAP Price Drivers (Log-to-AED MVT Decomposition)",
            height=300,
            margin={"l": 20, "r": 20, "t": 45, "b": 35},
            yaxis_title="Contribution (AED)",
        )
        st.plotly_chart(fig_wf, width="stretch")
    for d in drivers:
        st.markdown(f"- `{d['text']}`")

st.divider()
st.subheader("Comparable Recent Sales")
comps = tx_df.loc[
    (tx_df["area_name"] == ctx["area_name"])
    & (tx_df["property_type"] == ctx["property_type"])
    & (tx_df["rooms"] == ctx["rooms"])
].copy()
if comps.empty:
    comps = tx_df.loc[tx_df["area_name"] == ctx["area_name"]].copy()

if comps.empty:
    st.info("No recent comparables found in the active dataset for this segment.")
else:
    comps["size_diff"] = (comps["area_sqm"] - ctx["area_sqm"]).abs()
    top_comps = comps.sort_values(["instance_date", "size_diff"], ascending=[False, True]).head(12)
    disp = top_comps[
        [
            "instance_date",
            "area_name",
            "project_name",
            "property_type",
            "rooms",
            "actual_worth",
            "price_per_sqm",
            "area_sqm",
            "is_offplan",
        ]
    ].copy()
    disp[f"Area ({unit_suffix(unit)})"] = disp["area_sqm"].map(
        lambda v: round(area_sqm_to_display(v, unit), 1)
    )
    disp[f"Price / {unit_suffix(unit)} (AED)"] = disp["price_per_sqm"].map(
        lambda v: round(ppsqm_to_display(v, unit), 0)
    )
    disp["Status"] = disp["is_offplan"].map({1: "Off-plan", 0: "Ready"})
    show_cols = [
        "instance_date",
        "project_name",
        "property_type",
        "rooms",
        "Status",
        f"Area ({unit_suffix(unit)})",
        "actual_worth",
        f"Price / {unit_suffix(unit)} (AED)",
    ]
    st.dataframe(disp[show_cols], width="stretch", hide_index=True)

st.caption(
    f"Model: `{res['model_version']}` | Data through: `{res['data_through']}` | "
    "Unobserved floor height, sea/Burj view, interior upgrades, and payment plan terms explain residual variance."
)
