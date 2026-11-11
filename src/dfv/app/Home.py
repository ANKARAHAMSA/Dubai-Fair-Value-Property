"""Home page for DubaiFairValue multi-page Streamlit dashboard (`make app`)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DubaiFairValue — Dubai Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dfv.app.components.data_loader import (  # noqa: E402
    get_cached_model,
    load_analysis_bundle,
    load_transactions_view,
    ppsqm_to_display,
    render_sidebar_controls,
    unit_suffix,
)

mode, data_dir, model_dir, unit = render_sidebar_controls()
bundle, _ = get_cached_model(mode, str(model_dir))
tx_df = load_transactions_view(mode, str(data_dir))
analyses = load_analysis_bundle(mode, str(data_dir))
rent_path = data_dir / "rent_contracts.parquet"
n_rents = len(pd.read_parquet(rent_path, columns=["contract_id"])) if rent_path.exists() else 0

st.title("DubaiFairValue")
st.subheader(
    "Instant fair-value estimates, calibrated 90% confidence intervals, SHAP drivers, "
    "and Dubai market analytics"
)

if mode == "sample":
    st.info(
        f"**Sample Data Mode active**: running on the lightweight committed CI synthetic bundle "
        f"(`data/sample/`, {len(tx_df):,} sales + {n_rents:,} rent contracts). Connect processed Dubai Pulse / DLD data "
        "in `data/raw/` and run `make data features train analyze` for full-scale estimates."
    )
else:
    st.success(
        f"**Active Dataset**: `{bundle.metadata.get('dataset_mode', mode)}` | "
        f"Data through `{bundle.metadata['training_window']['test_end']}` | "
        f"Model `{bundle.metadata['model_version']}`"
    )

# Headline computed stats
m_meta = bundle.metadata.get("metrics", {})
tm = m_meta.get("test_metrics", m_meta)
recent_med_ppsqm = float(tx_df["price_per_sqm"].median()) if not tx_df.empty else 0.0
disp_med = ppsqm_to_display(recent_med_ppsqm, unit)
best_base_lbl = (
    str(tm.get("best_baseline_name", "baseline")).replace("baseline_", "").replace("_", " ")
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unflagged Residential Sales", f"{len(tx_df):,}")
c2.metric("Communities Covered", f"{tx_df['area_name'].nunique():,}")
c3.metric(f"Dataset Median AED/{unit_suffix(unit)}", f"AED {disp_med:,.0f}")
c4.metric(
    "Out-of-Time Test MdAPE",
    f"{float(tm.get('model_mdape', 0.0)) * 100:.2f}%"
    if tm.get("model_mdape") is not None
    else "n/a",
    delta=f"-{float(tm.get('mdape_improvement_rel_pct', 0.0)):.1f}% vs {best_base_lbl}"
    if tm.get("mdape_improvement_rel_pct") is not None
    else None,
)

c5, c6, c7, _ = st.columns(4)
c5.metric(
    "90% CQR Interval Coverage",
    f"{float(tm.get('interval_coverage', 0.0)) * 100:.1f}%"
    if tm.get("interval_coverage") is not None
    else "n/a",
)
top_yield = analyses.get("yield_risk", {}).get("area_ranking", [])
if top_yield:
    c6.metric(
        "Top Gross Yield-per-Risk Area",
        str(top_yield[0]["area_name"]),
        delta=f"{top_yield[0]['gross_yield_pct']:.2f}% gross yield",
    )
else:
    c6.metric("Top Gross Yield-per-Risk Area", "n/a")
ev_check = analyses.get("events_study", {}).get("official_index_sanity_check", {})
c7.metric(
    "Index vs Official Sanity r",
    f"{ev_check['pearson_corr']:.3f}" if ev_check.get("available") else "n/a",
)

st.divider()
st.markdown("### Explore Modules")
nav_cols = st.columns(3)
with nav_cols[0]:
    st.page_link(
        "pages/02_Fair_Value_Checker.py", label="Fair-Value Checker (SHAP + Verdict)", icon="🏷️"
    )
    st.page_link("pages/03_Market_Explorer.py", label="Market Explorer", icon="📊")
    st.page_link("pages/04_Map.py", label="Dubai Spatial & Metro Map", icon="🗺️")
with nav_cols[1]:
    st.page_link("pages/05_Yield_and_Risk.py", label="Yield & Risk Budget Tool", icon="💹")
    st.page_link("pages/06_Developer_Premium.py", label="Off-Plan Developer Premium", icon="🏗️")
    st.page_link("pages/07_Time_Travel.py", label="Price Growth Time-Travel", icon="⏳")
with nav_cols[2]:
    st.page_link("pages/08_Events.py", label="Market Cycles & Events", icon="📅")
    st.page_link("pages/09_Ask_the_Data.py", label="Ask the Data (NL Search)", icon="🔎")
    st.page_link(
        "pages/10_Model_and_Limits.py", label="Model Card, Error Analysis & Limits", icon="🧪"
    )

st.caption(
    "Disclaimer: Valuations and market analytics are statistical estimates for educational/research use "
    "only and do not constitute regulated appraisal or investment advice."
)
