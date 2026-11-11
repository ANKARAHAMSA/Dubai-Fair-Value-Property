"""Cached data/model loaders, unit conversions, and sidebar badges for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import shap
import streamlit as st

from dfv.config import PATHS, SQFT_PER_SQM, get_data_mode
from dfv.models.registry import ModelBundle, load_model_bundle

TX_VIEW_COLUMNS: list[str] = [
    "transaction_id",
    "instance_date",
    "year",
    "property_type",
    "property_sub_type",
    "area_name",
    "project_name",
    "developer_name",
    "rooms",
    "area_sqm",
    "actual_worth",
    "price_per_sqm",
    "is_offplan",
    "nearest_metro",
    "dist_nearest_metro_m",
    "lat",
    "lon",
    "data_source",
]


def resolve_active_dirs() -> tuple[str, Path, Path]:
    """Return `(data_mode, data_dir, model_dir)` honoring DFV_DATA_MODE & available files."""
    mode = get_data_mode()
    if mode == "processed" and (PATHS.processed / "features.parquet").exists():
        m_dir = PATHS.models if (PATHS.models / "metadata.json").exists() else PATHS.sample_models
        return "processed", PATHS.processed, m_dir
    return "sample", PATHS.sample, PATHS.sample_models


@st.cache_resource(show_spinner="Loading fair-value model & SHAP explainer...")
def get_cached_model(data_mode: str, model_dir_str: str) -> tuple[ModelBundle, Any]:
    """Cache serialized LightGBM/Hybrid bundle + SHAP explainer across reruns."""
    _ = data_mode  # cache key includes mode explicitly
    bundle = load_model_bundle(Path(model_dir_str))
    explainer: Any = (
        bundle.point_model
        if hasattr(bundle.point_model, "compute_shap")
        else shap.TreeExplainer(bundle.point_model)
    )
    return bundle, explainer


@st.cache_data(show_spinner="Loading transactions dataset...")
def load_transactions_view(data_mode: str, data_dir_str: str) -> pd.DataFrame:
    """Load slimmed unflagged transactions table cleanly using PyArrow column projection."""
    _ = data_mode
    feat_path = Path(data_dir_str) / "features.parquet"
    df = pd.read_parquet(feat_path, columns=TX_VIEW_COLUMNS)
    df["price_per_sqft"] = df["price_per_sqm"] / SQFT_PER_SQM
    df["area_sqft"] = df["area_sqm"] * SQFT_PER_SQM
    return df


@st.cache_data(show_spinner="Loading market analysis artifacts...")
def load_analysis_bundle(data_mode: str, data_dir_str: str) -> dict[str, Any]:
    """Load cached phase 3/4 analysis JSONs, growth parquet, metro stations, and events."""
    _ = data_mode
    data_dir = Path(data_dir_str)
    analysis_dir = data_dir / "analysis"
    if not analysis_dir.exists():
        analysis_dir = PATHS.sample / "analysis"

    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    growth_path = analysis_dir / "growth_quarterly.parquet"
    growth_df = pd.read_parquet(growth_path) if growth_path.exists() else pd.DataFrame()

    stations_path = data_dir / "metro_stations.parquet"
    if not stations_path.exists():
        stations_path = PATHS.sample / "metro_stations.parquet"
    stations_df = pd.read_parquet(stations_path) if stations_path.exists() else pd.DataFrame()

    events_path = PATHS.reference / "events.csv"
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()

    metrics_path = PATHS.reports / "metrics.json"
    metrics = _read_json(metrics_path)
    err_md_path = PATHS.reports / "error_analysis.md"
    error_md = err_md_path.read_text(encoding="utf-8") if err_md_path.exists() else ""

    return {
        "developer_premium": _read_json(analysis_dir / "developer_premium.json"),
        "metro_effect": _read_json(analysis_dir / "metro_effect.json"),
        "yield_risk": _read_json(analysis_dir / "yield_risk.json"),
        "events_study": _read_json(analysis_dir / "events_study.json"),
        "growth_quarterly": growth_df,
        "metro_stations": stations_df,
        "events_df": events_df,
        "metrics": metrics,
        "error_analysis_md": error_md,
    }


def render_sidebar_controls() -> tuple[str, Path, Path, str]:
    """Render shared sidebar data-mode badge, unit toggle, and observational reminder."""
    mode, data_dir, model_dir = resolve_active_dirs()
    bundle, _ = get_cached_model(mode, str(model_dir))
    data_through = str(bundle.metadata["training_window"]["test_end"])

    if "unit_mode" not in st.session_state:
        st.session_state["unit_mode"] = "sqm"

    with st.sidebar:
        st.markdown("### DubaiFairValue")
        if mode == "sample":
            st.warning(
                f"**Sample Data Mode** (synthetic CI bundle)\n\nData through: `{data_through}`"
            )
        else:
            st.success(
                f"**Real DLD (Fallback) Mode**\n\n"
                f"Sales: real public extract through `{data_through}`; rents/index fallback documented."
            )

        unit_choice = st.radio(
            "Display Size & Price Unit",
            options=["sqm", "sqft"],
            format_func=lambda u: "Metric (AED / sqm)" if u == "sqm" else "Imperial (AED / sqft)",
            index=0 if st.session_state["unit_mode"] == "sqm" else 1,
            key="dfv_unit_radio",
        )
        st.session_state["unit_mode"] = str(unit_choice)
        st.caption(
            "Not financial advice. Market studies report observational associations, not guaranteed causality."
        )

    return mode, data_dir, model_dir, str(unit_choice)


def ppsqm_to_display(val_ppsqm: float, unit: str) -> float:
    """Convert AED/sqm into active display unit (`sqm` or `sqft`)."""
    return float(val_ppsqm) / SQFT_PER_SQM if unit == "sqft" else float(val_ppsqm)


def area_sqm_to_display(val_sqm: float, unit: str) -> float:
    """Convert sqm area into active display unit (`sqm` or `sqft`)."""
    return float(val_sqm) * SQFT_PER_SQM if unit == "sqft" else float(val_sqm)


def unit_suffix(unit: str) -> str:
    """Return human-readable area unit suffix ('sqft' or 'sqm')."""
    return "sqft" if unit == "sqft" else "sqm"
