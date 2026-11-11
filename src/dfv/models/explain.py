"""SHAP attribution (TreeSHAP + exact additive Linear SHAP), log-to-AED conversion, and drivers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap

if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_READABLE_LABELS: dict[str, str] = {
    "area_sqm": "Unit size (sqm)",
    "log_area_sqm": "Unit size (log sqm)",
    "rooms": "Bedrooms",
    "has_parking": "Parking included",
    "is_offplan": "Off-plan vs Ready status",
    "months_to_completion": "Months to completion",
    "project_age_years": "Project age",
    "completion_pct": "Project completion %",
    "delivery_delay_months": "Delivery delay",
    "dist_nearest_metro_m": "Distance to nearest Metro",
    "dist_nearest_tram_m": "Distance to nearest Tram",
    "dist_nearest_mall_m": "Distance to nearest Mall",
    "dist_nearest_coast_m": "Distance to coastline",
    "dist_airport_m": "Distance to airport",
    "dist_downtown_m": "Distance to Downtown Dubai",
    "poi_count_1km": "Amenities within 1 km",
    "bus_stop_density_1km": "Transit stops within 1 km",
    "metro_within_500m": "Within 500m of Metro",
    "metro_within_1000m": "Within 1 km of Metro",
    "lat": "Latitude (micro-location)",
    "lon": "Longitude (micro-location)",
    "area_trailing_3m_ppsqm": "Area 3m price level",
    "area_trailing_6m_ppsqm": "Area 6m price level",
    "area_trailing_12m_ppsqm": "Area 12m price level",
    "log_area_trailing_12m_ppsqm": "Area baseline valuation",
    "area_trailing_3m_tx_count": "Recent area liquidity (3m)",
    "area_trailing_12m_tx_count": "Area liquidity (12m)",
    "project_trailing_6m_ppsqm": "Project 6m price level",
    "project_trailing_12m_ppsqm": "Project 12m price level",
    "segment_trailing_12m_ppsqm": "Comparable segment price",
    "log_segment_trailing_12m_ppsqm": "Comparable segment baseline",
    "area_trailing_12m_rent_ppsqm": "Area trailing rent level",
    "area_trailing_12m_yield": "Area trailing rental yield",
    "year": "Market year",
    "quarter": "Quarter",
    "month": "Month",
    "days_since_start": "Market cycle trend",
    "property_type": "Property type (Unit/Villa)",
    "property_sub_type": "Sub-type",
    "area_name": "Community / Area",
    "master_project": "Master community",
    "project_name": "Building / Project",
    "developer_name": "Developer",
    "nearest_metro": "Nearest Metro station",
    "developer_name_oof_enc": "Developer historical price tier",
    "project_name_oof_enc": "Project historical price tier",
}


@dataclass(frozen=True)
class DriverItem:
    """Single human-readable SHAP price driver."""

    feature: str
    aed_impact: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "aed_impact": round(float(self.aed_impact), 2),
            "text": self.text,
        }


def format_aed_short(amount: float) -> str:
    """Format signed AED impact concisely (+AED 180k, -AED 1.2M)."""
    sign = "+" if amount >= 0 else "-"
    val = abs(amount)
    if val >= 1_000_000:
        return f"{sign}AED {val / 1_000_000:.2f}M"
    if val >= 1_000:
        return f"{sign}AED {val / 1_000:.0f}k"
    return f"{sign}AED {val:.0f}"


def convert_log_shap_to_aed(
    shap_values_log: np.ndarray,
    base_value_log: float,
    area_sqm: float,
) -> np.ndarray:
    """Convert log(price/sqm) SHAP values into additive total AED contributions.

    Uses exact Mean-Value-Theorem proportional scaling so contributions preserve sign
    and sum identically to (predicted_total_aed - base_expected_total_aed).
    """
    phi = np.asarray(shap_values_log, dtype=float)
    sum_phi = float(np.sum(phi))
    base_total = float(np.exp(base_value_log) * area_sqm)
    pred_total = float(np.exp(base_value_log + sum_phi) * area_sqm)
    delta_total = pred_total - base_total

    if abs(sum_phi) > 1e-9:
        scale = delta_total / sum_phi
        return phi * scale
    return phi * base_total


def compute_log_shap_values(model_or_explainer: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Compute (shap_values_matrix, base_value_log) for either ResidualBoostedLGBM or TreeExplainer."""
    if hasattr(model_or_explainer, "compute_shap"):
        return model_or_explainer.compute_shap(X)
    if isinstance(model_or_explainer, shap.TreeExplainer):
        exp = model_or_explainer(X)
        base_val = float(np.asarray(exp.base_values).reshape(-1)[0])
        return np.asarray(exp.values, dtype=float), base_val
    explainer = shap.TreeExplainer(model_or_explainer)
    exp = explainer(X)
    base_val = float(np.asarray(exp.base_values).reshape(-1)[0])
    return np.asarray(exp.values, dtype=float), base_val


def explain_single_prediction(
    model_or_explainer: Any,
    X_row: pd.DataFrame,
    raw_context: dict[str, Any],
    area_sqm: float,
    top_k: int = 5,
) -> list[DriverItem]:
    """Compute top-k plain-English AED drivers for a single property prediction."""
    mat, base_val = compute_log_shap_values(model_or_explainer, X_row)
    vals = mat[0]

    aed_impacts = convert_log_shap_to_aed(vals, base_val, area_sqm)
    features = list(X_row.columns)

    order = np.argsort(np.abs(aed_impacts))[::-1]
    drivers: list[DriverItem] = []
    for idx in order:
        feat = features[int(idx)]
        impact = float(aed_impacts[int(idx)])
        if abs(impact) < 100.0 and len(drivers) >= 3:
            continue
        label = FEATURE_READABLE_LABELS.get(feat, feat.replace("_", " ").title())
        ctx_val = raw_context.get(feat, X_row.iloc[0][feat])
        if feat == "area_name":
            desc = f"{format_aed_short(impact)}: {ctx_val} location"
        elif feat == "dist_nearest_metro_m":
            desc = f"{format_aed_short(impact)}: Metro distance ({float(ctx_val):,.0f} m)"
        elif feat == "is_offplan":
            status = "Off-plan" if int(ctx_val) == 1 else "Ready property"
            desc = f"{format_aed_short(impact)}: {status}"
        elif feat in {"area_sqm", "log_area_sqm"}:
            desc = f"{format_aed_short(impact)}: Size ({area_sqm:,.0f} sqm)"
        elif feat == "rooms":
            r_lbl = "Studio" if int(ctx_val) == 0 else f"{int(ctx_val)} BR"
            desc = f"{format_aed_short(impact)}: {r_lbl} layout"
        else:
            desc = f"{format_aed_short(impact)}: {label}"
        drivers.append(DriverItem(feature=feat, aed_impact=impact, text=desc))
        if len(drivers) >= top_k:
            break

    return drivers


def generate_shap_plots(
    model: Any,
    X_sample: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate global SHAP importance and key dependence plots into reports/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    vals, _ = compute_log_shap_values(model, X_sample)
    feat_names = list(X_sample.columns)

    saved: list[Path] = []

    # 1. Global mean(|SHAP|) bar chart
    mean_abs = np.mean(np.abs(vals), axis=0)
    top_idx = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_READABLE_LABELS.get(feat_names[i], feat_names[i]) for i in top_idx],
        mean_abs[top_idx],
        color="#0F766E",
    )
    ax.set_xlabel("Mean |SHAP value| (log AED/sqm)")
    ax.set_title("Global Feature Importance (Unified Linear + LightGBM SHAP)")
    plt.tight_layout()
    bar_path = figures_dir / "shap_global_importance.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    saved.append(bar_path)

    # 2. Dependence plots: area_sqm, dist_nearest_metro_m, is_offplan
    dep_specs = [
        ("area_sqm", "Unit Area (sqm)", "shap_dep_area.png"),
        ("dist_nearest_metro_m", "Distance to Nearest Metro (m)", "shap_dep_metro.png"),
        ("is_offplan", "Off-Plan Flag (0=Ready, 1=Off-Plan)", "shap_dep_offplan.png"),
    ]
    for col, xlabel, fname in dep_specs:
        if col not in feat_names:
            continue
        col_i = feat_names.index(col)
        x_vals = pd.to_numeric(X_sample[col], errors="coerce").to_numpy()
        y_shap = vals[:, col_i]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(x_vals, y_shap, alpha=0.35, s=12, color="#1D4ED8")
        ax.axhline(0.0, color="#6B7280", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SHAP contribution (log AED/sqm)")
        ax.set_title(f"SHAP Dependence: {xlabel}")
        plt.tight_layout()
        p = figures_dir / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved
