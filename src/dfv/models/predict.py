"""Shared single-property prediction, calibrated CQR interval, SHAP explanation, and verdict."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dfv.models.explain import explain_single_prediction
from dfv.models.registry import ModelBundle, prepare_inference_frame
from dfv.models.verdict import compute_verdict


def _market_fallback_defaults(bundle: ModelBundle) -> dict[str, Any]:
    vals = list(bundle.area_defaults.values())
    if not vals:
        return {
            "lat": 25.18,
            "lon": 55.26,
            "master_project": "Dubai",
            "nearest_metro": "Business Bay",
            "dist_nearest_metro_m": 1500.0,
            "dist_nearest_tram_m": 4000.0,
            "dist_nearest_mall_m": 3000.0,
            "dist_nearest_coast_m": 5000.0,
            "dist_airport_m": 18000.0,
            "dist_downtown_m": 12000.0,
            "poi_count_1km": 5.0,
            "bus_stop_density_1km": 2.0,
            "area_trailing_3m_ppsqm": float(np.exp(bundle.global_prior_log)),
            "area_trailing_6m_ppsqm": float(np.exp(bundle.global_prior_log)),
            "area_trailing_12m_ppsqm": float(np.exp(bundle.global_prior_log)),
            "area_trailing_3m_tx_count": 20.0,
            "area_trailing_12m_tx_count": 80.0,
            "area_trailing_12m_rent_ppsqm": 1000.0,
            "area_trailing_12m_yield": 0.06,
            "segment_counts": {},
            "area_tx_count": 0,
        }
    return {
        "lat": float(np.median([v["lat"] for v in vals])),
        "lon": float(np.median([v["lon"] for v in vals])),
        "master_project": "Unknown",
        "nearest_metro": "Unknown",
        "dist_nearest_metro_m": float(np.median([v["dist_nearest_metro_m"] for v in vals])),
        "dist_nearest_tram_m": float(np.median([v["dist_nearest_tram_m"] for v in vals])),
        "dist_nearest_mall_m": float(np.median([v["dist_nearest_mall_m"] for v in vals])),
        "dist_nearest_coast_m": float(np.median([v["dist_nearest_coast_m"] for v in vals])),
        "dist_airport_m": float(np.median([v["dist_airport_m"] for v in vals])),
        "dist_downtown_m": float(np.median([v["dist_downtown_m"] for v in vals])),
        "poi_count_1km": float(np.median([v["poi_count_1km"] for v in vals])),
        "bus_stop_density_1km": float(np.median([v["bus_stop_density_1km"] for v in vals])),
        "area_trailing_3m_ppsqm": float(np.median([v["area_trailing_3m_ppsqm"] for v in vals])),
        "area_trailing_6m_ppsqm": float(np.median([v["area_trailing_6m_ppsqm"] for v in vals])),
        "area_trailing_12m_ppsqm": float(np.median([v["area_trailing_12m_ppsqm"] for v in vals])),
        "area_trailing_3m_tx_count": 0.0,
        "area_trailing_12m_tx_count": 0.0,
        "area_trailing_12m_rent_ppsqm": float(
            np.median([v["area_trailing_12m_rent_ppsqm"] for v in vals])
        ),
        "area_trailing_12m_yield": float(np.median([v["area_trailing_12m_yield"] for v in vals])),
        "segment_counts": {},
        "area_tx_count": 0,
    }


def predict_single_property(
    payload: dict[str, Any],
    bundle: ModelBundle,
    explainer_or_model: Any | None = None,
) -> dict[str, Any]:
    """Run single-property fair-value prediction, CQR bounds, SHAP explanation, and verdict."""
    area_name = str(payload["area_name"]).strip()
    area_lookup = {k.lower(): k for k in bundle.area_defaults}
    canon_area = area_lookup.get(area_name.lower(), area_name)
    known_area = canon_area in bundle.area_defaults
    defaults = bundle.area_defaults[canon_area] if known_area else _market_fallback_defaults(bundle)

    prop_type = str(payload.get("property_type", "Unit"))
    sub_type = payload.get("property_sub_type") or ("Villa" if prop_type == "Villa" else "Flat")
    rooms = int(payload["rooms"])
    area_sqm = float(payload["area_sqm"])
    asking_price = float(payload["asking_price_aed"])
    is_offplan = 1 if bool(payload.get("is_offplan", False)) else 0
    has_parking = 1 if bool(payload.get("has_parking", True)) else 0
    mtc_raw = payload.get("months_to_completion")
    mtc = float(mtc_raw) if mtc_raw is not None else (24.0 if is_offplan else 0.0)
    dist_raw = payload.get("dist_nearest_metro_m")
    metro_dist = (
        float(dist_raw) if dist_raw is not None else float(defaults["dist_nearest_metro_m"])
    )

    data_through = str(bundle.metadata["training_window"]["test_end"])
    ref_dt = pd.Timestamp(data_through)

    area_12m = float(defaults["area_trailing_12m_ppsqm"])
    area_6m = float(defaults["area_trailing_6m_ppsqm"])
    area_3m = float(defaults["area_trailing_3m_ppsqm"])

    row_dict: dict[str, Any] = {
        "area_sqm": area_sqm,
        "log_area_sqm": float(np.log(max(area_sqm, 1.0))),
        "rooms": rooms,
        "has_parking": has_parking,
        "is_offplan": is_offplan,
        "months_to_completion": mtc,
        "project_age_years": 1.0 if is_offplan else 6.0,
        "completion_pct": 55.0 if is_offplan else 100.0,
        "delivery_delay_months": 0.0,
        "dist_nearest_metro_m": metro_dist,
        "dist_nearest_tram_m": float(defaults["dist_nearest_tram_m"]),
        "dist_nearest_mall_m": float(defaults["dist_nearest_mall_m"]),
        "dist_nearest_coast_m": float(defaults["dist_nearest_coast_m"]),
        "dist_airport_m": float(defaults["dist_airport_m"]),
        "dist_downtown_m": float(defaults["dist_downtown_m"]),
        "poi_count_1km": float(defaults["poi_count_1km"]),
        "bus_stop_density_1km": float(defaults["bus_stop_density_1km"]),
        "metro_within_500m": 1 if metro_dist < 500.0 else 0,
        "metro_within_1000m": 1 if metro_dist < 1000.0 else 0,
        "lat": float(defaults["lat"]),
        "lon": float(defaults["lon"]),
        "area_trailing_3m_ppsqm": area_3m,
        "area_trailing_6m_ppsqm": area_6m,
        "area_trailing_12m_ppsqm": area_12m,
        "log_area_trailing_12m_ppsqm": float(np.log(max(area_12m, 1.0))),
        "area_trailing_3m_tx_count": float(defaults["area_trailing_3m_tx_count"]),
        "area_trailing_12m_tx_count": float(defaults["area_trailing_12m_tx_count"]),
        "project_trailing_6m_ppsqm": area_6m,
        "project_trailing_12m_ppsqm": area_12m,
        "segment_trailing_12m_ppsqm": area_12m,
        "log_segment_trailing_12m_ppsqm": float(np.log(max(area_12m, 1.0))),
        "area_trailing_12m_rent_ppsqm": float(defaults["area_trailing_12m_rent_ppsqm"]),
        "area_trailing_12m_yield": float(defaults["area_trailing_12m_yield"]),
        "year": int(ref_dt.year),
        "quarter": int(ref_dt.quarter),
        "month": int(ref_dt.month),
        "days_since_start": int((ref_dt - pd.Timestamp("2015-01-01")).days),
        "property_type": prop_type,
        "property_sub_type": sub_type,
        "area_name": canon_area,
        "master_project": str(defaults.get("master_project", canon_area)),
        "project_name": str(payload.get("project_name") or "Unknown"),
        "developer_name": str(payload.get("developer_name") or "Unknown"),
        "nearest_metro": str(defaults.get("nearest_metro", "Unknown")),
    }

    X_row = prepare_inference_frame(pd.DataFrame([row_dict]), bundle)
    pred_log = np.asarray(bundle.point_model.predict(X_row), dtype=float)
    low_arr, med_arr, high_arr = bundle.interval_model.predict_aed_interval(
        X_row, np.array([area_sqm], dtype=float), pred_log
    )

    seg_key = f"{prop_type}|{rooms}"
    seg_counts = defaults.get("segment_counts", {})
    seg_tx_count = int(seg_counts.get(seg_key, 0)) if known_area else 0

    verdict_res = compute_verdict(
        asking_price=asking_price,
        predicted_median=float(med_arr[0]),
        low=float(low_arr[0]),
        high=float(high_arr[0]),
        segment_tx_count=seg_tx_count,
        min_segment_samples=5,
    )

    explainer = explainer_or_model or bundle.point_model
    drivers = explain_single_prediction(explainer, X_row, row_dict, area_sqm=area_sqm, top_k=5)

    return {
        "estimate": round(float(verdict_res.estimate), 2),
        "low": round(float(verdict_res.low), 2),
        "high": round(float(verdict_res.high), 2),
        "verdict": verdict_res.verdict,
        "gap_pct": round(float(verdict_res.gap_pct), 2),
        "confidence": verdict_res.confidence,
        "reason": verdict_res.reason,
        "drivers": [d.to_dict() for d in drivers],
        "model_version": str(bundle.metadata["model_version"]),
        "data_through": data_through,
    }
