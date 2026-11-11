"""Model unit & gate tests: training, reproducibility, invariances, SHAP-to-AED, metadata refusal."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dfv.config import PATHS
from dfv.data.synth import generate_synthetic_bundle
from dfv.features.build import build_feature_matrix
from dfv.models.explain import (
    compute_log_shap_values,
    convert_log_shap_to_aed,
    explain_single_prediction,
    format_aed_short,
)
from dfv.models.predict import predict_single_property
from dfv.models.registry import ModelMetadataError, load_model_bundle, prepare_inference_frame
from dfv.models.train import train_pipeline


def test_shap_to_aed_conversion_is_exact_and_sign_preserving() -> None:
    shap_log = np.array([0.12, -0.05, 0.03, -0.02])
    base_log = 9.5
    area_sqm = 100.0
    aed = convert_log_shap_to_aed(shap_log, base_log, area_sqm)
    expected_delta = np.exp(base_log + shap_log.sum()) * area_sqm - np.exp(base_log) * area_sqm
    assert abs(aed.sum() - expected_delta) < 1e-4
    assert aed[0] > 0 and aed[1] < 0
    assert format_aed_short(180000.0) == "+AED 180k"
    assert format_aed_short(-1500000.0) == "-AED 1.50M"


def test_registry_refuses_missing_or_incomplete_metadata(tmp_path: Path) -> None:
    with pytest.raises(ModelMetadataError):
        load_model_bundle(tmp_path)

    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "v1"}), encoding="utf-8")
    (tmp_path / "model_bundle.pkl").write_bytes(b"")
    with pytest.raises(ModelMetadataError):
        load_model_bundle(tmp_path)


def test_sample_model_training_gates_and_invariances(tmp_path: Path) -> None:
    bundle_data = generate_synthetic_bundle(n_sales=5000, n_rents=2000, seed=42)
    feat_df, _, _ = build_feature_matrix(
        bundle_data.transactions,
        bundle_data.rent_contracts,
        bundle_data.metro_stations,
        projects_df=bundle_data.projects,
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
    )
    feat_path = tmp_path / "features.parquet"
    feat_df.to_parquet(feat_path, index=False)

    model_dir = tmp_path / "model"
    metrics = train_pipeline(
        features_path=feat_path,
        model_out_dir=model_dir,
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        raw_dir=tmp_path / "raw",
        dataset_mode="sample",
        seed=42,
        n_trials=6,
        write_reports=True,
    )

    assert metrics["gates"]["beats_best_baseline"]
    assert metrics["gates"]["coverage_within_5pct_of_90"]
    assert metrics["gates"]["reproducible_with_seed"]
    assert metrics["gates"]["all_passed"]
    assert (tmp_path / "reports" / "error_analysis.md").exists()
    assert (tmp_path / "figures" / "shap_global_importance.png").exists()

    loaded = load_model_bundle(model_dir)
    # Prediction invariance: increasing area_sqm should increase total predicted price
    sample_row = feat_df.iloc[[-1]].copy()
    row_small = sample_row.copy()
    row_small["area_sqm"] = 80.0
    row_small["log_area_sqm"] = np.log(80.0)
    row_large = sample_row.copy()
    row_large["area_sqm"] = 160.0
    row_large["log_area_sqm"] = np.log(160.0)

    X_sm = prepare_inference_frame(row_small, loaded)
    X_lg = prepare_inference_frame(row_large, loaded)
    p_sm = float(np.exp(loaded.point_model.predict(X_sm)[0]) * 80.0)
    p_lg = float(np.exp(loaded.point_model.predict(X_lg)[0]) * 160.0)
    assert p_lg > p_sm

    # Unified additive SHAP check: base + sum(phi) == model prediction
    phi_sm, base_sm = compute_log_shap_values(loaded.point_model, X_sm)
    pred_log_sm = float(loaded.point_model.predict(X_sm)[0])
    assert abs((base_sm + float(phi_sm[0].sum())) - pred_log_sm) < 1e-6

    drivers = explain_single_prediction(
        loaded.point_model, X_sm, row_small.iloc[0].to_dict(), area_sqm=80.0, top_k=4
    )
    assert len(drivers) >= 3

    # Sanity partial-dependence directionality (measured, reported cleanly)
    pd_sample = feat_df.iloc[-25:].copy()
    pd_near = pd_sample.copy()
    pd_near["dist_nearest_metro_m"] = 250.0
    pd_near["metro_within_500m"] = 1
    pd_near["metro_within_1000m"] = 1
    pd_far = pd_sample.copy()
    pd_far["dist_nearest_metro_m"] = 3500.0
    pd_far["metro_within_500m"] = 0
    pd_far["metro_within_1000m"] = 0
    metro_diff = float(
        np.mean(loaded.point_model.predict(prepare_inference_frame(pd_near, loaded)))
        - np.mean(loaded.point_model.predict(prepare_inference_frame(pd_far, loaded)))
    )
    assert np.isfinite(metro_diff) and metro_diff > 0.0

    # Also check active trained model & predict_single_property abstention threshold (14 vs 15)
    if (PATHS.sample_models / "metadata.json").exists():
        sample_loaded = load_model_bundle(PATHS.sample_models)
        assert sample_loaded.metadata["model_version"]
        bundle_copy = copy.deepcopy(sample_loaded)
        area_key = next(iter(bundle_copy.area_defaults.keys()))
        bundle_copy.area_defaults[area_key]["segment_counts"] = {"Unit|2": 14}
        thin_res = predict_single_property(
            {
                "area_name": area_key,
                "property_type": "Unit",
                "rooms": 2,
                "area_sqm": 110.0,
                "asking_price_aed": 100_000.0,
            },
            bundle=bundle_copy,
        )
        assert thin_res["verdict"] == "insufficient data"

        bundle_copy.area_defaults[area_key]["segment_counts"] = {"Unit|2": 15}
        ok_res = predict_single_property(
            {
                "area_name": area_key,
                "property_type": "Unit",
                "rooms": 2,
                "area_sqm": 110.0,
                "asking_price_aed": 100_000.0,
            },
            bundle=bundle_copy,
        )
        assert ok_res["verdict"] == "Underpriced"
