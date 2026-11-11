"""Unit tests for the synthetic DLD-like data generator."""

from __future__ import annotations

import numpy as np

from dfv.data.synth import (
    SyntheticBundle,
    SyntheticGroundTruth,
    generate_synthetic_bundle,
    ground_truth_summary,
)


def test_synthetic_bundle_shapes_and_columns(synthetic_bundle: SyntheticBundle) -> None:
    assert len(synthetic_bundle.transactions) == 1200
    assert len(synthetic_bundle.rent_contracts) == 1500
    assert len(synthetic_bundle.projects) == 75
    assert len(synthetic_bundle.metro_stations) >= 10
    assert len(synthetic_bundle.sale_index) > 100

    tx = synthetic_bundle.transactions
    assert (tx["area_sqm"] > 0).all()
    assert (tx["actual_worth"] > 0).all()
    assert (tx["price_per_sqm"] > 0).all()
    assert set(tx["is_offplan"].unique()).issubset({0, 1})
    assert tx["data_source"].eq("synthetic").all()


def test_synthetic_bundle_is_reproducible() -> None:
    b1 = generate_synthetic_bundle(n_sales=300, n_rents=300, seed=123)
    b2 = generate_synthetic_bundle(n_sales=300, n_rents=300, seed=123)
    assert b1.transactions.equals(b2.transactions)
    assert b1.rent_contracts.equals(b2.rent_contracts)


def test_synthetic_area_ordering_realism(synthetic_bundle: SyntheticBundle) -> None:
    tx = synthetic_bundle.transactions
    med = tx.groupby("area_name")["price_per_sqm"].median()
    assert med["Downtown Dubai"] > med["Dubai South"]
    assert med["Palm Jumeirah"] > med["Discovery Gardens"]


def test_synthetic_dirty_rows_injection(dirty_synthetic_bundle: SyntheticBundle) -> None:
    tx = dirty_synthetic_bundle.transactions
    assert len(tx) == 626
    assert (tx["procedure_name"] != "Sales").sum() == 5
    assert (tx["property_usage"] != "Residential").sum() == 5
    assert (tx["area_sqm"] <= 0).sum() == 3


def test_synthetic_planted_effects_directionality() -> None:
    truth = SyntheticGroundTruth(
        metro_lt500_log_effect=0.20,
        offplan_log_effect=0.10,
        noise_std=0.04,
    )
    bundle = generate_synthetic_bundle(n_sales=3000, n_rents=1000, seed=7, truth=truth)
    tx = bundle.transactions
    # Within Downtown Dubai, <500m from metro should have higher log price/sqm than >2000m
    dt = tx[tx["area_name"] == "Downtown Dubai"]
    near = np.log(dt[dt["metro_distance_m"] < 500]["price_per_sqm"]).mean()
    far = np.log(dt[dt["metro_distance_m"] >= 2000]["price_per_sqm"]).mean()
    assert near > far
    summary = ground_truth_summary(truth)
    assert summary["metro_lt500_log_effect"] == 0.20
