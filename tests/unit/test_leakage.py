"""Leakage audit tests: temporal splits, future-deletion invariance, target exclusion, OOF encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfv.data.synth import SyntheticBundle
from dfv.features.build import (
    ALL_FEATURE_COLS,
    build_feature_matrix,
    compute_trailing_market_features,
)
from dfv.features.leakage import (
    assert_no_target_leakage_in_features,
    assert_oof_encoding_strict,
    assert_temporal_split_valid,
    assert_trailing_features_future_invariant,
)


def test_no_forbidden_target_columns_in_features() -> None:
    assert_no_target_leakage_in_features(ALL_FEATURE_COLS)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_target_leakage_in_features(ALL_FEATURE_COLS + ["price_per_sqm"])
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_target_leakage_in_features(ALL_FEATURE_COLS + ["instance_date"])


def test_trailing_features_invariant_when_future_deleted(synthetic_bundle: SyntheticBundle) -> None:
    tx = synthetic_bundle.transactions.sort_values("instance_date").reset_index(drop=True)
    rents = synthetic_bundle.rent_contracts
    trailing_cols = [
        "area_trailing_3m_ppsqm",
        "area_trailing_6m_ppsqm",
        "area_trailing_12m_ppsqm",
        "area_trailing_3m_tx_count",
        "area_trailing_12m_tx_count",
        "project_trailing_6m_ppsqm",
        "project_trailing_12m_ppsqm",
        "segment_trailing_12m_ppsqm",
        "area_trailing_12m_rent_ppsqm",
        "area_trailing_12m_yield",
    ]
    assert_trailing_features_future_invariant(
        tx,
        build_trailing_fn=lambda df: compute_trailing_market_features(df, rents),
        cutoff_date="2023-06-01",
        trailing_cols=trailing_cols,
    )


def test_full_feature_matrix_future_invariant(synthetic_bundle: SyntheticBundle) -> None:
    tx = synthetic_bundle.transactions.sort_values("instance_date").reset_index(drop=True)
    cutoff = "2024-01-01"

    full_feat, feat_cols, _ = build_feature_matrix(
        tx,
        synthetic_bundle.rent_contracts,
        synthetic_bundle.metro_stations,
        projects_df=synthetic_bundle.projects,
    )
    trunc_tx = tx.loc[pd.to_datetime(tx["instance_date"]) < pd.Timestamp(cutoff)].copy()
    trunc_feat, _, _ = build_feature_matrix(
        trunc_tx,
        synthetic_bundle.rent_contracts,
        synthetic_bundle.metro_stations,
        projects_df=synthetic_bundle.projects,
    )
    pre_full = full_feat.loc[pd.to_datetime(full_feat["instance_date"]) < pd.Timestamp(cutoff)]
    assert len(pre_full) == len(trunc_feat)
    for col in [
        "area_trailing_3m_ppsqm",
        "area_trailing_12m_ppsqm",
        "project_trailing_12m_ppsqm",
        "area_trailing_12m_yield",
    ]:
        assert np.allclose(
            pre_full[col].fillna(-1.0).to_numpy(),
            trunc_feat[col].fillna(-1.0).to_numpy(),
            rtol=1e-9,
            atol=1e-9,
        )


def test_temporal_split_validation_and_oof_check() -> None:
    train = pd.DataFrame(
        {"transaction_id": ["1", "2"], "instance_date": ["2022-01-01", "2023-01-01"]}
    )
    val = pd.DataFrame({"transaction_id": ["3"], "instance_date": ["2023-06-01"]})
    test = pd.DataFrame({"transaction_id": ["4"], "instance_date": ["2024-01-01"]})
    assert_temporal_split_valid(train, val, test)

    bad_val = pd.DataFrame({"transaction_id": ["5"], "instance_date": ["2022-06-01"]})
    with pytest.raises(AssertionError, match="Temporal overlap"):
        assert_temporal_split_valid(train, bad_val, test)

    df_oof = pd.DataFrame({"dev": ["A", "A", "B", "B"], "y": [10.0, 10.5, 9.5, 9.8]})
    assert_oof_encoding_strict(
        df_oof, "dev", "y", np.array([0, 1, 0, 1]), np.array([10.1, 10.0, 9.7, 9.6])
    )
