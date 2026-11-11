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


def test_temporal_split_validation_and_oof_check(synthetic_bundle: SyntheticBundle) -> None:
    from dfv.models.train import compute_oof_target_encoding

    train = pd.DataFrame(
        {"transaction_id": ["1", "2"], "instance_date": ["2022-01-01", "2023-01-01"]}
    )
    val = pd.DataFrame({"transaction_id": ["3"], "instance_date": ["2023-06-01"]})
    cal = pd.DataFrame({"transaction_id": ["4"], "instance_date": ["2023-09-01"]})
    test = pd.DataFrame({"transaction_id": ["5"], "instance_date": ["2024-01-01"]})
    assert_temporal_split_valid(train, val, test, cal_df=cal)

    bad_val = pd.DataFrame({"transaction_id": ["6"], "instance_date": ["2022-06-01"]})
    with pytest.raises(AssertionError, match="Temporal overlap"):
        assert_temporal_split_valid(train, bad_val, test)

    bad_cal = pd.DataFrame({"transaction_id": ["7"], "instance_date": ["2023-06-01"]})
    with pytest.raises(AssertionError, match="val->cal"):
        assert_temporal_split_valid(train, val, test, cal_df=bad_cal)

    # Real OOF target encoding must match fold-excluded prior formula and reject leaked values
    df_oof = pd.DataFrame(
        {"dev": ["A", "A", "B", "B", "A", "B"], "y": [10.0, 10.5, 9.5, 9.8, 10.2, 9.4]}
    )
    tr_enc, [ev_enc], _, _ = compute_oof_target_encoding(
        df_oof, [df_oof.iloc[:2].copy()], ["dev"], target_col="y", prior_weight=20.0, n_splits=2
    )
    assert "dev_oof_enc" in tr_enc.columns and "dev_oof_enc" in ev_enc.columns

    # Perturbing row 0 to include in-fold target leakage must trigger AssertionError
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    fids = np.zeros(len(df_oof), dtype=int)
    for f_idx, (_, vi) in enumerate(kf.split(df_oof)):
        fids[vi] = f_idx
    assert_oof_encoding_strict(
        tr_enc, "dev", "y", fids, tr_enc["dev_oof_enc"].to_numpy(), global_prior_weight=20.0
    )
    leaked_vals = tr_enc["dev_oof_enc"].to_numpy().copy()
    leaked_vals[0] += 0.25
    with pytest.raises(AssertionError, match="OOF encoding mismatch/leakage"):
        assert_oof_encoding_strict(df_oof, "dev", "y", fids, leaked_vals, global_prior_weight=20.0)

    # Historical off-plan rows must never leak future completion_pct==100 or unopened Route 2020 stations
    feat_df, _, _ = build_feature_matrix(
        synthetic_bundle.transactions,
        synthetic_bundle.rent_contracts,
        synthetic_bundle.metro_stations,
        projects_df=synthetic_bundle.projects,
    )
    pre20_off = feat_df[(feat_df["instance_date"] < "2020-01-01") & (feat_df["is_offplan"] == 1)]
    assert (pre20_off["completion_pct"] < 100.0).all()
    r2020_names = set(
        synthetic_bundle.metro_stations.loc[
            synthetic_bundle.metro_stations["is_route_2020"] == True, "station_name"  # noqa: E712
        ]
    )
    pre21 = feat_df[feat_df["instance_date"] < "2021-01-01"]
    assert not pre21["nearest_metro"].isin(r2020_names).any()
