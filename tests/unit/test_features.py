"""Unit tests for time-aware feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfv.data.synth import SyntheticBundle
from dfv.features.build import (
    ALL_FEATURE_COLS,
    build_feature_matrix,
    compute_strictly_prior_trailing_stats,
)


def test_strictly_prior_trailing_stats_excludes_same_day_and_future() -> None:
    df = pd.DataFrame(
        {
            "area_name": ["A", "A", "A", "A"],
            "instance_date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-10"],
            "price_per_sqm": [10000.0, 20000.0, 30000.0, 50000.0],
        }
    )
    stats = compute_strictly_prior_trailing_stats(
        df,
        group_cols=["area_name"],
        date_col="instance_date",
        value_col="price_per_sqm",
        windows_days=(30,),
        min_obs=1,
        compute_counts=True,
    )
    assert np.isnan(stats.loc[0, "med_30"])
    assert stats.loc[0, "cnt_30"] == 0.0
    # Both rows on 2024-01-02 only see 2024-01-01 (10000.0)
    assert stats.loc[1, "med_30"] == 10000.0
    assert stats.loc[2, "med_30"] == 10000.0
    assert stats.loc[1, "cnt_30"] == 1.0
    # 2024-01-10 sees [10000, 20000, 30000] -> median 20000
    assert stats.loc[3, "med_30"] == 20000.0
    assert stats.loc[3, "cnt_30"] == 3.0


def test_build_feature_matrix_end_to_end(synthetic_bundle: SyntheticBundle) -> None:
    feat_df, feat_cols, geo_stats = build_feature_matrix(
        synthetic_bundle.transactions,
        synthetic_bundle.rent_contracts,
        synthetic_bundle.metro_stations,
        projects_df=synthetic_bundle.projects,
    )
    assert feat_cols == ALL_FEATURE_COLS
    assert len(feat_df) == len(synthetic_bundle.transactions)
    assert feat_df["transaction_id"].is_unique
    for col in feat_cols:
        assert col in feat_df.columns
    assert "log_price_per_sqm" in feat_df.columns
    assert geo_stats["total_rows"] == len(feat_df)
