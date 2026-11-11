"""Leakage audit utilities used by build.py and automated leakage tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

FORBIDDEN_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "actual_worth",
        "price_per_sqm",
        "price_per_sqft",
        "log_price_per_sqm",
        "instance_date",
        "transaction_id",
        "flag_outlier",
        "flag_non_market",
        "flag_duplicate",
        "embedded_rent_value",
        "embedded_rent_psf",
    }
)


def assert_no_target_leakage_in_features(feature_cols: list[str]) -> None:
    """Assert that no target, price derivative, flag, or raw date column enters model features."""
    bad = set(feature_cols).intersection(FORBIDDEN_FEATURE_COLS)
    if bad:
        raise AssertionError(f"Target/date leakage columns detected in features: {sorted(bad)}")


def assert_temporal_split_valid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "instance_date",
    id_col: str = "transaction_id",
    cal_df: pd.DataFrame | None = None,
) -> None:
    """Assert strict chronological ordering and disjoint row IDs across train/val/(cal)/test."""
    tr_dt = pd.to_datetime(train_df[date_col])
    va_dt = pd.to_datetime(val_df[date_col])
    te_dt = pd.to_datetime(test_df[date_col])

    if not (tr_dt.max() < va_dt.min()):
        raise AssertionError(f"Temporal overlap train->val: {tr_dt.max()} >= {va_dt.min()}")
    if cal_df is not None:
        cal_dt = pd.to_datetime(cal_df[date_col])
        if not (va_dt.max() < cal_dt.min()):
            raise AssertionError(f"Temporal overlap val->cal: {va_dt.max()} >= {cal_dt.min()}")
        if not (cal_dt.max() < te_dt.min()):
            raise AssertionError(f"Temporal overlap cal->test: {cal_dt.max()} >= {te_dt.min()}")
    elif not (va_dt.max() < te_dt.min()):
        raise AssertionError(f"Temporal overlap val->test: {va_dt.max()} >= {te_dt.min()}")

    if id_col in train_df.columns:
        s_tr = set(train_df[id_col])
        s_va = set(val_df[id_col])
        s_te = set(test_df[id_col])
        if s_tr & s_va or s_va & s_te or s_tr & s_te:
            raise AssertionError("Row ID overlap detected between train/val/test splits.")
        if cal_df is not None and id_col in cal_df.columns:
            s_cal = set(cal_df[id_col])
            if s_cal & s_tr or s_cal & s_va or s_cal & s_te:
                raise AssertionError("Row ID overlap detected involving calibration split.")


def assert_trailing_features_future_invariant(
    df: pd.DataFrame,
    build_trailing_fn: Callable[[pd.DataFrame], pd.DataFrame],
    cutoff_date: str,
    trailing_cols: list[str],
    date_col: str = "instance_date",
) -> None:
    """Assert trailing features at rows < cutoff_date are identical when rows >= cutoff_date are deleted."""
    full_out = build_trailing_fn(df)
    dt_full = pd.to_datetime(full_out[date_col])
    pre_mask = dt_full < pd.Timestamp(cutoff_date)

    df_truncated = df.loc[pd.to_datetime(df[date_col]) < pd.Timestamp(cutoff_date)].copy()
    trunc_out = build_trailing_fn(df_truncated)

    left = full_out.loc[pre_mask, trailing_cols].reset_index(drop=True)
    right = trunc_out[trailing_cols].reset_index(drop=True)

    for col in trailing_cols:
        l_vals = left[col].to_numpy(dtype=float)
        r_vals = right[col].to_numpy(dtype=float)
        both_nan = np.isnan(l_vals) & np.isnan(r_vals)
        close = np.isclose(np.nan_to_num(l_vals), np.nan_to_num(r_vals), rtol=1e-9, atol=1e-9)
        if not np.all(both_nan | close):
            raise AssertionError(f"Future leakage detected in trailing feature '{col}'!")


def assert_oof_encoding_strict(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    fold_ids: np.ndarray,
    encoded_values: np.ndarray,
    global_prior_weight: float = 20.0,
) -> None:
    """Verify out-of-fold target encoding at row i matches ONLY rows with fold != fold_ids[i]."""
    if len(df) == 0:
        return
    col_vals = df[group_col].astype(str).to_numpy()
    target_vals = df[target_col].to_numpy(dtype=float)
    sample_indices = np.linspace(0, len(df) - 1, min(200, len(df)), dtype=int)

    for idx in sample_indices:
        if not np.isfinite(encoded_values[idx]):
            raise AssertionError(f"Non-finite OOF encoding at idx={idx}")
        f = fold_ids[idx]
        in_fold_mask = fold_ids != f
        fold_global_mean = float(target_vals[in_fold_mask].mean())
        group_mask = in_fold_mask & (col_vals == col_vals[idx])
        other_vals = target_vals[group_mask]
        n_other = len(other_vals)
        expected = (
            float(
                (other_vals.sum() + global_prior_weight * fold_global_mean)
                / (n_other + global_prior_weight)
            )
            if n_other > 0
            else fold_global_mean
        )
        if not np.isclose(float(encoded_values[idx]), expected, rtol=1e-6, atol=1e-6):
            raise AssertionError(
                f"OOF encoding mismatch/leakage at idx={idx}: got {encoded_values[idx]:.8f}, expected {expected:.8f}"
            )
