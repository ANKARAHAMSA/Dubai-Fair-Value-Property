"""Baseline models for DubaiFairValue evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BaselineMetrics:
    """Evaluation metrics for a regression model on total AED and log(price/sqm)."""

    name: str
    mdape: float
    mape: float
    rmse_log: float
    r2_log: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mdape": round(float(self.mdape), 5),
            "mape": round(float(self.mape), 5),
            "rmse_log": round(float(self.rmse_log), 5),
            "r2_log": round(float(self.r2_log), 5),
        }


def compute_regression_metrics(
    name: str,
    y_true_total: np.ndarray,
    y_pred_total: np.ndarray,
    y_true_log_ppsqm: np.ndarray,
    y_pred_log_ppsqm: np.ndarray,
) -> BaselineMetrics:
    """Compute MdAPE, MAPE (in total AED), RMSE (log ppsqm), and R2 (log ppsqm)."""
    ape = np.abs(y_true_total - y_pred_total) / np.maximum(y_true_total, 1.0)
    mdape = float(np.median(ape))
    mape = float(np.mean(ape))
    err_log = y_true_log_ppsqm - y_pred_log_ppsqm
    rmse_log = float(np.sqrt(np.mean(err_log**2)))
    ss_res = float(np.sum(err_log**2))
    ss_tot = float(np.sum((y_true_log_ppsqm - np.mean(y_true_log_ppsqm)) ** 2))
    r2_log = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return BaselineMetrics(name=name, mdape=mdape, mape=mape, rmse_log=rmse_log, r2_log=r2_log)


def evaluate_global_median_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> BaselineMetrics:
    """Baseline (a): global train median price/sqm * area_sqm."""
    global_med_ppsqm = float(train_df["price_per_sqm"].median())
    pred_log_ppsqm = np.full(len(test_df), np.log(global_med_ppsqm), dtype=float)
    pred_total = global_med_ppsqm * test_df["area_sqm"].to_numpy(dtype=float)
    return compute_regression_metrics(
        "baseline_global_median",
        test_df["actual_worth"].to_numpy(dtype=float),
        pred_total,
        test_df["log_price_per_sqm"].to_numpy(dtype=float),
        pred_log_ppsqm,
    )


def evaluate_trailing_segment_baseline(test_df: pd.DataFrame) -> BaselineMetrics:
    """Baseline (b): strictly-prior trailing-12m median by area x property_type x rooms * area_sqm."""
    pred_ppsqm = test_df["segment_trailing_12m_ppsqm"].to_numpy(dtype=float)
    pred_log_ppsqm = np.log(np.maximum(pred_ppsqm, 1.0))
    pred_total = pred_ppsqm * test_df["area_sqm"].to_numpy(dtype=float)
    return compute_regression_metrics(
        "baseline_trailing_12m_segment",
        test_df["actual_worth"].to_numpy(dtype=float),
        pred_total,
        test_df["log_price_per_sqm"].to_numpy(dtype=float),
        pred_log_ppsqm,
    )


def fit_and_evaluate_ridge_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[Pipeline, BaselineMetrics]:
    """Baseline (c): Ridge regression on log(price_per_sqm)."""
    # Limit high-cardinality project/developer one-hot columns cleanly to avoid memory blowup
    low_card_cats = [
        c for c in categorical_cols if c in {"property_type", "property_sub_type", "area_name"}
    ]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                low_card_cats,
            ),
        ]
    )
    pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=5.0))])
    pipe.fit(
        train_df[numeric_cols + low_card_cats], train_df["log_price_per_sqm"].to_numpy(dtype=float)
    )

    pred_log_ppsqm = pipe.predict(test_df[numeric_cols + low_card_cats])
    pred_total = np.exp(pred_log_ppsqm) * test_df["area_sqm"].to_numpy(dtype=float)
    metrics = compute_regression_metrics(
        "baseline_ridge_log",
        test_df["actual_worth"].to_numpy(dtype=float),
        pred_total,
        test_df["log_price_per_sqm"].to_numpy(dtype=float),
        pred_log_ppsqm,
    )
    return pipe, metrics
