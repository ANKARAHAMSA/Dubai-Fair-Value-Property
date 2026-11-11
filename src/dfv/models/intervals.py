"""Quantile LightGBM + Split-Conformalized Quantile Regression (CQR) intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class ConformalIntervalModel:
    """Calibrated quantile LightGBM wrapper producing nominal 90% prediction intervals."""

    q_low_model: lgb.LGBMRegressor
    q_high_model: lgb.LGBMRegressor
    conformal_q: float
    alpha: float = 0.10
    nominal_coverage: float = 0.90
    trend_ridge: Any | None = None

    def _trend_offset(self, X: pd.DataFrame) -> np.ndarray:
        if self.trend_ridge is None:
            return np.zeros(len(X), dtype=float)
        return np.asarray(self.trend_ridge.predict_linear(X), dtype=float)

    def predict_log_bounds(
        self, X: pd.DataFrame, point_pred_log: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated (low_log, high_log) guaranteed to satisfy low <= point <= high."""
        offset = self._trend_offset(X)
        raw_low = offset + np.asarray(self.q_low_model.predict(X), dtype=float)
        raw_high = offset + np.asarray(self.q_high_model.predict(X), dtype=float)
        # Enforce quantile ordering before applying conformal correction
        q_lo = np.minimum(raw_low, raw_high)
        q_hi = np.maximum(raw_low, raw_high)

        cal_low = q_lo - self.conformal_q
        cal_high = q_hi + self.conformal_q

        if point_pred_log is not None:
            cal_low = np.minimum(cal_low, point_pred_log)
            cal_high = np.maximum(cal_high, point_pred_log)
        return cal_low, cal_high

    def predict_aed_interval(
        self,
        X: pd.DataFrame,
        area_sqm: np.ndarray,
        point_pred_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert calibrated log bounds into AED total prices (low, median, high)."""
        cal_low_log, cal_high_log = self.predict_log_bounds(X, point_pred_log=point_pred_log)
        med = np.exp(point_pred_log) * area_sqm
        low = np.minimum(np.exp(cal_low_log) * area_sqm, med)
        high = np.maximum(np.exp(cal_high_log) * area_sqm, med)
        return low, med, high


def fit_conformal_quantile_intervals(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    best_params: dict[str, Any],
    point_model: Any | None = None,
    seed: int = 42,
    alpha: float = 0.10,
) -> ConformalIntervalModel:
    """Fit 5th/95th quantile LightGBM models (detrended if point_model has linear base) + CQR."""
    base_params: dict[str, Any] = {
        "n_estimators": int(best_params.get("n_estimators", 250)),
        "learning_rate": float(best_params.get("learning_rate", 0.05)),
        "max_depth": int(best_params.get("max_depth", -1)),
        "num_leaves": int(best_params.get("num_leaves", 31)),
        "min_child_samples": int(best_params.get("min_child_samples", 25)),
        "subsample": float(best_params.get("subsample", 0.85)),
        "colsample_bytree": float(best_params.get("colsample_bytree", 0.85)),
        "reg_alpha": float(best_params.get("reg_alpha", 0.05)),
        "reg_lambda": float(best_params.get("reg_lambda", 1.0)),
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 4,
        "verbose": -1,
    }

    has_linear = point_model is not None and hasattr(point_model, "predict_linear")
    if point_model is not None and has_linear:
        offset_tr = np.asarray(point_model.predict_linear(X_train), dtype=float)
        offset_cal = np.asarray(point_model.predict_linear(X_cal), dtype=float)
    else:
        offset_tr = np.zeros(len(X_train), dtype=float)
        offset_cal = np.zeros(len(X_cal), dtype=float)

    y_tr_target = np.asarray(y_train, dtype=float) - offset_tr

    q_low = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2.0, **base_params)
    q_high = lgb.LGBMRegressor(objective="quantile", alpha=1.0 - (alpha / 2.0), **base_params)

    q_low.fit(X_train, y_tr_target)
    q_high.fit(X_train, y_tr_target)

    # Split-conformal CQR nonconformity score: max(q_low - y, y - q_high)
    cal_lo = offset_cal + q_low.predict(X_cal)
    cal_hi = offset_cal + q_high.predict(X_cal)
    lo_ordered = np.minimum(cal_lo, cal_hi)
    hi_ordered = np.maximum(cal_lo, cal_hi)

    scores = np.maximum(lo_ordered - y_cal, y_cal - hi_ordered)
    n = len(scores)
    q_level = min(1.0, np.ceil((1.0 - alpha) * (n + 1)) / max(n, 1))
    conformal_q = float(np.quantile(scores, q_level, method="higher"))

    return ConformalIntervalModel(
        q_low_model=q_low,
        q_high_model=q_high,
        conformal_q=conformal_q,
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        trend_ridge=point_model if has_linear else None,
    )


def evaluate_interval_coverage(
    y_true_total: np.ndarray,
    low: np.ndarray,
    med: np.ndarray,
    high: np.ndarray,
) -> dict[str, float]:
    """Compute empirical interval coverage and median relative width."""
    covered = (y_true_total >= low) & (y_true_total <= high)
    rel_width = (high - low) / np.maximum(med, 1.0)
    return {
        "coverage": round(float(np.mean(covered)), 4),
        "median_rel_width": round(float(np.median(rel_width)), 4),
        "median_abs_width_aed": round(float(np.median(high - low)), 2),
    }
