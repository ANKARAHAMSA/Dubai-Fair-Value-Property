"""Optuna hyperparameter tuning for DubaiFairValue gradient-boosted models."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from dfv.models.hybrid import ResidualBoostedLGBM

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("dfv.models.tune")


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    area_val: np.ndarray,
    y_true_val_total: np.ndarray,
    ridge_numeric_cols: list[str] | None = None,
    ridge_cat_cols: list[str] | None = None,
    use_residual_hybrid: bool = True,
    seed: int = 42,
    n_trials: int = 12,
    max_tune_rows: int = 120000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run bounded seeded Optuna search minimizing validation MdAPE."""
    if len(X_train) > max_tune_rows:
        X_tr_sub = X_train.iloc[-max_tune_rows:].copy()
        y_tr_sub = y_train[-max_tune_rows:]
    else:
        X_tr_sub = X_train
        y_tr_sub = y_train

    small_data = len(X_tr_sub) < 20_000
    trial_logs: list[dict[str, Any]] = []
    num_cols = ridge_numeric_cols or [
        c for c in X_train.columns if not str(X_train[c].dtype) == "category"
    ]
    cat_cols = ridge_cat_cols or [
        c for c in ("property_type", "property_sub_type", "area_name") if c in X_train.columns
    ]

    def objective(trial: optuna.Trial) -> float:
        if small_data:
            lgb_params: dict[str, Any] = {
                "objective": "regression_l1",
                "n_estimators": trial.suggest_int("n_estimators", 150, 260),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.055, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 5),
                "num_leaves": trial.suggest_int("num_leaves", 12, 24),
                "min_child_samples": trial.suggest_int("min_child_samples", 12, 28),
                "subsample": trial.suggest_float("subsample", 0.75, 0.95),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.15, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
                "random_state": seed,
                "deterministic": True,
                "force_col_wise": True,
                "n_jobs": 4,
                "verbose": -1,
            }
        else:
            lgb_params = {
                "objective": "regression_l1",
                "n_estimators": trial.suggest_int("n_estimators", 200, 650),
                "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.10, log=True),
                "max_depth": -1,
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 90),
                "subsample": trial.suggest_float("subsample", 0.70, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                "random_state": seed,
                "deterministic": True,
                "force_col_wise": True,
                "n_jobs": 4,
                "verbose": -1,
            }

        ridge_alpha = (
            trial.suggest_float("ridge_alpha", 3.0, 10.0, log=True) if use_residual_hybrid else 10.0
        )

        es_rounds = 50 if small_data else 35
        if use_residual_hybrid:
            model: Any = ResidualBoostedLGBM(
                ridge_numeric_cols=num_cols,
                ridge_cat_cols=cat_cols,
                lgb_params=lgb_params,
                ridge_alpha=ridge_alpha,
            )
            model.fit(
                X_tr_sub,
                y_tr_sub,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(stopping_rounds=es_rounds, verbose=False)],
            )
        else:
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_tr_sub,
                y_tr_sub,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(stopping_rounds=es_rounds, verbose=False)],
            )

        pred_log = model.predict(X_val)
        pred_total = np.exp(pred_log) * area_val
        mdape = float(
            np.median(np.abs(y_true_val_total - pred_total) / np.maximum(y_true_val_total, 1.0))
        )
        best_iter = getattr(model, "best_iteration_", None) or lgb_params["n_estimators"]
        if small_data:
            best_iter = max(int(best_iter), 140)
        trial.set_user_attr("best_iteration", int(best_iter))
        trial_logs.append({"trial": trial.number, "val_mdape": round(mdape, 5)})
        return mdape

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_iter = study.best_trial.user_attrs.get("best_iteration")
    if best_iter and best_iter > 20:
        best_params["n_estimators"] = int(best_iter)
    logger.info("Optuna best val MdAPE=%.4f with params=%s", study.best_value, best_params)
    return best_params, trial_logs
