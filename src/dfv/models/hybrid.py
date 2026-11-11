"""Two-stage Ridge + Residual LightGBM estimator with exact additive SHAP decomposition."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ResidualBoostedLGBM:
    """Linear-detrended LightGBM: Ridge captures smooth temporal/size/area trends; LGBM fits L1 residuals.

    Why this exists (ADR-008): axis-aligned trees cannot extrapolate continuous time trends
    (`days_since_start`) beyond the last date seen in training, and on small sample datasets
    (<5k rows) pure trees overfit additive linear structure. Decomposing y = Ridge(X) + LGBM(X)
    beats Ridge and standalone LightGBM on both the 459k real DLD split and the 3.5k sample split,
    while preserving exact closed-form linear SHAP + TreeSHAP additivity.
    """

    def __init__(
        self,
        ridge_numeric_cols: list[str],
        ridge_cat_cols: list[str],
        lgb_params: dict[str, Any],
        ridge_alpha: float = 10.0,
    ) -> None:
        self.ridge_numeric_cols = list(ridge_numeric_cols)
        self.ridge_cat_cols = list(ridge_cat_cols)
        self.lgb_params = dict(lgb_params)
        self.ridge_alpha = float(ridge_alpha)

        self.ridge_pipe_: Pipeline | None = None
        self.lgb_model_: lgb.LGBMRegressor | None = None
        self.feature_names_in_: list[str] = []
        self._ridge_mean_z: np.ndarray | None = None
        self._ridge_base_val: float = 0.0
        self._ridge_col_groups: dict[str, list[int]] = {}
        self.best_iteration_: int | None = None

    def _fit_ridge_state(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        num_cols = [c for c in self.ridge_numeric_cols if c in X.columns]
        cat_cols = [c for c in self.ridge_cat_cols if c in X.columns]

        transformers: list[tuple[str, Any, list[str]]] = [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num_cols,
            )
        ]
        for col in cat_cols:
            transformers.append(
                (
                    f"cat_{col}",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                    [col],
                )
            )
        pre = ColumnTransformer(transformers=transformers, remainder="drop")
        pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=self.ridge_alpha))])
        ridge_df = X[num_cols + cat_cols].copy()
        for c in cat_cols:
            ridge_df[c] = ridge_df[c].astype(str)
        pipe.fit(ridge_df, y)

        Z = pre.transform(ridge_df)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        Z = np.asarray(Z, dtype=float)
        mean_z = np.mean(Z, axis=0)
        ridge = pipe.named_steps["ridge"]
        coefs = np.asarray(ridge.coef_, dtype=float)

        # Map transformed columns back to exact source feature names
        col_groups: dict[str, list[int]] = {}
        for i, col in enumerate(num_cols):
            col_groups[col] = [i]

        offset = len(num_cols)
        for col in cat_cols:
            ohe: OneHotEncoder = pre.named_transformers_[f"cat_{col}"]
            n_out = len(ohe.get_feature_names_out())
            col_groups[col] = list(range(offset, offset + n_out))
            offset += n_out

        self.ridge_pipe_ = pipe
        self._ridge_mean_z = mean_z
        self._ridge_base_val = float(ridge.intercept_ + float(np.dot(mean_z, coefs)))
        self._ridge_col_groups = col_groups
        return np.asarray(pipe.predict(ridge_df), dtype=float)

    def predict_linear(self, X: pd.DataFrame) -> np.ndarray:
        """Return strictly the smooth Ridge trend/baseline component."""
        if self.ridge_pipe_ is None:
            raise RuntimeError("ResidualBoostedLGBM must be fitted before calling predict_linear.")
        num_cols = [c for c in self.ridge_numeric_cols if c in X.columns]
        cat_cols = [c for c in self.ridge_cat_cols if c in X.columns]
        ridge_df = X[num_cols + cat_cols].copy()
        for c in cat_cols:
            ridge_df[c] = ridge_df[c].astype(str)
        return np.asarray(self.ridge_pipe_.predict(ridge_df), dtype=float)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_X: pd.DataFrame | None = None,
        eval_y: np.ndarray | None = None,
        callbacks: list[Any] | None = None,
    ) -> ResidualBoostedLGBM:
        """Fit Stage-1 Ridge baseline and Stage-2 LightGBM on log residuals."""
        self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype=float)
        y_lin_tr = self._fit_ridge_state(X, y_arr)
        res_tr = y_arr - y_lin_tr

        params = dict(self.lgb_params)
        self.lgb_model_ = lgb.LGBMRegressor(**params)
        if eval_X is not None and eval_y is not None:
            y_lin_val = self.predict_linear(eval_X)
            res_val = np.asarray(eval_y, dtype=float) - y_lin_val
            self.lgb_model_.fit(
                X,
                res_tr,
                eval_X=eval_X,
                eval_y=res_val,
                callbacks=callbacks,
            )
            self.best_iteration_ = getattr(self.lgb_model_, "best_iteration_", None)
        else:
            self.lgb_model_.fit(X, res_tr)
            self.best_iteration_ = params.get("n_estimators")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return Ridge trend + LightGBM residual prediction in log(price/sqm)."""
        if self.lgb_model_ is None:
            raise RuntimeError("ResidualBoostedLGBM must be fitted before predict.")
        lin_pred = self.predict_linear(X)
        res_pred = np.asarray(self.lgb_model_.predict(X), dtype=float)
        return lin_pred + res_pred

    def compute_shap(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Return exact additive SHAP values (n_rows x n_features) and scalar base value in log space."""
        if self.ridge_pipe_ is None or self.lgb_model_ is None or self._ridge_mean_z is None:
            raise RuntimeError("ResidualBoostedLGBM must be fitted before compute_shap.")

        # 1. Exact Ridge SHAP grouped by input feature
        num_cols = [c for c in self.ridge_numeric_cols if c in X.columns]
        cat_cols = [c for c in self.ridge_cat_cols if c in X.columns]
        ridge_df = X[num_cols + cat_cols].copy()
        for c in cat_cols:
            ridge_df[c] = ridge_df[c].astype(str)

        pre = self.ridge_pipe_.named_steps["pre"]
        ridge = self.ridge_pipe_.named_steps["ridge"]
        Z = pre.transform(ridge_df)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        Z = np.asarray(Z, dtype=float)
        coefs = np.asarray(ridge.coef_, dtype=float)
        weighted_diff = (Z - self._ridge_mean_z) * coefs

        feat_names = list(X.columns)
        col_to_idx = {c: i for i, c in enumerate(feat_names)}
        phi_ridge = np.zeros((len(X), len(feat_names)), dtype=float)
        for col, idxs in self._ridge_col_groups.items():
            if col in col_to_idx and idxs:
                phi_ridge[:, col_to_idx[col]] = np.sum(weighted_diff[:, idxs], axis=1)

        # 2. Exact TreeSHAP on LightGBM residuals
        tree_exp = shap.TreeExplainer(self.lgb_model_)
        sv = tree_exp(X)
        phi_lgb = np.asarray(sv.values, dtype=float)
        base_lgb = float(np.asarray(sv.base_values).reshape(-1)[0])

        return phi_ridge + phi_lgb, self._ridge_base_val + base_lgb
