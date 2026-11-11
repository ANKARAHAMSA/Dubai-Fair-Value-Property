"""Shared econometric helpers (FWL one-way / two-way fixed effects + HC1 OLS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class HedonicTermResult:
    """Estimated fixed-effects hedonic coefficient with exact % conversion and 95% CI."""

    term: str
    log_coef: float
    se_robust: float
    pct_effect: float
    ci_low_pct: float
    ci_high_pct: float
    p_value: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "log_coef": round(self.log_coef, 5),
            "se_robust": round(self.se_robust, 5),
            "pct_effect": round(self.pct_effect, 2),
            "ci_low_pct": round(self.ci_low_pct, 2),
            "ci_high_pct": round(self.ci_high_pct, 2),
            "p_value": round(self.p_value, 6),
            "n_obs": self.n_obs,
        }


def _independent_column_indices(X_mat: np.ndarray, tol: float = 1e-6) -> list[int]:
    """Return linearly independent column indices in left-to-right order (modified Gram-Schmidt)."""
    basis: list[np.ndarray] = []
    keep: list[int] = []
    for j in range(X_mat.shape[1]):
        v = X_mat[:, j].astype(float, copy=True)
        norm0 = float(np.linalg.norm(v))
        if norm0 <= 1e-10:
            continue
        for b in basis:
            v -= float(np.dot(b, v)) * b
        rem = float(np.linalg.norm(v))
        if rem > tol * norm0:
            basis.append(v / rem)
            keep.append(j)
    return keep


def _solve_demeaned_hc1(
    y_dem: np.ndarray,
    X_dem_mat: np.ndarray,
    col_names: list[str],
    n_fe: int,
) -> tuple[dict[str, HedonicTermResult], float, int]:
    """Prune collinear columns and run HC1 OLS on demeaned arrays."""
    keep = _independent_column_indices(X_dem_mat)
    if not keep:
        return {}, 0.0, int(len(y_dem))

    active_cols = [col_names[i] for i in keep]
    X_use = X_dem_mat[:, keep]
    n_obs = int(len(y_dem))
    k = len(active_cols)

    fit = sm.OLS(y_dem, X_use).fit(cov_type="HC1")
    dof_scale = float(np.sqrt(max(n_obs - k, 1) / max(n_obs - k - n_fe, 1)))

    results: dict[str, HedonicTermResult] = {}
    params = np.asarray(fit.params, dtype=float)
    bse = np.asarray(fit.bse, dtype=float) * dof_scale

    for i, col in enumerate(active_cols):
        coef = float(params[i])
        se = max(float(bse[i]), 1e-9)
        if not np.isfinite(coef) or not np.isfinite(se) or se > 10.0:
            continue
        t_stat = coef / se
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        ci_lo = float(np.clip(coef - 1.96 * se, -5.0, 5.0))
        ci_hi = float(np.clip(coef + 1.96 * se, -5.0, 5.0))
        results[col] = HedonicTermResult(
            term=col,
            log_coef=float(np.clip(coef, -5.0, 5.0)),
            se_robust=se,
            pct_effect=float((np.exp(np.clip(coef, -5.0, 5.0)) - 1.0) * 100.0),
            ci_low_pct=float((np.exp(ci_lo) - 1.0) * 100.0),
            ci_high_pct=float((np.exp(ci_hi) - 1.0) * 100.0),
            p_value=p_val,
            n_obs=n_obs,
        )

    return results, round(float(fit.rsquared), 4), n_obs


def fit_fe_ols(
    df: pd.DataFrame,
    y_col: str,
    x_df: pd.DataFrame,
    fe_cols: list[str],
    min_group_size: int = 2,
) -> tuple[dict[str, HedonicTermResult], float, int]:
    """Demean y and X within fixed-effect cells (FWL theorem) and fit HC1-robust OLS.

    Returns (term_results, r2_within, n_obs_used).
    """
    work = x_df.copy()
    work["_y"] = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    fe_key = df[fe_cols].astype(str).agg("|".join, axis=1)
    work["_fe"] = fe_key.to_numpy()

    counts = work.groupby("_fe")["_y"].transform("count")
    work = work.loc[counts >= min_group_size].copy()
    if work.empty or len(work) < 30:
        return {}, 0.0, int(len(work))

    reg_cols = [c for c in x_df.columns]
    y_dem = (work["_y"] - work.groupby("_fe")["_y"].transform("mean")).to_numpy(dtype=float)

    dem_cols: list[np.ndarray] = []
    candidate_names: list[str] = []
    fe_series = work["_fe"]
    for c in reg_cols:
        col_vals = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
        dem = col_vals - col_vals.groupby(fe_series).transform("mean")
        if float(dem.std()) > 1e-8:
            dem_cols.append(dem.to_numpy(dtype=float))
            candidate_names.append(c)

    if not candidate_names:
        return {}, 0.0, int(len(work))

    X_dem_mat = np.column_stack(dem_cols)
    n_fe = int(fe_series.nunique())
    return _solve_demeaned_hc1(y_dem, X_dem_mat, candidate_names, n_fe=n_fe)


def fit_twoway_fe_ols(
    df: pd.DataFrame,
    y_col: str,
    x_df: pd.DataFrame,
    fe_col1: str,
    fe_col2: str,
    max_iter: int = 30,
    tol: float = 1e-7,
) -> tuple[dict[str, HedonicTermResult], float, int]:
    """Additive two-way fixed effects via alternating projections (Guimaraes-Portugal FWL)."""
    if len(df) < 30:
        return {}, 0.0, int(len(df))

    fe1 = pd.Series(df[fe_col1].astype(str).to_numpy(), index=df.index)
    fe2 = pd.Series(df[fe_col2].astype(str).to_numpy(), index=df.index)
    # Iteratively prune singletons in either FE dimension
    keep_mask = pd.Series(True, index=df.index)
    for _ in range(5):
        c1 = fe1.loc[keep_mask].groupby(fe1.loc[keep_mask]).transform("count")
        c2 = fe2.loc[keep_mask].groupby(fe2.loc[keep_mask]).transform("count")
        valid = (c1 >= 2) & (c2 >= 2)
        if valid.all():
            break
        keep_mask.loc[keep_mask] = valid.to_numpy()

    n_obs = int(keep_mask.sum())
    if n_obs < 30:
        return {}, 0.0, n_obs

    fe1_sub = pd.Categorical(fe1.loc[keep_mask]).codes
    fe2_sub = pd.Categorical(fe2.loc[keep_mask]).codes
    n1 = int(fe1_sub.max()) + 1
    n2 = int(fe2_sub.max()) + 1

    cols = list(x_df.columns)
    mat = np.column_stack(
        [
            pd.to_numeric(df.loc[keep_mask, y_col], errors="coerce").to_numpy(dtype=float),
            *[
                pd.to_numeric(x_df.loc[keep_mask, c], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                for c in cols
            ],
        ]
    )

    counts1 = np.bincount(fe1_sub, minlength=n1)[:, None].astype(float)
    counts2 = np.bincount(fe2_sub, minlength=n2)[:, None].astype(float)
    m_cols = mat.shape[1]

    def _group_means(
        arr: np.ndarray, codes: np.ndarray, n_grp: int, counts: np.ndarray
    ) -> np.ndarray:
        sums = np.zeros((n_grp, m_cols), dtype=float)
        np.add.at(sums, codes, arr)
        return sums / np.maximum(counts, 1.0)

    for _ in range(max_iter):
        prev = mat.copy()
        means1 = _group_means(mat, fe1_sub, n1, counts1)
        mat -= means1[fe1_sub]
        means2 = _group_means(mat, fe2_sub, n2, counts2)
        mat -= means2[fe2_sub]
        if float(np.max(np.abs(mat - prev))) < tol:
            break

    y_dem = mat[:, 0]
    X_dem = mat[:, 1:]
    valid_cols: list[int] = [j for j in range(len(cols)) if float(np.std(X_dem[:, j])) > 1e-7]
    if not valid_cols:
        return {}, 0.0, n_obs

    candidate_names = [cols[j] for j in valid_cols]
    X_cand = X_dem[:, valid_cols]
    n_fe = max(n1 + n2 - 1, 1)
    return _solve_demeaned_hc1(y_dem, X_cand, candidate_names, n_fe=n_fe)
