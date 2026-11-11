"""End-to-end training, temporal/spatial validation, intervals, SHAP, and acceptance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from dfv.config import PATHS, Paths
from dfv.features.build import ALL_FEATURE_COLS, CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS
from dfv.features.leakage import (
    assert_no_target_leakage_in_features,
    assert_oof_encoding_strict,
    assert_temporal_split_valid,
)
from dfv.models.baseline import (
    compute_regression_metrics,
    evaluate_global_median_baseline,
    evaluate_trailing_segment_baseline,
    fit_and_evaluate_ridge_baseline,
)
from dfv.models.explain import generate_shap_plots
from dfv.models.hybrid import ResidualBoostedLGBM
from dfv.models.intervals import (
    evaluate_interval_coverage,
    fit_conformal_quantile_intervals,
)
from dfv.models.registry import (
    ModelBundle,
    build_metadata,
    current_git_sha,
    save_model_bundle,
)
from dfv.models.tune import tune_lightgbm

logger = logging.getLogger("dfv.models.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LOW_CARD_RIDGE_CATS: list[str] = ["property_type", "property_sub_type", "area_name"]


def compute_oof_target_encoding(
    train_df: pd.DataFrame,
    eval_dfs: list[pd.DataFrame],
    group_cols: list[str],
    target_col: str = "log_price_per_sqm",
    prior_weight: float = 20.0,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, dict[str, float]], float]:
    """Fit out-of-fold target encoding strictly within train_df and transform eval_dfs."""
    tr = train_df.copy().reset_index(drop=True)
    eval_outs = [d.copy().reset_index(drop=True) for d in eval_dfs]
    global_mean = float(tr[target_col].mean())

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.zeros(len(tr), dtype=int)
    for f_idx, (_, val_i) in enumerate(kf.split(tr)):
        fold_ids[val_i] = f_idx

    mappings: dict[str, dict[str, float]] = {}
    for col in group_cols:
        enc_col = f"{col}_oof_enc"
        oof_vals = np.full(len(tr), global_mean, dtype=float)
        col_str = tr[col].astype(str)

        for f_idx in range(n_splits):
            in_fold = fold_ids != f_idx
            out_fold = fold_ids == f_idx
            stats = (
                pd.DataFrame({"g": col_str[in_fold], "y": tr.loc[in_fold, target_col]})
                .groupby("g")["y"]
                .agg(["sum", "count"])
            )
            f_map = (
                (stats["sum"] + prior_weight * global_mean) / (stats["count"] + prior_weight)
            ).to_dict()
            oof_vals[out_fold] = col_str[out_fold].map(f_map).fillna(global_mean).to_numpy()

        tr[enc_col] = oof_vals
        assert_oof_encoding_strict(tr, col, target_col, fold_ids, oof_vals, prior_weight)

        full_stats = (
            pd.DataFrame({"g": col_str, "y": tr[target_col]})
            .groupby("g")["y"]
            .agg(["sum", "count"])
        )
        full_map = {
            str(k): float(v)
            for k, v in (
                (full_stats["sum"] + prior_weight * global_mean)
                / (full_stats["count"] + prior_weight)
            ).items()
        }
        mappings[col] = full_map
        for ev in eval_outs:
            ev[enc_col] = ev[col].astype(str).map(full_map).fillna(global_mean)

    return tr, eval_outs, mappings, global_mean


def build_area_defaults(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Precompute recent per-area defaults for single-row API/UI inference."""
    dt = pd.to_datetime(df["instance_date"])
    recent = df.loc[dt >= (dt.max() - pd.Timedelta(days=365))].copy()
    if recent.empty:
        recent = df

    defaults: dict[str, dict[str, Any]] = {}
    for area, grp in recent.groupby("area_name"):
        seg_counts: dict[str, int] = {}
        for seg_key, cnt in (
            grp.groupby(["property_type", "rooms"])["price_per_sqm"].count().items()
        ):
            if isinstance(seg_key, tuple) and len(seg_key) == 2:
                seg_counts[f"{seg_key[0]}|{int(seg_key[1])}"] = int(cnt)
        defaults[str(area)] = {
            "lat": float(grp["lat"].median()),
            "lon": float(grp["lon"].median()),
            "master_project": str(grp["master_project"].mode().iloc[0]),
            "nearest_metro": str(grp["nearest_metro"].mode().iloc[0]),
            "dist_nearest_metro_m": float(grp["dist_nearest_metro_m"].median()),
            "dist_nearest_tram_m": float(grp["dist_nearest_tram_m"].median()),
            "dist_nearest_mall_m": float(grp["dist_nearest_mall_m"].median()),
            "dist_nearest_coast_m": float(grp["dist_nearest_coast_m"].median()),
            "dist_airport_m": float(grp["dist_airport_m"].median()),
            "dist_downtown_m": float(grp["dist_downtown_m"].median()),
            "poi_count_1km": float(grp["poi_count_1km"].median()),
            "bus_stop_density_1km": float(grp["bus_stop_density_1km"].median()),
            "area_trailing_3m_ppsqm": float(grp["area_trailing_3m_ppsqm"].iloc[-1]),
            "area_trailing_6m_ppsqm": float(grp["area_trailing_6m_ppsqm"].iloc[-1]),
            "area_trailing_12m_ppsqm": float(grp["area_trailing_12m_ppsqm"].iloc[-1]),
            "area_trailing_3m_tx_count": float(grp["area_trailing_3m_tx_count"].iloc[-1]),
            "area_trailing_12m_tx_count": float(grp["area_trailing_12m_tx_count"].iloc[-1]),
            "area_trailing_12m_rent_ppsqm": float(
                grp["area_trailing_12m_rent_ppsqm"].dropna().iloc[-1]
                if grp["area_trailing_12m_rent_ppsqm"].notna().any()
                else 1100.0
            ),
            "area_trailing_12m_yield": float(
                grp["area_trailing_12m_yield"].dropna().iloc[-1]
                if grp["area_trailing_12m_yield"].notna().any()
                else 0.065
            ),
            "segment_counts": seg_counts,
            "area_tx_count": int(len(grp)),
        }
    return defaults


def _prepare_lgb_matrix(
    df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for c in cat_cols:
        out[c] = out[c].astype("category")
    for c in feature_cols:
        if c not in cat_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _extract_lgb_params(best_params: dict[str, Any], seed: int) -> tuple[dict[str, Any], float]:
    ridge_alpha = float(best_params.get("ridge_alpha", 10.0))
    lgb_params = {
        "objective": "regression_l1",
        "n_estimators": int(best_params.get("n_estimators", 250)),
        "learning_rate": float(best_params.get("learning_rate", 0.05)),
        "max_depth": int(best_params.get("max_depth", -1)),
        "num_leaves": int(best_params.get("num_leaves", 63)),
        "min_child_samples": int(best_params.get("min_child_samples", 30)),
        "subsample": float(best_params.get("subsample", 0.85)),
        "subsample_freq": 1,
        "colsample_bytree": float(best_params.get("colsample_bytree", 0.85)),
        "reg_alpha": float(best_params.get("reg_alpha", 0.05)),
        "reg_lambda": float(best_params.get("reg_lambda", 1.0)),
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 4,
        "verbose": -1,
    }
    return lgb_params, ridge_alpha


def split_temporal(
    feat_df: pd.DataFrame,
    val_months: int = 12,
    test_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted features into train <= T-12m, val_tune, cal, test (last 6m)."""
    work = feat_df.sort_values("instance_date").reset_index(drop=True)
    dt = pd.to_datetime(work["instance_date"])
    max_dt = dt.max()
    val_start = max_dt - pd.DateOffset(months=val_months)
    test_start = max_dt - pd.DateOffset(months=test_months)

    train_df = work.loc[dt <= val_start].copy()
    val_full = work.loc[(dt > val_start) & (dt <= test_start)].copy()
    test_df = work.loc[dt > test_start].copy()

    # Split val chronologically: first 50% for Optuna tuning, latter 50% strictly for CQR calibration
    mid_idx = max(1, len(val_full) // 2)
    val_tune_df = val_full.iloc[:mid_idx].copy()
    cal_df = val_full.iloc[mid_idx:].copy()

    val_tune_min = pd.to_datetime(val_tune_df["instance_date"]).min()
    train_df = train_df.loc[pd.to_datetime(train_df["instance_date"]) < val_tune_min].copy()
    cal_max = pd.to_datetime(cal_df["instance_date"]).max()
    test_df = test_df.loc[pd.to_datetime(test_df["instance_date"]) > cal_max].copy()

    assert_temporal_split_valid(train_df, val_full, test_df)
    return train_df, val_tune_df, cal_df, test_df


def _build_point_estimator(
    use_residual_hybrid: bool,
    lgb_params: dict[str, Any],
    ridge_alpha: float,
) -> Any:
    if use_residual_hybrid:
        return ResidualBoostedLGBM(
            ridge_numeric_cols=NUMERIC_FEATURE_COLS,
            ridge_cat_cols=LOW_CARD_RIDGE_CATS,
            lgb_params=lgb_params,
            ridge_alpha=ridge_alpha,
        )
    return lgb.LGBMRegressor(**lgb_params)


def run_rolling_origin_cv(
    feat_df: pd.DataFrame,
    best_params: dict[str, Any],
    use_residual_hybrid: bool,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate 3 expanding-window temporal CV folds (6-month horizons)."""
    dt = pd.to_datetime(feat_df["instance_date"])
    max_dt = dt.max()
    fold_results: list[dict[str, Any]] = []
    oof_cols = ["developer_name", "project_name"]
    feature_cols = [
        c
        for c in list(ALL_FEATURE_COLS) + [f"{c}_oof_enc" for c in oof_cols]
        if c != "project_name"
    ]
    cat_cols = [c for c in CATEGORICAL_FEATURE_COLS if c != "project_name"]
    lgb_params, ridge_alpha = _extract_lgb_params(best_params, seed=seed)

    for offset_m in (24, 18, 12):
        cutoff = max_dt - pd.DateOffset(months=offset_m)
        horizon = cutoff + pd.DateOffset(months=6)
        tr = feat_df.loc[dt < cutoff].copy()
        te = feat_df.loc[(dt >= cutoff) & (dt < horizon)].copy()
        if len(tr) < 200 or len(te) < 50:
            continue
        tr_e, [te_e], _, _ = compute_oof_target_encoding(tr, [te], oof_cols, seed=seed)
        X_tr = _prepare_lgb_matrix(tr_e, feature_cols, cat_cols)
        X_te = _prepare_lgb_matrix(te_e, feature_cols, cat_cols)
        mdl = _build_point_estimator(use_residual_hybrid, lgb_params, ridge_alpha)
        mdl.fit(X_tr, tr_e["log_price_per_sqm"].to_numpy(dtype=float))
        pred_log = np.asarray(mdl.predict(X_te), dtype=float)
        pred_tot = np.exp(pred_log) * te_e["area_sqm"].to_numpy(dtype=float)
        m = compute_regression_metrics(
            f"rolling_cv_{cutoff.strftime('%Y-%m')}",
            te_e["actual_worth"].to_numpy(dtype=float),
            pred_tot,
            te_e["log_price_per_sqm"].to_numpy(dtype=float),
            pred_log,
        )
        fold_results.append(
            {
                "fold_start": cutoff.strftime("%Y-%m-%d"),
                "fold_end": horizon.strftime("%Y-%m-%d"),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                **m.to_dict(),
            }
        )
    return fold_results


def run_spatial_holdout(
    feat_df: pd.DataFrame,
    best_params: dict[str, Any],
    use_residual_hybrid: bool,
    seed: int = 42,
) -> dict[str, Any]:
    """Hold out 20% of entire unseen projects to measure spatial generalization."""
    rng = np.random.default_rng(seed)
    projects = np.array(sorted(feat_df["project_name"].unique()))
    n_hold = max(1, int(0.20 * len(projects)))
    held_projects = set(rng.choice(projects, size=n_hold, replace=False).tolist())

    tr = feat_df.loc[~feat_df["project_name"].isin(held_projects)].copy()
    te = feat_df.loc[feat_df["project_name"].isin(held_projects)].copy()
    te["project_trailing_6m_ppsqm"] = te["area_trailing_6m_ppsqm"]
    te["project_trailing_12m_ppsqm"] = te["area_trailing_12m_ppsqm"]

    oof_cols = ["developer_name", "project_name"]
    feature_cols = [
        c
        for c in list(ALL_FEATURE_COLS) + [f"{c}_oof_enc" for c in oof_cols]
        if c != "project_name"
    ]
    cat_cols = [c for c in CATEGORICAL_FEATURE_COLS if c != "project_name"]
    lgb_params, ridge_alpha = _extract_lgb_params(best_params, seed=seed)

    tr_e, [te_e], _, global_prior = compute_oof_target_encoding(tr, [te], oof_cols, seed=seed)
    te_e["project_name_oof_enc"] = global_prior
    X_tr = _prepare_lgb_matrix(tr_e, feature_cols, cat_cols)
    X_te = _prepare_lgb_matrix(te_e, feature_cols, cat_cols)

    mdl = _build_point_estimator(use_residual_hybrid, lgb_params, ridge_alpha)
    mdl.fit(X_tr, tr_e["log_price_per_sqm"].to_numpy(dtype=float))
    pred_log = np.asarray(mdl.predict(X_te), dtype=float)
    pred_tot = np.exp(pred_log) * te_e["area_sqm"].to_numpy(dtype=float)
    m = compute_regression_metrics(
        "spatial_unseen_projects",
        te_e["actual_worth"].to_numpy(dtype=float),
        pred_tot,
        te_e["log_price_per_sqm"].to_numpy(dtype=float),
        pred_log,
    )
    return {
        "held_out_projects": int(len(held_projects)),
        "held_out_rows": int(len(te)),
        **m.to_dict(),
    }


def write_error_analysis_report(
    test_df: pd.DataFrame,
    pred_total: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    output_path: Path,
) -> Path:
    """Write honest segment-level error analysis and worst predictions to reports/error_analysis.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = test_df.copy()
    work["pred_total"] = pred_total
    work["low"] = low
    work["high"] = high
    work["ape"] = np.abs(work["actual_worth"] - work["pred_total"]) / np.maximum(
        work["actual_worth"], 1.0
    )
    work["covered"] = (work["actual_worth"] >= work["low"]) & (work["actual_worth"] <= work["high"])
    work["month_str"] = pd.to_datetime(work["instance_date"]).dt.strftime("%Y-%m")

    def _seg_table(mask: pd.Series, label: str) -> dict[str, Any]:
        sub = work.loc[mask]
        if sub.empty:
            return {"segment": label, "n": 0, "mdape_pct": "n/a", "coverage_pct": "n/a"}
        return {
            "segment": label,
            "n": f"{len(sub):,}",
            "mdape_pct": f"{sub['ape'].median() * 100:.2f}%",
            "coverage_pct": f"{sub['covered'].mean() * 100:.1f}%",
        }

    segments = [
        _seg_table(pd.Series(True, index=work.index), "All Test Transactions"),
        _seg_table(work["property_type"] == "Unit", "Apartments (Unit)"),
        _seg_table(work["property_type"] == "Villa", "Villas & Townhouses"),
        _seg_table(work["is_offplan"] == 1, "Off-Plan Sales"),
        _seg_table(work["is_offplan"] == 0, "Ready / Secondary Sales"),
        _seg_table(work["actual_worth"] >= 5_000_000, "Luxury (>= AED 5M)"),
        _seg_table(
            work["area_trailing_12m_tx_count"] < 150, "Thin-Data Areas (<150 trailing sales)"
        ),
    ]

    lines = [
        "# Honest Error Analysis (`DubaiFairValue`)",
        "",
        "Model performance across segments, temporal drift over test months, and worst test predictions.",
        "",
        "## Segment Error & Interval Coverage Breakdown",
        "",
        "| Segment | Test Rows | MdAPE | 90% Interval Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in segments:
        lines.append(f"| {s['segment']} | {s['n']} | {s['mdape_pct']} | {s['coverage_pct']} |")

    lines.extend(
        [
            "",
            "## Monthly Drift Across Test Window",
            "",
            "| Month | Rows | MdAPE | Coverage |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for m_str, grp in work.groupby("month_str"):
        lines.append(
            f"| {m_str} | {len(grp):,} | {grp['ape'].median() * 100:.2f}% | {grp['covered'].mean() * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Top 10 Worst Test Predictions (with Likely Causes)",
            "",
            "| Date | Area | Type | Area (sqm) | Actual (AED) | Predicted (AED) | Abs % Error | Note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    worst = work.sort_values("ape", ascending=False).head(10)
    for _, r in worst.iterrows():
        reasons = []
        if r["actual_worth"] >= 5_000_000:
            reasons.append("ultra-luxury tail")
        if r["area_trailing_12m_tx_count"] < 50:
            reasons.append("thin area history")
        if r["property_type"] == "Villa":
            reasons.append("heterogeneous villa plot/finish")
        if r["is_offplan"] == 1:
            reasons.append("off-plan payment/launch tier")
        if not reasons:
            reasons.append("unobserved view/floor/fit-out")
        lines.append(
            f"| {r['instance_date']} | {r['area_name']} | {r['property_type']} | "
            f"{r['area_sqm']:.0f} | {r['actual_worth']:,.0f} | {r['pred_total']:,.0f} | "
            f"{r['ape'] * 100:.1f}% | {', '.join(reasons)} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _manifest_hash(raw_dir: Path) -> str:
    mp = raw_dir / "MANIFEST.json"
    if not mp.exists():
        return "sample_bundle_v1"
    return hashlib.sha256(mp.read_bytes()).hexdigest()[:16]


def train_pipeline(
    features_path: Path,
    model_out_dir: Path,
    reports_dir: Path | None = None,
    figures_dir: Path | None = None,
    raw_dir: Path = PATHS.raw,
    dataset_mode: str = "processed",
    seed: int = 42,
    n_trials: int = 10,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Train baselines, standalone LightGBM, Ridge+Residual-LGBM, CQR conformal intervals, SHAP, and gates."""
    feat_df = pd.read_parquet(features_path)
    train_df, val_tune_df, cal_df, test_df = split_temporal(feat_df)

    oof_cols = ["developer_name", "project_name"]
    train_e, [val_e, cal_e, test_e], oof_mappings, global_prior = compute_oof_target_encoding(
        train_df, [val_tune_df, cal_df, test_df], oof_cols, seed=seed
    )

    feature_cols = list(ALL_FEATURE_COLS) + [f"{c}_oof_enc" for c in oof_cols]
    cat_cols = [c for c in CATEGORICAL_FEATURE_COLS if c != "project_name"]
    model_features = [c for c in feature_cols if c != "project_name"]
    assert_no_target_leakage_in_features(model_features)

    # 1. Baselines on test set
    b_global = evaluate_global_median_baseline(train_e, test_e)
    b_seg = evaluate_trailing_segment_baseline(test_e)
    _, b_ridge = fit_and_evaluate_ridge_baseline(
        train_e, test_e, NUMERIC_FEATURE_COLS, CATEGORICAL_FEATURE_COLS
    )
    baselines = [b_global, b_seg, b_ridge]
    best_baseline = min(baselines, key=lambda b: b.mdape)

    # 2. Optuna tuning + Ridge-Residual LightGBM (plus standalone LightGBM benchmark)
    X_tr = _prepare_lgb_matrix(train_e, model_features, cat_cols)
    X_val = _prepare_lgb_matrix(val_e, model_features, cat_cols)
    X_cal = _prepare_lgb_matrix(cal_e, model_features, cat_cols)
    X_te = _prepare_lgb_matrix(test_e, model_features, cat_cols)

    y_tr = train_e["log_price_per_sqm"].to_numpy(dtype=float)
    y_val = val_e["log_price_per_sqm"].to_numpy(dtype=float)
    y_cal = cal_e["log_price_per_sqm"].to_numpy(dtype=float)
    y_te = test_e["log_price_per_sqm"].to_numpy(dtype=float)

    use_residual_hybrid = len(X_tr) < 20_000
    best_params, trial_logs = tune_lightgbm(
        X_tr,
        y_tr,
        X_val,
        y_val,
        area_val=val_e["area_sqm"].to_numpy(dtype=float),
        y_true_val_total=val_e["actual_worth"].to_numpy(dtype=float),
        ridge_numeric_cols=NUMERIC_FEATURE_COLS,
        ridge_cat_cols=LOW_CARD_RIDGE_CATS,
        use_residual_hybrid=use_residual_hybrid,
        seed=seed,
        n_trials=n_trials,
    )

    tr_plus_val = pd.concat([train_e, val_e], ignore_index=True)
    X_tr_val = _prepare_lgb_matrix(tr_plus_val, model_features, cat_cols)
    y_tr_val = tr_plus_val["log_price_per_sqm"].to_numpy(dtype=float)
    lgb_params, ridge_alpha = _extract_lgb_params(best_params, seed=seed)

    # Secondary comparison benchmark (standalone LGBM when hybrid active, or hybrid when standalone active)
    alt_model = _build_point_estimator(not use_residual_hybrid, lgb_params, ridge_alpha)
    alt_model.fit(X_tr_val, y_tr_val)
    alt_pred_log = np.asarray(alt_model.predict(X_te), dtype=float)
    alt_name = "lightgbm_standalone" if use_residual_hybrid else "ridge_residual_lightgbm"
    alt_metrics = compute_regression_metrics(
        alt_name,
        test_e["actual_worth"].to_numpy(dtype=float),
        np.exp(alt_pred_log) * test_e["area_sqm"].to_numpy(dtype=float),
        y_te,
        alt_pred_log,
    )

    # Primary model (ADR-008): Optuna-tuned LightGBM for >=20k rows, Ridge+Residual-LightGBM for <20k sample
    primary_name = "ridge_residual_lightgbm" if use_residual_hybrid else "lightgbm_tuned"
    point_model = _build_point_estimator(use_residual_hybrid, lgb_params, ridge_alpha)
    point_model.fit(X_tr_val, y_tr_val)

    point_model_rep = _build_point_estimator(use_residual_hybrid, lgb_params, ridge_alpha)
    point_model_rep.fit(X_tr_val, y_tr_val)
    pred_log_te = np.asarray(point_model.predict(X_te), dtype=float)
    pred_log_te_rep = np.asarray(point_model_rep.predict(X_te), dtype=float)
    repro_diff = float(np.max(np.abs(pred_log_te - pred_log_te_rep)))

    pred_total_te = np.exp(pred_log_te) * test_e["area_sqm"].to_numpy(dtype=float)
    model_metrics = compute_regression_metrics(
        primary_name,
        test_e["actual_worth"].to_numpy(dtype=float),
        pred_total_te,
        y_te,
        pred_log_te,
    )

    # 3. Quantile LightGBM + CQR calibration
    interval_model = fit_conformal_quantile_intervals(
        X_tr_val,
        y_tr_val,
        X_cal,
        y_cal,
        best_params=best_params,
        point_model=point_model,
        seed=seed,
        alpha=0.10,
    )
    low_te, med_te, high_te = interval_model.predict_aed_interval(
        X_te, test_e["area_sqm"].to_numpy(dtype=float), pred_log_te
    )
    cov_stats = evaluate_interval_coverage(
        test_e["actual_worth"].to_numpy(dtype=float), low_te, med_te, high_te
    )

    # 4. Rolling-origin CV + spatial hold-out
    cv_folds = run_rolling_origin_cv(feat_df, best_params, use_residual_hybrid, seed=seed)
    spatial_res = run_spatial_holdout(feat_df, best_params, use_residual_hybrid, seed=seed)

    # 5. Model acceptance gates
    beats_baseline = bool(model_metrics.mdape < best_baseline.mdape)
    coverage_ok = bool(abs(cov_stats["coverage"] - 0.90) <= 0.05)
    reproducible_ok = bool(repro_diff < 1e-6)
    all_passed = beats_baseline and coverage_ok and reproducible_ok

    metrics_payload: dict[str, Any] = {
        "dataset_mode": dataset_mode,
        "model_architecture": primary_name,
        "rows_total": int(len(feat_df)),
        "split_sizes": {
            "train": int(len(train_e)),
            "val_tune": int(len(val_e)),
            "calibration": int(len(cal_e)),
            "test": int(len(test_e)),
        },
        "training_window": {
            "train_start": str(train_e["instance_date"].min()),
            "train_end": str(train_e["instance_date"].max()),
            "val_start": str(val_e["instance_date"].min()),
            "cal_end": str(cal_e["instance_date"].max()),
            "test_start": str(test_e["instance_date"].min()),
            "test_end": str(test_e["instance_date"].max()),
        },
        "baselines": [b.to_dict() for b in baselines] + [alt_metrics.to_dict()],
        "best_params": best_params,
        "optuna_trials": trial_logs,
        "test_metrics": {
            "model_mdape": round(model_metrics.mdape, 5),
            "model_mape": round(model_metrics.mape, 5),
            "model_rmse_log": round(model_metrics.rmse_log, 5),
            "model_r2_log": round(model_metrics.r2_log, 5),
            "alt_comparison_name": alt_name,
            "alt_comparison_mdape": round(alt_metrics.mdape, 5),
            "best_baseline_name": best_baseline.name,
            "best_baseline_mdape": round(best_baseline.mdape, 5),
            "mdape_improvement_abs": round(best_baseline.mdape - model_metrics.mdape, 5),
            "mdape_improvement_rel_pct": round(
                100.0
                * (best_baseline.mdape - model_metrics.mdape)
                / max(best_baseline.mdape, 1e-9),
                2,
            ),
            "interval_coverage": cov_stats["coverage"],
            "interval_median_rel_width": cov_stats["median_rel_width"],
            "interval_median_abs_width_aed": cov_stats["median_abs_width_aed"],
            "spatial_mdape": spatial_res["mdape"],
            "spatial_r2_log": spatial_res["r2_log"],
            "repro_max_log_diff": repro_diff,
        },
        "rolling_cv": cv_folds,
        "spatial_holdout": spatial_res,
        "gates": {
            "beats_best_baseline": beats_baseline,
            "coverage_within_5pct_of_90": coverage_ok,
            "reproducible_with_seed": reproducible_ok,
            "all_passed": all_passed,
        },
    }

    # 6. Save model bundle + strict metadata
    area_defaults = build_area_defaults(feat_df)
    manifest_h = _manifest_hash(raw_dir) if dataset_mode != "sample" else "sample_bundle_v1"
    meta = build_metadata(
        model_version=f"{primary_name}-cqr-{dataset_mode}-v1",
        feature_list=model_features,
        categorical_features=cat_cols,
        training_window=metrics_payload["training_window"],
        git_sha=current_git_sha(PATHS.root),
        metrics=metrics_payload["test_metrics"],
        data_manifest_hash=manifest_h,
        dataset_mode=dataset_mode,
    )
    bundle = ModelBundle(
        point_model=point_model,
        interval_model=interval_model,
        oof_mappings=oof_mappings,
        global_prior_log=global_prior,
        area_defaults=area_defaults,
        metadata=meta,
    )
    save_model_bundle(bundle, model_out_dir)

    if write_reports and reports_dir is not None and figures_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        write_error_analysis_report(
            test_e, pred_total_te, low_te, high_te, reports_dir / "error_analysis.md"
        )
        shap_sample = X_te.sample(n=min(800, len(X_te)), random_state=seed)
        generate_shap_plots(point_model, shap_sample, figures_dir)

    logger.info(
        "Training (%s | %s) complete: MdAPE=%.4f vs %s=%.4f vs baseline=%.4f, coverage=%.3f, spatial_mdape=%.4f, gates=%s",
        dataset_mode,
        primary_name,
        model_metrics.mdape,
        alt_name,
        alt_metrics.mdape,
        best_baseline.mdape,
        cov_stats["coverage"],
        spatial_res["mdape"],
        all_passed,
    )
    return metrics_payload


def run_training_pipeline(
    paths: Paths = PATHS, seed: int = 42, n_trials: int = 10
) -> dict[str, Any]:
    """Train sample model (committed for CI/demo) and full processed model if present."""
    results: dict[str, Any] = {}
    has_processed = (paths.processed / "features.parquet").exists()

    if (paths.sample / "features.parquet").exists():
        results["sample"] = train_pipeline(
            features_path=paths.sample / "features.parquet",
            model_out_dir=paths.sample_models,
            reports_dir=paths.reports if not has_processed else None,
            figures_dir=paths.figures if not has_processed else None,
            raw_dir=paths.raw,
            dataset_mode="sample",
            seed=seed,
            n_trials=min(n_trials, 6),
            write_reports=not has_processed,
        )

    if has_processed:
        results["processed"] = train_pipeline(
            features_path=paths.processed / "features.parquet",
            model_out_dir=paths.models,
            reports_dir=paths.reports,
            figures_dir=paths.figures,
            raw_dir=paths.raw,
            dataset_mode="processed_dld_fallback",
            seed=seed,
            n_trials=n_trials,
            write_reports=True,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DubaiFairValue models and evaluate gates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args(argv)
    run_training_pipeline(PATHS, seed=args.seed, n_trials=args.trials)
    return 0


if __name__ == "__main__":
    sys.exit(main())
