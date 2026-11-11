"""Time-aware, strictly leakage-safe feature engineering pipeline (`make features`)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dfv.config import PATHS, Paths
from dfv.features.leakage import assert_no_target_leakage_in_features
from dfv.geo.features import enrich_with_geo_features, fetch_and_cache_osm_pois

logger = logging.getLogger("dfv.features.build")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NUMERIC_FEATURE_COLS: list[str] = [
    "area_sqm",
    "log_area_sqm",
    "rooms",
    "has_parking",
    "is_offplan",
    "months_to_completion",
    "project_age_years",
    "completion_pct",
    "delivery_delay_months",
    "dist_nearest_metro_m",
    "dist_nearest_tram_m",
    "dist_nearest_mall_m",
    "dist_nearest_coast_m",
    "dist_airport_m",
    "dist_downtown_m",
    "poi_count_1km",
    "bus_stop_density_1km",
    "metro_within_500m",
    "metro_within_1000m",
    "lat",
    "lon",
    "area_trailing_3m_ppsqm",
    "area_trailing_6m_ppsqm",
    "area_trailing_12m_ppsqm",
    "log_area_trailing_12m_ppsqm",
    "area_trailing_3m_tx_count",
    "area_trailing_12m_tx_count",
    "project_trailing_6m_ppsqm",
    "project_trailing_12m_ppsqm",
    "segment_trailing_12m_ppsqm",
    "log_segment_trailing_12m_ppsqm",
    "area_trailing_12m_rent_ppsqm",
    "area_trailing_12m_yield",
    "year",
    "quarter",
    "month",
    "days_since_start",
]

CATEGORICAL_FEATURE_COLS: list[str] = [
    "property_type",
    "property_sub_type",
    "area_name",
    "master_project",
    "project_name",
    "developer_name",
    "nearest_metro",
]

ALL_FEATURE_COLS: list[str] = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


def _dates_to_days(dates: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    return ts.astype("int64")


def compute_strictly_prior_trailing_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    date_col: str,
    value_col: str,
    windows_days: tuple[int, ...],
    min_obs: int = 3,
    compute_counts: bool = False,
) -> pd.DataFrame:
    """Compute strictly-prior (< target date) trailing medians and optional counts per group."""
    n = len(df)
    res: dict[str, np.ndarray] = {}
    for w in windows_days:
        res[f"med_{w}"] = np.full(n, np.nan, dtype=float)
        if compute_counts:
            res[f"cnt_{w}"] = np.zeros(n, dtype=float)

    if n == 0:
        return pd.DataFrame(res, index=df.index)

    day_vals = _dates_to_days(df[date_col])
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)

    # Group row integer positions
    grouped = df.groupby(group_cols, sort=False, dropna=False).indices
    for pos_idx in grouped.values():
        idx_arr = np.asarray(pos_idx, dtype=int)
        g_days = day_vals[idx_arr]
        g_vals = vals[idx_arr]

        order = np.argsort(g_days, kind="mergesort")
        s_days = g_days[order]
        s_vals = g_vals[order]
        s_idx = idx_arr[order]

        unique_days, first_pos = np.unique(s_days, return_index=True)
        # Slices for identical days end at next first_pos (or len(s_days))
        end_pos = np.append(first_pos[1:], len(s_days))

        for w in windows_days:
            med_out = np.full(len(s_days), np.nan, dtype=float)
            cnt_out = np.zeros(len(s_days), dtype=float) if compute_counts else None

            lo_arr = np.searchsorted(s_days, unique_days - w, side="left")
            for k in range(len(unique_days)):
                hi = int(first_pos[k])
                lo = int(lo_arr[k])
                cnt = hi - lo
                end_k = int(end_pos[k])
                if cnt_out is not None:
                    cnt_out[hi:end_k] = float(cnt)
                if cnt >= min_obs:
                    slice_v = s_vals[lo:hi]
                    valid_v = slice_v[np.isfinite(slice_v)]
                    if len(valid_v) >= min_obs:
                        med_out[hi:end_k] = float(np.median(valid_v))

            res[f"med_{w}"][s_idx] = med_out
            if cnt_out is not None:
                res[f"cnt_{w}"][s_idx] = cnt_out

    return pd.DataFrame(res, index=df.index)


def compute_strictly_prior_rent_ppsqm(
    tx_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    window_days: int = 365,
    min_obs: int = 3,
) -> pd.Series:
    """Compute strictly-prior trailing 12-month area rent/sqm preferring New Ejari contracts."""
    n = len(tx_df)
    out = np.full(n, np.nan, dtype=float)
    if n == 0 or rent_df.empty:
        return pd.Series(out, index=tx_df.index, dtype=float)

    # Prefer New contracts where available per area, fallback to all contracts for thin areas
    if "is_renewal" in rent_df.columns:
        new_rents = rent_df[rent_df["is_renewal"] == 0]
        if len(new_rents) >= 20:
            rent_src = new_rents
        else:
            rent_src = rent_df
    else:
        rent_src = rent_df

    tx_days = _dates_to_days(tx_df["instance_date"])
    r_days_all = _dates_to_days(rent_src["contract_start_date"])
    r_vals_all = pd.to_numeric(rent_src["rent_per_sqm"], errors="coerce").to_numpy(dtype=float)

    tx_groups = tx_df.groupby("area_name", sort=False).indices
    r_groups = rent_src.groupby("area_name", sort=False).indices

    # Also prepare global market median fallback per day for areas lacking Ejari coverage
    global_order = np.argsort(r_days_all, kind="mergesort")
    g_r_days = r_days_all[global_order]
    g_r_vals = r_vals_all[global_order]

    for area, pos_idx in tx_groups.items():
        idx_arr = np.asarray(pos_idx, dtype=int)
        t_days = tx_days[idx_arr]
        if area in r_groups:
            r_idx = np.asarray(r_groups[area], dtype=int)
            r_d = r_days_all[r_idx]
            r_v = r_vals_all[r_idx]
            r_ord = np.argsort(r_d, kind="mergesort")
            r_d = r_d[r_ord]
            r_v = r_v[r_ord]
        else:
            r_d = g_r_days
            r_v = g_r_vals

        unique_t, inv = np.unique(t_days, return_inverse=True)
        lo_arr = np.atleast_1d(np.searchsorted(r_d, unique_t - window_days, side="left"))
        hi_arr = np.atleast_1d(np.searchsorted(r_d, unique_t, side="left"))
        u_meds = np.full(len(unique_t), np.nan, dtype=float)
        for k in range(len(unique_t)):
            lo, hi = int(lo_arr[k]), int(hi_arr[k])
            if hi - lo >= min_obs:
                sv = r_v[lo:hi]
                sv = sv[np.isfinite(sv)]
                if len(sv) >= min_obs:
                    u_meds[k] = float(np.median(sv))
        out[idx_arr] = u_meds[inv]

    return pd.Series(out, index=tx_df.index, dtype=float)


def compute_trailing_market_features(
    tx_df: pd.DataFrame,
    rent_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach strictly leakage-safe trailing market price, liquidity, rent, and yield features."""
    out = tx_df.copy()

    # Area trailing 3m, 6m, 12m medians and liquidity counts
    area_stats = compute_strictly_prior_trailing_stats(
        out,
        group_cols=["area_name"],
        date_col="instance_date",
        value_col="price_per_sqm",
        windows_days=(90, 180, 365),
        min_obs=3,
        compute_counts=True,
    )
    out["area_trailing_3m_ppsqm"] = area_stats["med_90"]
    out["area_trailing_6m_ppsqm"] = area_stats["med_180"]
    out["area_trailing_12m_ppsqm"] = area_stats["med_365"]
    out["area_trailing_3m_tx_count"] = area_stats["cnt_90"]
    out["area_trailing_12m_tx_count"] = area_stats["cnt_365"]

    # Project trailing 6m, 12m
    proj_stats = compute_strictly_prior_trailing_stats(
        out,
        group_cols=["project_name"],
        date_col="instance_date",
        value_col="price_per_sqm",
        windows_days=(180, 365),
        min_obs=2,
        compute_counts=False,
    )
    out["project_trailing_6m_ppsqm"] = proj_stats["med_180"]
    out["project_trailing_12m_ppsqm"] = proj_stats["med_365"]

    # Segment (area x type x rooms) trailing 12m
    seg_stats = compute_strictly_prior_trailing_stats(
        out,
        group_cols=["area_name", "property_type", "rooms"],
        date_col="instance_date",
        value_col="price_per_sqm",
        windows_days=(365,),
        min_obs=2,
        compute_counts=False,
    )
    out["segment_trailing_12m_ppsqm"] = seg_stats["med_365"]

    # Strictly-prior rent/sqm and yield
    out["area_trailing_12m_rent_ppsqm"] = compute_strictly_prior_rent_ppsqm(out, rent_df)

    # Hierarchical strictly-prior global fallback ONLY when local area/project history is cold-start
    # Computed using strictly-prior global median (< date) so cold-start fill never leaks future!
    global_stats = compute_strictly_prior_trailing_stats(
        out.assign(_global="ALL"),
        group_cols=["_global"],
        date_col="instance_date",
        value_col="price_per_sqm",
        windows_days=(365,),
        min_obs=1,
        compute_counts=False,
    )["med_365"]

    out["area_trailing_12m_ppsqm"] = out["area_trailing_12m_ppsqm"].fillna(global_stats)
    out["area_trailing_6m_ppsqm"] = out["area_trailing_6m_ppsqm"].fillna(
        out["area_trailing_12m_ppsqm"]
    )
    out["area_trailing_3m_ppsqm"] = out["area_trailing_3m_ppsqm"].fillna(
        out["area_trailing_6m_ppsqm"]
    )
    out["project_trailing_12m_ppsqm"] = out["project_trailing_12m_ppsqm"].fillna(
        out["area_trailing_12m_ppsqm"]
    )
    out["project_trailing_6m_ppsqm"] = out["project_trailing_6m_ppsqm"].fillna(
        out["project_trailing_12m_ppsqm"]
    )
    out["segment_trailing_12m_ppsqm"] = out["segment_trailing_12m_ppsqm"].fillna(
        out["area_trailing_12m_ppsqm"]
    )

    out["log_area_trailing_12m_ppsqm"] = np.log(out["area_trailing_12m_ppsqm"].clip(lower=1.0))
    out["log_segment_trailing_12m_ppsqm"] = np.log(
        out["segment_trailing_12m_ppsqm"].clip(lower=1.0)
    )
    out["area_trailing_12m_yield"] = (
        out["area_trailing_12m_rent_ppsqm"] / out["area_trailing_12m_ppsqm"]
    )
    return out


def build_feature_matrix(
    tx_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    metro_df: pd.DataFrame,
    projects_df: pd.DataFrame | None = None,
    cache_dir: Path = PATHS.cache,
    raw_dir: Path = PATHS.raw,
    allow_nominatim: bool = False,
    drop_flagged: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build complete leakage-safe feature dataset and return (feature_df, feature_cols, geo_stats)."""
    work = tx_df.copy()
    if drop_flagged and {"flag_outlier", "flag_non_market", "flag_duplicate"}.issubset(
        work.columns
    ):
        work = work.loc[
            ~work["flag_outlier"] & ~work["flag_non_market"] & ~work["flag_duplicate"]
        ].copy()

    work = work.sort_values("instance_date").reset_index(drop=True)

    # 1. Geospatial enrichment
    work, geo_stats = enrich_with_geo_features(
        work,
        metro_df=metro_df,
        projects_df=projects_df,
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        allow_nominatim=allow_nominatim,
    )

    # 2. Unit & time features
    work["log_area_sqm"] = np.log(work["area_sqm"].clip(lower=1.0))
    dt = pd.to_datetime(work["instance_date"])
    work["year"] = dt.dt.year.astype(int)
    work["quarter"] = dt.dt.quarter.astype(int)
    work["month"] = dt.dt.month.astype(int)
    work["days_since_start"] = (dt - pd.Timestamp("2015-01-01")).dt.days.astype(int)

    # 3. Project/developer lifecycle features (strictly evaluated at transaction date `dt`)
    if projects_df is not None and not projects_df.empty and "project_name" in projects_df.columns:
        p_meta = projects_df.drop_duplicates(subset=["project_name"])[
            [
                c
                for c in (
                    "project_name",
                    "developer_name",
                    "launch_date",
                    "completion_date",
                    "scheduled_completion_date",
                )
                if c in projects_df.columns
            ]
        ].rename(columns={"developer_name": "_proj_dev"})
        work = work.merge(p_meta, on="project_name", how="left")
        if "_proj_dev" in work.columns:
            is_unk = work["developer_name"].astype(str).isin(["Unknown", "nan", "None", ""])
            work["developer_name"] = np.where(
                is_unk & work["_proj_dev"].notna(), work["_proj_dev"], work["developer_name"]
            )
            work = work.drop(columns=["_proj_dev"])

    default_age = pd.Series(
        np.where(work["is_offplan"] == 1, 0.5, 5.0), index=work.index, dtype="float64"
    )
    launch_dt = (
        pd.to_datetime(work["launch_date"], errors="coerce")
        if "launch_date" in work.columns
        else pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    )
    if "launch_date" in work.columns:
        work["project_age_years"] = (
            ((dt - launch_dt).dt.days / 365.25).clip(0.0, 40.0).fillna(default_age)
        )
    else:
        work["project_age_years"] = default_age

    default_mtc = pd.Series(
        np.where(work["is_offplan"] == 1, 24.0, 0.0), index=work.index, dtype="float64"
    )
    comp_dt = (
        pd.to_datetime(work["completion_date"], errors="coerce")
        if "completion_date" in work.columns
        else pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    )

    # Time-aware completion_pct: Ready resales = 100%, off-plan interpolated strictly at `dt` and capped <= 95%
    interp_comp = pd.Series(50.0, index=work.index, dtype="float64")
    has_span = launch_dt.notna() & comp_dt.notna() & (comp_dt > launch_dt)
    span_days = (comp_dt - launch_dt).dt.days.astype("float64").replace(0.0, np.nan)
    progress = ((dt - launch_dt).dt.days.astype("float64") / span_days).clip(0.0, 1.0)
    interp_comp = pd.Series(
        np.where(has_span, np.clip(5.0 + 90.0 * progress.fillna(0.5), 5.0, 95.0), 50.0),
        index=work.index,
        dtype="float64",
    )
    work["completion_pct"] = np.where(work["is_offplan"] == 0, 100.0, interp_comp)

    if "completion_date" in work.columns:
        diff_months = (comp_dt - dt).dt.days / 30.4375
        if "months_to_completion" not in work.columns:
            work["months_to_completion"] = np.where(
                work["is_offplan"] == 1, np.clip(diff_months.fillna(24.0), 0.0, 60.0), 0.0
            )
        else:
            work["months_to_completion"] = pd.to_numeric(
                work["months_to_completion"], errors="coerce"
            ).fillna(default_mtc)
    else:
        if "months_to_completion" not in work.columns:
            work["months_to_completion"] = default_mtc

    # Delivery delay is defined only when an explicit scheduled_completion_date exists and was already passed at `dt`
    if "scheduled_completion_date" in work.columns:
        sched_dt = pd.to_datetime(work["scheduled_completion_date"], errors="coerce")
        past_sched = sched_dt.notna() & (dt > sched_dt)
        actual_end = comp_dt.where(comp_dt.notna() & (comp_dt <= dt), dt)
        raw_delay = (actual_end - sched_dt).dt.days / 30.4375
        work["delivery_delay_months"] = np.where(past_sched, np.maximum(raw_delay, 0.0), 0.0)
    else:
        work["delivery_delay_months"] = 0.0

    for drop_tmp in ("launch_date", "completion_date", "scheduled_completion_date"):
        if drop_tmp in work.columns:
            work = work.drop(columns=[drop_tmp])

    # 4. Strictly-prior trailing market state features
    work = compute_trailing_market_features(work, rent_df)

    # 5. Clean categorical string dtype + target
    for col in CATEGORICAL_FEATURE_COLS:
        if col not in work.columns:
            work[col] = "Unknown"
        work[col] = work[col].astype("string").fillna("Unknown").astype(str)

    work["log_price_per_sqm"] = np.log(work["price_per_sqm"].clip(lower=1.0))

    assert_no_target_leakage_in_features(ALL_FEATURE_COLS)
    return work, list(ALL_FEATURE_COLS), geo_stats


def run_feature_pipeline(paths: Paths = PATHS, fetch_osm: bool = False) -> dict[str, Any]:
    """Build feature matrix for processed data (if present) and sample data."""
    if fetch_osm:
        fetch_and_cache_osm_pois(paths.cache)

    summary: dict[str, Any] = {}
    has_processed = (paths.processed / "transactions.parquet").exists()
    if has_processed:
        tx = pd.read_parquet(paths.processed / "transactions.parquet")
        rents = pd.read_parquet(paths.processed / "rent_contracts.parquet")
        projs = pd.read_parquet(paths.processed / "projects.parquet")
        metro = pd.read_parquet(paths.processed / "metro_stations.parquet")
        feat_df, feat_cols, geo_stats = build_feature_matrix(
            tx,
            rents,
            metro,
            projects_df=projs,
            cache_dir=paths.cache,
            raw_dir=paths.raw,
        )
        feat_df.to_parquet(paths.processed / "features.parquet", index=False)
        logger.info(
            "Built processed features: %s rows, %d features, geocode=%s",
            f"{len(feat_df):,}",
            len(feat_cols),
            geo_stats,
        )
        summary["processed"] = {"rows": len(feat_df), "features": len(feat_cols), "geo": geo_stats}

    if (paths.sample / "transactions.parquet").exists():
        tx_s = pd.read_parquet(paths.sample / "transactions.parquet")
        rents_s = pd.read_parquet(paths.sample / "rent_contracts.parquet")
        projs_s = pd.read_parquet(paths.sample / "projects.parquet")
        metro_s = pd.read_parquet(paths.sample / "metro_stations.parquet")
        feat_s, feat_cols_s, geo_s = build_feature_matrix(
            tx_s,
            rents_s,
            metro_s,
            projects_df=projs_s,
            cache_dir=paths.cache,
            raw_dir=paths.raw,
        )
        feat_s.to_parquet(paths.sample / "features.parquet", index=False)
        logger.info(
            "Built sample features: %s rows, %d features", f"{len(feat_s):,}", len(feat_cols_s)
        )
        summary["sample"] = {"rows": len(feat_s), "features": len(feat_cols_s), "geo": geo_s}

    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build leakage-safe geo & market features.")
    parser.add_argument(
        "--fetch-osm",
        action="store_true",
        help="Attempt OSM Overpass POI download and cache if not present.",
    )
    args = parser.parse_args(argv)
    run_feature_pipeline(PATHS, fetch_osm=args.fetch_osm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
