"""Monthly market-cycle index, DLD index cross-check, ITS event study, and placebo tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_monthly_price_index(sales_df: pd.DataFrame, min_month_tx: int = 5) -> pd.DataFrame:
    """Compute monthly median and mix-adjusted stratified residential price/sqm index (2019=100)."""
    work = sales_df.copy()
    if "flag_outlier" in work.columns:
        work = work.loc[
            ~work["flag_outlier"] & ~work["flag_non_market"] & ~work["flag_duplicate"]
        ].copy()

    work["dt"] = pd.to_datetime(work["instance_date"])
    work["month"] = work["dt"].dt.to_period("M").dt.to_timestamp()

    # Mix weights from overall (area_name x property_type) composition
    cell_weights = (
        work.groupby(["area_name", "property_type"])["price_per_sqm"].count() / len(work)
    ).to_dict()

    rows: list[dict[str, Any]] = []
    for m_ts, grp in work.groupby("month"):
        if len(grp) < min_month_tx:
            continue
        raw_med = float(grp["price_per_sqm"].median())
        cell_meds = grp.groupby(["area_name", "property_type"])["price_per_sqm"].median()
        w_sum = 0.0
        val_sum = 0.0
        for key, med_v in cell_meds.items():
            w = float(cell_weights.get(key, 0.0))
            val_sum += w * float(med_v)
            w_sum += w
        strat_ppsqm = val_sum / w_sum if w_sum > 0 else raw_med
        rows.append(
            {
                "month": pd.Timestamp(str(m_ts)).strftime("%Y-%m-%d"),
                "tx_count": int(len(grp)),
                "median_ppsqm": round(raw_med, 1),
                "stratified_ppsqm": round(strat_ppsqm, 1),
            }
        )

    idx_df = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    if idx_df.empty:
        return idx_df

    # Anchor base = 2019 mean (or first 12 months if 2019 missing)
    mask_2019 = idx_df["month"].str.startswith("2019-")
    base_val = float(
        idx_df.loc[mask_2019, "stratified_ppsqm"].mean()
        if mask_2019.any()
        else idx_df["stratified_ppsqm"].head(12).mean()
    )
    base_raw = float(
        idx_df.loc[mask_2019, "median_ppsqm"].mean()
        if mask_2019.any()
        else idx_df["median_ppsqm"].head(12).mean()
    )
    idx_df["computed_index"] = np.round(100.0 * idx_df["stratified_ppsqm"] / max(base_val, 1.0), 2)
    idx_df["raw_median_index"] = np.round(100.0 * idx_df["median_ppsqm"] / max(base_raw, 1.0), 2)
    return idx_df


def _run_its_regression(
    monthly_df: pd.DataFrame, event_month_str: str, pre_m: int = 12, post_m: int = 12
) -> dict[str, Any] | None:
    """Segmented linear regression: log(index) ~ t_rel + post + (t_rel * post) over [-pre_m, +post_m]."""
    work = monthly_df.copy()
    work["m_dt"] = pd.to_datetime(work["month"])
    ev_dt = pd.Timestamp(event_month_str).to_period("M").to_timestamp()
    work["t_rel"] = (
        (work["m_dt"].dt.year - ev_dt.year) * 12 + (work["m_dt"].dt.month - ev_dt.month)
    ).astype(int)

    win = work.loc[(work["t_rel"] >= -pre_m) & (work["t_rel"] <= post_m)].copy()
    n_pre = int((win["t_rel"] < 0).sum())
    n_post = int((win["t_rel"] >= 0).sum())
    if n_pre < 4 or n_post < 4:
        return None

    win["post"] = (win["t_rel"] >= 0).astype(float)
    win["post_t"] = win["post"] * win["t_rel"].astype(float)
    win["log_idx"] = np.log(pd.to_numeric(win["computed_index"]).clip(lower=1.0))

    X = sm.add_constant(win[["t_rel", "post", "post_t"]].to_numpy(dtype=float))
    fit = sm.OLS(win["log_idx"].to_numpy(dtype=float), X).fit(cov_type="HC1")

    step_coef = float(fit.params[2])
    step_p = float(fit.pvalues[2])
    slope_chg = float(fit.params[3])
    slope_p = float(fit.pvalues[3])
    return {
        "n_months": int(len(win)),
        "level_step_pct": round(float((np.exp(step_coef) - 1.0) * 100.0), 2),
        "level_step_p_value": round(step_p, 4),
        "monthly_slope_change_pct": round(float((np.exp(slope_chg) - 1.0) * 100.0), 2),
        "slope_change_p_value": round(slope_p, 4),
    }


def analyze_market_events(
    sales_df: pd.DataFrame,
    official_index_df: pd.DataFrame,
    events_csv_path: Path,
) -> dict[str, Any]:
    """Compare computed monthly index against DLD index and quantify curated event ITS + placebos."""
    idx_df = build_monthly_price_index(sales_df)
    events_df = pd.read_csv(events_csv_path) if events_csv_path.exists() else pd.DataFrame()

    # Cross-check against official/fallback DLD index
    idx_check: dict[str, Any] = {"available": False}
    if not idx_df.empty and not official_index_df.empty:
        off = official_index_df.copy()
        off["month"] = (
            pd.DatetimeIndex(pd.to_datetime(pd.Series(off["month"], dtype="string")))
            .to_period("M")
            .to_timestamp()
            .strftime("%Y-%m-%d")
        )
        off_agg = off.groupby("month", as_index=False).agg(index_value=("index_value", "mean"))
        merged = idx_df.merge(off_agg, on="month", how="inner")
        if len(merged) >= 6:
            corr = float(np.corrcoef(merged["computed_index"], merged["index_value"])[0, 1])
            mae = float(np.mean(np.abs(merged["computed_index"] - merged["index_value"])))
            idx_check = {
                "available": True,
                "overlap_months": int(len(merged)),
                "pearson_corr": round(corr, 4),
                "mean_abs_index_gap": round(mae, 2),
            }
        idx_df = idx_df.merge(
            off_agg.rename(columns={"index_value": "official_dld_index"}), on="month", how="left"
        )

    event_results: list[dict[str, Any]] = []
    if not idx_df.empty and not events_df.empty:
        for _, row in events_df.iterrows():
            ev_dt = pd.Timestamp(str(row["start_date"]))
            its = _run_its_regression(idx_df, ev_dt.strftime("%Y-%m-%d"), pre_m=12, post_m=12)
            # Placebo date strictly 6 months BEFORE actual event, restricting window to pre-event (-14m..-1m)
            placebo_dt = ev_dt - pd.DateOffset(months=6)
            pre_only = idx_df.loc[pd.to_datetime(idx_df["month"]) < ev_dt].copy()
            placebo = _run_its_regression(
                pre_only, placebo_dt.strftime("%Y-%m-%d"), pre_m=8, post_m=5
            )
            event_results.append(
                {
                    "event_id": str(row["event_id"]),
                    "title": str(row["title"]),
                    "start_date": str(row["start_date"]),
                    "end_date": str(row["end_date"]),
                    "category": str(row["category"]),
                    "source_url": str(row["source_url"]),
                    "its": its,
                    "placebo_minus_6m": placebo,
                }
            )

    return {
        "methodology": "Quality-mix-stratified monthly index (2019=100) + Interrupted Time Series (HC1 OLS) + pre-event placebo tests",
        "caveat": (
            "Descriptive time-series segmentation cannot separate concurrent global rate cycles, "
            "fx/wealth inflows, or supply handovers from policy announcements. Significant placebo breaks "
            "indicate pre-existing momentum rather than a clean discontinuity."
        ),
        "official_index_sanity_check": idx_check,
        "monthly_series": idx_df.to_dict(orient="records"),
        "events_analysis": event_results,
    }
