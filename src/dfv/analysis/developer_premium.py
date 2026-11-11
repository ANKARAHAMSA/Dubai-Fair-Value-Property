"""Off-plan vs Ready 'Developer Premium' hedonic analysis, matching, and time-to-catch-up."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dfv.analysis._stats import fit_fe_ols


def _build_hedonic_covariates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    covs = pd.DataFrame(index=work.index)
    covs["is_offplan"] = pd.to_numeric(work["is_offplan"], errors="coerce").fillna(0.0)
    covs["log_area_sqm"] = np.log(pd.to_numeric(work["area_sqm"], errors="coerce").clip(lower=1.0))
    covs["is_villa"] = (work["property_type"].astype(str) == "Villa").astype(float)
    rooms_num = pd.to_numeric(work["rooms"], errors="coerce").fillna(1).astype(int).clip(0, 6)
    for r in range(1, 7):
        covs[f"rooms_{r}"] = (rooms_num == r).astype(float)
    return covs


def compute_matched_offplan_comparison(
    df: pd.DataFrame,
    min_per_side: int = 10,
) -> dict[str, Any]:
    """Compare off-plan vs ready sales matched within exact (area x type x rooms x year) cells."""
    work = df.copy()
    dt = pd.to_datetime(work["instance_date"])
    work["year"] = dt.dt.year
    grp_cols = ["area_name", "property_type", "rooms", "year"]

    grouped = work.groupby(grp_cols)
    diffs_log: list[float] = []
    weights: list[int] = []
    matched_cells = 0

    for _, grp in grouped:
        off = grp.loc[grp["is_offplan"] == 1, "log_price_per_sqm"]
        rdy = grp.loc[grp["is_offplan"] == 0, "log_price_per_sqm"]
        if len(off) >= min_per_side and len(rdy) >= min_per_side:
            matched_cells += 1
            diffs_log.append(float(off.median() - rdy.median()))
            weights.append(int(min(len(off), len(rdy))))

    if not diffs_log:
        # Relax cell matching threshold slightly for tiny datasets
        for _, grp in grouped:
            off = grp.loc[grp["is_offplan"] == 1, "log_price_per_sqm"]
            rdy = grp.loc[grp["is_offplan"] == 0, "log_price_per_sqm"]
            if len(off) >= 3 and len(rdy) >= 3:
                matched_cells += 1
                diffs_log.append(float(off.median() - rdy.median()))
                weights.append(int(min(len(off), len(rdy))))

    if not diffs_log:
        return {
            "matched_cells": 0,
            "matched_pct_premium": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }

    arr = np.asarray(diffs_log, dtype=float)
    w = np.asarray(weights, dtype=float)
    weighted_mean = float(np.average(arr, weights=w))
    se = float(np.sqrt(np.average((arr - weighted_mean) ** 2, weights=w) / max(len(arr), 1)))
    return {
        "matched_cells": int(matched_cells),
        "matched_pct_premium": round(float((np.exp(weighted_mean) - 1.0) * 100.0), 2),
        "ci_low_pct": round(float((np.exp(weighted_mean - 1.96 * se) - 1.0) * 100.0), 2),
        "ci_high_pct": round(float((np.exp(weighted_mean + 1.96 * se) - 1.0) * 100.0), 2),
    }


def compute_time_to_catch_up(
    df: pd.DataFrame,
    min_cohort_rows: int = 10,
) -> list[dict[str, Any]]:
    """Estimate months until later ready resales in the same area reach an earlier off-plan launch cohort's price/sqm."""
    work = df.copy()
    work["dt"] = pd.to_datetime(work["instance_date"])
    work["quarter_start"] = work["dt"].dt.to_period("Q").dt.start_time

    rows: list[dict[str, Any]] = []
    for area, grp in work.groupby("area_name"):
        off_q = (
            grp.loc[grp["is_offplan"] == 1]
            .groupby("quarter_start")["price_per_sqm"]
            .agg(["median", "count"])
        )
        rdy_q = (
            grp.loc[grp["is_offplan"] == 0]
            .groupby("quarter_start")["price_per_sqm"]
            .agg(["median", "count"])
        )
        valid_off = off_q.loc[off_q["count"] >= min_cohort_rows]
        valid_rdy = rdy_q.loc[rdy_q["count"] >= min_cohort_rows].sort_index()
        if valid_off.empty or valid_rdy.empty:
            continue

        catch_months: list[float] = []
        caught_cohorts = 0
        total_cohorts = 0
        for q_start, row in valid_off.iterrows():
            entry_p = float(row["median"])
            future_rdy = valid_rdy.loc[valid_rdy.index >= q_start]
            if len(future_rdy) < 2:
                continue
            total_cohorts += 1
            hit = future_rdy.loc[future_rdy["median"] >= entry_p]
            if not hit.empty:
                caught_cohorts += 1
                months = float((hit.index[0] - q_start).days / 30.4375)
                catch_months.append(max(months, 0.0))

        if total_cohorts >= 2:
            rows.append(
                {
                    "area_name": str(area),
                    "evaluated_cohorts": int(total_cohorts),
                    "cohorts_caught_up": int(caught_cohorts),
                    "median_catchup_months": round(float(np.median(catch_months)), 1)
                    if catch_months
                    else None,
                    "status": "Observed catch-up"
                    if catch_months
                    else "Ready prices remained below cohort entry within window",
                }
            )
    return sorted(
        rows,
        key=lambda r: (r["median_catchup_months"] is None, r["median_catchup_months"] or 999.0),
    )


def analyze_developer_premium(
    df: pd.DataFrame,
    min_group_obs: int = 15,
) -> dict[str, Any]:
    """Estimate overall, area-level, and developer-level off-plan vs ready price associations."""
    work = df.copy()
    if "flag_outlier" in work.columns:
        work = work.loc[
            ~work["flag_outlier"] & ~work["flag_non_market"] & ~work["flag_duplicate"]
        ].copy()
    if "log_price_per_sqm" not in work.columns:
        work["log_price_per_sqm"] = np.log(
            pd.to_numeric(work["price_per_sqm"], errors="coerce").clip(lower=1.0)
        )

    dt = pd.to_datetime(work["instance_date"])
    work["year_quarter"] = dt.dt.to_period("Q").astype(str)

    # 1. Hedonic model A: Area x Quarter fixed effects (controls for community & market timing)
    covs = _build_hedonic_covariates(work)
    res_area_fe, r2_area_fe, n_area_fe = fit_fe_ols(
        work, "log_price_per_sqm", covs, fe_cols=["area_name", "year_quarter"]
    )

    # 2. Hedonic model B: Project + Year-Quarter FE where projects have both off-plan & ready
    # Using (project_name x year) absorbs within-building variation cleanly
    res_proj_fe, r2_proj_fe, n_proj_fe = fit_fe_ols(
        work, "log_price_per_sqm", covs, fe_cols=["project_name", "year_quarter"]
    )

    overall_area = res_area_fe["is_offplan"].to_dict() if "is_offplan" in res_area_fe else {}
    overall_proj = res_proj_fe["is_offplan"].to_dict() if "is_offplan" in res_proj_fe else {}

    # 3. Breakout by Area (minimum-sample enforced for both off-plan and ready)
    area_rows: list[dict[str, Any]] = []
    for area, grp in work.groupby("area_name"):
        n_off = int((grp["is_offplan"] == 1).sum())
        n_rdy = int((grp["is_offplan"] == 0).sum())
        if n_off < min_group_obs or n_rdy < min_group_obs:
            continue
        sub_cov = _build_hedonic_covariates(grp)
        res_a, _, n_a = fit_fe_ols(grp, "log_price_per_sqm", sub_cov, fe_cols=["year_quarter"])
        if "is_offplan" not in res_a:
            continue
        term = res_a["is_offplan"]
        area_rows.append(
            {
                "area_name": str(area),
                "n_offplan": n_off,
                "n_ready": n_rdy,
                "n_used": n_a,
                "pct_premium": round(term.pct_effect, 2),
                "ci_low_pct": round(term.ci_low_pct, 2),
                "ci_high_pct": round(term.ci_high_pct, 2),
                "p_value": round(term.p_value, 5),
                "offplan_median_ppsqm": round(
                    float(grp.loc[grp["is_offplan"] == 1, "price_per_sqm"].median()), 1
                ),
                "ready_median_ppsqm": round(
                    float(grp.loc[grp["is_offplan"] == 0, "price_per_sqm"].median()), 1
                ),
            }
        )
    area_rows.sort(key=lambda r: r["pct_premium"], reverse=True)

    # 4. Breakout by Developer (strictly excluding 'Unknown' and enforcing minimum sample threshold)
    dev_rows: list[dict[str, Any]] = []
    known_dev = work.loc[~work["developer_name"].astype(str).isin(["Unknown", "nan", "None", ""])]
    for dev, grp in known_dev.groupby("developer_name"):
        n_off = int((grp["is_offplan"] == 1).sum())
        n_rdy = int((grp["is_offplan"] == 0).sum())
        if n_off < min_group_obs or n_rdy < min_group_obs:
            continue
        sub_cov = _build_hedonic_covariates(grp)
        res_d, _, n_d = fit_fe_ols(
            grp, "log_price_per_sqm", sub_cov, fe_cols=["area_name", "year_quarter"]
        )
        if "is_offplan" not in res_d:
            continue
        term = res_d["is_offplan"]
        dev_rows.append(
            {
                "developer_name": str(dev),
                "n_offplan": n_off,
                "n_ready": n_rdy,
                "n_used": n_d,
                "pct_premium": round(term.pct_effect, 2),
                "ci_low_pct": round(term.ci_low_pct, 2),
                "ci_high_pct": round(term.ci_high_pct, 2),
                "p_value": round(term.p_value, 5),
            }
        )
    dev_rows.sort(key=lambda r: r["pct_premium"], reverse=True)

    unknown_dev_share = round(
        float(
            work["developer_name"].astype(str).isin(["Unknown", "nan", "None", ""]).mean() * 100.0
        ),
        1,
    )

    matched = compute_matched_offplan_comparison(work, min_per_side=max(3, min_group_obs // 2))
    catchup = compute_time_to_catch_up(work, min_cohort_rows=max(5, min_group_obs // 2))

    return {
        "methodology": "Hedonic OLS with Area x Quarter and Project x Quarter fixed effects (FWL + HC1 SEs)",
        "caveat": (
            "Observational association only; does not adjust for unobserved post-handover payment plans, "
            "launch incentives, view/floor tiers, or snagging risk. Named developers with thin samples "
            f"(<{min_group_obs} off-plan & ready sales each) are excluded."
        ),
        "unknown_developer_pct": unknown_dev_share,
        "overall_area_quarter_fe": {**overall_area, "r2_within": r2_area_fe, "n_obs": n_area_fe},
        "overall_project_quarter_fe": {**overall_proj, "r2_within": r2_proj_fe, "n_obs": n_proj_fe},
        "matched_comparison": matched,
        "by_area": area_rows,
        "by_developer": dev_rows,
        "time_to_catch_up": catchup,
    }
