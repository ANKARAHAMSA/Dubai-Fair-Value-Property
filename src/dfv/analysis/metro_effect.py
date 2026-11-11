"""Metro-proximity hedonic premium (sales & rents) and Route 2020 opening event study."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dfv.analysis._stats import fit_fe_ols, fit_twoway_fe_ols


def _distance_band_covariates(
    df: pd.DataFrame, dist_col: str, extra_offplan: bool = True
) -> pd.DataFrame:
    work = df.copy()
    d = pd.to_numeric(work[dist_col], errors="coerce").fillna(5000.0)
    covs = pd.DataFrame(index=work.index)
    # Disjoint distance bands; reference group is >= 2000m
    covs["band_lt500m"] = (d < 500.0).astype(float)
    covs["band_500_1000m"] = ((d >= 500.0) & (d < 1000.0)).astype(float)
    covs["band_1000_2000m"] = ((d >= 1000.0) & (d < 2000.0)).astype(float)

    covs["log_area_sqm"] = np.log(pd.to_numeric(work["area_sqm"], errors="coerce").clip(lower=1.0))
    covs["is_villa"] = (work["property_type"].astype(str) == "Villa").astype(float)
    if extra_offplan and "is_offplan" in work.columns:
        covs["is_offplan"] = pd.to_numeric(work["is_offplan"], errors="coerce").fillna(0.0)
    if "is_renewal" in work.columns:
        covs["is_renewal"] = pd.to_numeric(work["is_renewal"], errors="coerce").fillna(0.0)
    rooms_num = pd.to_numeric(work["rooms"], errors="coerce").fillna(1).astype(int).clip(0, 6)
    for r in range(1, 7):
        covs[f"rooms_{r}"] = (rooms_num == r).astype(float)
    return covs


def run_route2020_event_study(
    sales_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    treatment_radius_m: float = 1200.0,
) -> dict[str, Any]:
    """Difference-in-differences & dynamic event study around documented Route 2020 station openings.

    Uses additive two-way fixed effects (area + year-quarter) so area-level post-opening timing
    is not absorbed by interacted Area x Quarter FEs. Only stations with non-null opening_date are used.
    """
    r2020 = stations_df.copy()
    if "is_route_2020" in r2020.columns:
        r2020 = r2020.loc[r2020["is_route_2020"] == True]  # noqa: E712
    r2020 = r2020.loc[
        r2020["opening_date"].notna() & (r2020["opening_date"].astype(str) != "")
    ].copy()
    if r2020.empty:
        return {"available": False, "reason": "No Route 2020 stations with known opening_date"}

    work = sales_df.copy()
    if "flag_outlier" in work.columns:
        work = work.loc[
            ~work["flag_outlier"] & ~work["flag_non_market"] & ~work["flag_duplicate"]
        ].copy()

    if {"lat", "lon"}.issubset(work.columns) and {"lat", "lon"}.issubset(r2020.columns):
        tx_lat = np.radians(work["lat"].to_numpy(dtype=float))[:, None]
        tx_lon = np.radians(work["lon"].to_numpy(dtype=float))[:, None]
        st_lat = np.radians(r2020["lat"].to_numpy(dtype=float))[None, :]
        st_lon = np.radians(r2020["lon"].to_numpy(dtype=float))[None, :]
        dlat = st_lat - tx_lat
        dlon = st_lon - tx_lon
        a = np.sin(dlat / 2.0) ** 2 + np.cos(tx_lat) * np.cos(st_lat) * np.sin(dlon / 2.0) ** 2
        dists = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        min_idx = np.argmin(dists, axis=1)
        min_dist = dists[np.arange(len(work)), min_idx]
        open_dates = pd.to_datetime(r2020["opening_date"].iloc[min_idx].to_numpy())
    else:
        return {"available": False, "reason": "Coordinates missing for Route 2020 event study"}

    work["r2020_dist_m"] = min_dist
    work["tx_dt"] = pd.to_datetime(work["instance_date"])
    work["open_dt"] = open_dates
    work["year_quarter"] = work["tx_dt"].dt.to_period("Q").astype(str)
    work["treated"] = (work["r2020_dist_m"] < treatment_radius_m).astype(int)
    work["rel_q"] = np.floor((work["tx_dt"] - work["open_dt"]).dt.days / 91.25).astype(int)

    # Restrict window to [-6, +6] quarters around station opening for clean DiD comparison
    window = work.loc[(work["rel_q"] >= -6) & (work["rel_q"] <= 6)].copy()
    if window["treated"].sum() < 20 or (1 - window["treated"]).sum() < 20:
        return {
            "available": False,
            "reason": f"Insufficient treated rows ({int(window['treated'].sum())}) inside Route 2020 window",
        }

    if "log_price_per_sqm" not in window.columns:
        window["log_price_per_sqm"] = np.log(pd.to_numeric(window["price_per_sqm"]).clip(lower=1.0))

    # Additive two-way FE DiD (treated_post + treated [if within-area variance] + area FE + quarter FE)
    cov_did = pd.DataFrame(index=window.index)
    cov_did["treated_post"] = (window["treated"] * (window["rel_q"] >= 0)).astype(float)
    cov_did["treated"] = window["treated"].astype(float)
    cov_did["log_area_sqm"] = np.log(
        pd.to_numeric(window["area_sqm"], errors="coerce").clip(lower=1.0)
    )
    cov_did["is_villa"] = (window["property_type"].astype(str) == "Villa").astype(float)
    cov_did["is_offplan"] = pd.to_numeric(window["is_offplan"], errors="coerce").fillna(0.0)

    res_did, r2_did, n_did = fit_twoway_fe_ols(
        window, "log_price_per_sqm", cov_did, fe_col1="area_name", fe_col2="year_quarter"
    )
    did_post = res_did.get("treated_post")

    # Dynamic event-study leads & lags relative to rel_q == -1
    ev_cov = pd.DataFrame(index=window.index)
    rel_bins = [-4, -3, -2, 0, 1, 2, 3, 4]
    for qk in rel_bins:
        if qk == -4:
            ev_cov[f"lead_lag_{qk}"] = ((window["treated"] == 1) & (window["rel_q"] <= -4)).astype(
                float
            )
        elif qk == 4:
            ev_cov[f"lead_lag_{qk}"] = ((window["treated"] == 1) & (window["rel_q"] >= 4)).astype(
                float
            )
        else:
            ev_cov[f"lead_lag_{qk}"] = ((window["treated"] == 1) & (window["rel_q"] == qk)).astype(
                float
            )
    ev_cov["treated"] = cov_did["treated"]
    ev_cov["log_area_sqm"] = cov_did["log_area_sqm"]
    ev_cov["is_villa"] = cov_did["is_villa"]
    ev_cov["is_offplan"] = cov_did["is_offplan"]

    res_ev, _, n_ev = fit_twoway_fe_ols(
        window, "log_price_per_sqm", ev_cov, fe_col1="area_name", fe_col2="year_quarter"
    )
    event_curve: list[dict[str, Any]] = []
    pre_pvals: list[float] = []
    for qk in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        if qk == -1:
            event_curve.append(
                {
                    "rel_quarter": -1,
                    "pct_effect": 0.0,
                    "ci_low_pct": 0.0,
                    "ci_high_pct": 0.0,
                    "p_value": 1.0,
                }
            )
        elif f"lead_lag_{qk}" in res_ev:
            t = res_ev[f"lead_lag_{qk}"]
            event_curve.append(
                {
                    "rel_quarter": qk,
                    "pct_effect": round(t.pct_effect, 2),
                    "ci_low_pct": round(t.ci_low_pct, 2),
                    "ci_high_pct": round(t.ci_high_pct, 2),
                    "p_value": round(t.p_value, 4),
                }
            )
            if qk <= -2:
                pre_pvals.append(t.p_value)

    pre_trend_ok = bool(all(p > 0.05 for p in pre_pvals)) if pre_pvals else False
    return {
        "available": True,
        "fe_specification": "Two-way additive FE (Area + Year-Quarter)",
        "stations_evaluated": sorted(r2020["station_name"].astype(str).tolist()),
        "treated_rows": int(window["treated"].sum()),
        "control_rows": int((1 - window["treated"]).sum()),
        "did_post_effect": did_post.to_dict() if did_post else None,
        "r2_within": r2_did,
        "n_obs": n_did or n_ev,
        "pre_trend_parallel_pass": pre_trend_ok,
        "event_study_curve": event_curve,
    }


def analyze_metro_effect(
    sales_features_df: pd.DataFrame,
    rent_contracts_df: pd.DataFrame,
    stations_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run hedonic metro distance-band analysis on sales and rents plus Route 2020 event study."""
    sales = sales_features_df.copy()
    if "flag_outlier" in sales.columns:
        sales = sales.loc[
            ~sales["flag_outlier"] & ~sales["flag_non_market"] & ~sales["flag_duplicate"]
        ].copy()
    if "log_price_per_sqm" not in sales.columns:
        sales["log_price_per_sqm"] = np.log(pd.to_numeric(sales["price_per_sqm"]).clip(lower=1.0))

    dist_col = (
        "dist_nearest_metro_m" if "dist_nearest_metro_m" in sales.columns else "metro_distance_m"
    )
    sales["year_quarter"] = pd.to_datetime(sales["instance_date"]).dt.to_period("Q").astype(str)

    # Primary Sales FE: Area x Quarter (identifiable when within-area coordinate variation exists)
    cov_s = _distance_band_covariates(sales, dist_col=dist_col, extra_offplan=True)
    res_sales, r2_sales, n_sales = fit_fe_ols(
        sales, "log_price_per_sqm", cov_s, fe_cols=["area_name", "year_quarter"]
    )
    band_keys = ("band_lt500m", "band_500_1000m", "band_1000_2000m")
    sales_bands = {k: res_sales[k].to_dict() for k in band_keys if k in res_sales}

    within_area_identifiable = len(sales_bands) > 0
    cross_area_bands: dict[str, Any] = {}
    if not within_area_identifiable:
        res_cross, _, _ = fit_fe_ols(sales, "log_price_per_sqm", cov_s, fe_cols=["year_quarter"])
        cross_area_bands = {k: res_cross[k].to_dict() for k in band_keys if k in res_cross}

    # Rents FE: Area x Quarter
    rents = rent_contracts_df.copy()
    rent_bands: dict[str, Any] = {}
    r2_rents = 0.0
    n_rents = 0
    if not rents.empty:
        if "metro_distance_m" not in rents.columns and {"lat", "lon"}.issubset(stations_df.columns):
            area_d = sales.groupby("area_name")[dist_col].median().to_dict()
            rents["metro_distance_m"] = rents["area_name"].map(area_d).fillna(1500.0)
        r_dist_col = "metro_distance_m" if "metro_distance_m" in rents.columns else dist_col
        if r_dist_col in rents.columns:
            rents["log_rent_per_sqm"] = np.log(
                pd.to_numeric(rents["rent_per_sqm"], errors="coerce").clip(lower=1.0)
            )
            rents["year_quarter"] = (
                pd.to_datetime(rents["contract_start_date"]).dt.to_period("Q").astype(str)
            )
            cov_r = _distance_band_covariates(rents, dist_col=r_dist_col, extra_offplan=False)
            res_rents, r2_rents, n_rents = fit_fe_ols(
                rents, "log_rent_per_sqm", cov_r, fe_cols=["area_name", "year_quarter"]
            )
            rent_bands = {k: res_rents[k].to_dict() for k in band_keys if k in res_rents}

    band_counts = {
        "<500m": int((pd.to_numeric(sales[dist_col], errors="coerce") < 500.0).sum()),
        "500-1000m": int(
            (
                (pd.to_numeric(sales[dist_col], errors="coerce") >= 500.0)
                & (pd.to_numeric(sales[dist_col], errors="coerce") < 1000.0)
            ).sum()
        ),
        "1000-2000m": int(
            (
                (pd.to_numeric(sales[dist_col], errors="coerce") >= 1000.0)
                & (pd.to_numeric(sales[dist_col], errors="coerce") < 2000.0)
            ).sum()
        ),
        ">=2000m": int((pd.to_numeric(sales[dist_col], errors="coerce") >= 2000.0).sum()),
    }

    event_study = run_route2020_event_study(sales, stations_df)

    return {
        "methodology": "Fixed-effects hedonic OLS (Area x Quarter FE, reference >= 2 km) + Two-way FE Route 2020 event study",
        "caveat": (
            "Distance-band premiums represent observational within-area associations rather than isolated causal "
            "effects (proximity correlates with building age, density, and retail clusters). Real DLD fallback "
            "uses area-centroid coordinates, so within-area sales distance variance is absent until project-level "
            "coordinates/dld_projects-open are supplied."
        ),
        "within_area_identifiable": within_area_identifiable,
        "sales_distance_bands": sales_bands,
        "sales_cross_area_bands": cross_area_bands,
        "sales_r2_within": r2_sales,
        "sales_n_obs": n_sales,
        "sales_band_counts": band_counts,
        "rents_distance_bands": rent_bands,
        "rents_r2_within": r2_rents,
        "rents_n_obs": n_rents,
        "route_2020_event_study": event_study,
    }
