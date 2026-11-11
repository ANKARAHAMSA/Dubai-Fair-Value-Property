"""Rental yield estimation, risk scoring, and budget-based area recommendations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

YIELD_CAVEAT = (
    "Gross rental yield equals annual rent divided by purchase price using trailing 12-month "
    "medians. It ignores DLD transfer fees (4%), Oqood/agency fees, service charges, vacancy, "
    "and maintenance costs."
)


def _winsorize(series: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return s
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lower=float(lo), upper=float(hi))


def compute_yield_and_risk(
    sales_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    min_sales: int = 10,
    min_rents: int = 8,
) -> dict[str, Any]:
    """Compute segment and area-level trailing 12m gross yields, volatility/liquidity risk, and ranking."""
    sales = sales_df.copy()
    if "flag_outlier" in sales.columns:
        sales = sales.loc[
            ~sales["flag_outlier"] & ~sales["flag_non_market"] & ~sales["flag_duplicate"]
        ].copy()

    sales["dt"] = pd.to_datetime(sales["instance_date"])
    max_s_dt = sales["dt"].max()
    recent_sales = sales.loc[sales["dt"] >= (max_s_dt - pd.Timedelta(days=365))].copy()
    if len(recent_sales) < 50:
        recent_sales = sales.copy()

    rents = rent_df.copy()
    rents["dt"] = pd.to_datetime(rents["contract_start_date"])
    max_r_dt = rents["dt"].max()
    recent_rents = rents.loc[rents["dt"] >= (max_r_dt - pd.Timedelta(days=365))].copy()
    if len(recent_rents) < 50:
        recent_rents = rents.copy()

    # Area-level price volatility (monthly log return std over trailing 24m)
    vol_window = sales.loc[sales["dt"] >= (max_s_dt - pd.Timedelta(days=730))].copy()
    vol_window["ym"] = vol_window["dt"].dt.to_period("M")
    area_vol: dict[str, float] = {}
    for area, grp in vol_window.groupby("area_name"):
        m_med = grp.groupby("ym")["price_per_sqm"].median().sort_index()
        if len(m_med) >= 4:
            log_ret = np.diff(np.log(np.maximum(m_med.to_numpy(dtype=float), 1.0)))
            area_vol[str(area)] = float(np.std(log_ret))
        else:
            area_vol[str(area)] = 0.08  # conservative default for sparse history

    # Segment-level yields: area_name x property_type x rooms
    seg_rows: list[dict[str, Any]] = []
    min_s_eff = min(min_sales, max(3, len(recent_sales) // 150))
    min_r_eff = min(min_rents, max(3, len(recent_rents) // 150))

    rent_by_seg = {k: v for k, v in recent_rents.groupby(["area_name", "property_type", "rooms"])}
    rent_by_area = {k: v for k, v in recent_rents.groupby("area_name")}

    for (area, ptype, rooms), s_grp in recent_sales.groupby(
        ["area_name", "property_type", "rooms"]
    ):  # type: ignore[misc]
        n_s = int(len(s_grp))
        if n_s < min_s_eff:
            continue

        r_grp = rent_by_seg.get((area, ptype, rooms))
        fallback_level = "segment"
        if r_grp is None or len(r_grp) < min_r_eff:
            r_area = rent_by_area.get(area)
            if r_area is None or len(r_area) < min_r_eff:
                continue
            r_grp = r_area
            fallback_level = "area_fallback"

        # Prefer New contracts where sufficient count exists
        if "is_renewal" in r_grp.columns:
            new_r = r_grp.loc[r_grp["is_renewal"] == 0]
            new_share = float((r_grp["is_renewal"] == 0).mean())
            if len(new_r) >= max(3, min_r_eff // 2):
                r_use = new_r
                used_new_only = True
            else:
                r_use = r_grp
                used_new_only = False
        else:
            r_use = r_grp
            new_share = 0.5
            used_new_only = False

        s_ppsqm_w = _winsorize(s_grp["price_per_sqm"])
        r_ppsqm_w = _winsorize(r_use["rent_per_sqm"])
        s_price_w = _winsorize(s_grp["actual_worth"])
        if s_ppsqm_w.empty or r_ppsqm_w.empty:
            continue

        med_sale_ppsqm = float(s_ppsqm_w.median())
        med_rent_ppsqm = float(r_ppsqm_w.median())
        med_price_aed = float(s_price_w.median())
        gross_yield = med_rent_ppsqm / max(med_sale_ppsqm, 1.0)
        # Sanity clip for extreme unmatched cells
        if gross_yield < 0.01 or gross_yield > 0.18:
            continue

        vol_m = area_vol.get(str(area), 0.06)
        area_tx_12m = int(len(recent_sales.loc[recent_sales["area_name"] == area]))
        # Risk score components in [0, 100] (lower = safer / more stable & liquid)
        vol_score = min(100.0, (vol_m / 0.12) * 100.0)
        liq_score = max(0.0, 100.0 * (1.0 - np.log1p(area_tx_12m) / np.log1p(5000.0)))
        qual_penalty = (
            0.0
            if (fallback_level == "segment" and n_s >= 25 and len(r_use) >= 15)
            else (25.0 if fallback_level == "segment" else 45.0)
        )
        risk_score = float(
            np.clip(0.45 * vol_score + 0.35 * liq_score + 0.20 * qual_penalty, 5.0, 95.0)
        )
        conf = (
            "High"
            if (fallback_level == "segment" and n_s >= 25 and len(r_use) >= 15)
            else ("Medium" if n_s >= 10 else "Low")
        )

        yield_pct = gross_yield * 100.0
        yield_per_risk = yield_pct / (1.0 + risk_score / 30.0)

        seg_rows.append(
            {
                "area_name": str(area),
                "property_type": str(ptype),
                "rooms": int(rooms),
                "median_price_aed": round(med_price_aed, 0),
                "median_sale_ppsqm": round(med_sale_ppsqm, 1),
                "median_rent_ppsqm": round(med_rent_ppsqm, 1),
                "gross_yield_pct": round(yield_pct, 2),
                "monthly_price_vol": round(vol_m, 4),
                "sales_12m_count": n_s,
                "rents_12m_count": int(len(r_use)),
                "new_contract_share_pct": round(new_share * 100.0, 1),
                "used_new_contracts_only": used_new_only,
                "match_level": fallback_level,
                "confidence": conf,
                "risk_score": round(risk_score, 1),
                "yield_per_risk_score": round(yield_per_risk, 3),
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    if not seg_df.empty:
        seg_df = seg_df.sort_values("yield_per_risk_score", ascending=False).reset_index(drop=True)

    # Aggregate area-level ranking
    area_rows: list[dict[str, Any]] = []
    if not seg_df.empty:
        for area, grp in seg_df.groupby("area_name"):
            w = grp["sales_12m_count"].to_numpy(dtype=float)
            gy = float(np.average(grp["gross_yield_pct"], weights=w))
            rk = float(np.average(grp["risk_score"], weights=w))
            mp = float(np.average(grp["median_price_aed"], weights=w))
            ms_p = float(np.average(grp["median_sale_ppsqm"], weights=w))
            mr_p = float(np.average(grp["median_rent_ppsqm"], weights=w))
            tot_s = int(grp["sales_12m_count"].sum())
            tot_r = int(grp["rents_12m_count"].sum())
            area_rows.append(
                {
                    "area_name": str(area),
                    "gross_yield_pct": round(gy, 2),
                    "risk_score": round(rk, 1),
                    "yield_per_risk_score": round(gy / (1.0 + rk / 30.0), 3),
                    "median_price_aed": round(mp, 0),
                    "median_sale_ppsqm": round(ms_p, 1),
                    "median_rent_ppsqm": round(mr_p, 1),
                    "sales_12m_count": tot_s,
                    "rents_12m_count": tot_r,
                    "confidence": "High"
                    if tot_s >= 40 and tot_r >= 20
                    else ("Medium" if tot_s >= 15 else "Low"),
                }
            )
    area_df = pd.DataFrame(area_rows)
    if not area_df.empty:
        area_df = area_df.sort_values("yield_per_risk_score", ascending=False).reset_index(
            drop=True
        )

    return {
        "caveat": YIELD_CAVEAT,
        "window_days": 365,
        "as_of_date": max_s_dt.strftime("%Y-%m-%d"),
        "area_ranking": area_df.to_dict(orient="records"),
        "segment_ranking": seg_df.to_dict(orient="records"),
    }


def recommend_areas_for_budget(
    segment_ranking: list[dict[str, Any]],
    budget_aed: float,
    rooms: int | None = None,
    property_type: str | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return top-k areas ranked by yield_per_risk_score within the buyer's budget."""
    if not segment_ranking:
        return []
    df = pd.DataFrame(segment_ranking)
    if rooms is not None:
        df = df.loc[df["rooms"] == int(rooms)]
    if property_type is not None:
        df = df.loc[df["property_type"].str.lower() == property_type.lower()]
    if df.empty:
        return []

    affordable = df.loc[df["median_price_aed"] <= float(budget_aed)].copy()
    if affordable.empty:
        return []

    # Collapse per area picking the best yield-per-risk unit within budget
    best_per_area = (
        affordable.sort_values("yield_per_risk_score", ascending=False)
        .groupby("area_name", as_index=False)
        .first()
        .sort_values("yield_per_risk_score", ascending=False)
        .head(top_k)
    )
    return [{str(k): v for k, v in rec.items()} for rec in best_per_area.to_dict(orient="records")]
