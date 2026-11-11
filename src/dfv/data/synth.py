"""Realistic synthetic DLD-like dataset generator with planted ground-truth effects."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from dfv.config import SQFT_PER_SQM


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """Planted structural parameters for synthetic DLD data and recovery tests."""

    seed: int = 42
    metro_lt500_log_effect: float = 0.14
    metro_500_1000_log_effect: float = 0.08
    metro_1000_2000_log_effect: float = 0.03
    offplan_log_effect: float = 0.06
    route_2020_opening_date: str = "2021-01-01"
    route_2020_post_open_log_effect: float = 0.07
    area_size_elasticity: float = -0.08
    villa_log_effect: float = 0.09
    noise_std: float = 0.085


@dataclass
class SyntheticBundle:
    """Collection of synthetic DLD/RTA tables and planted ground-truth parameters."""

    transactions: pd.DataFrame
    rent_contracts: pd.DataFrame
    projects: pd.DataFrame
    metro_stations: pd.DataFrame
    sale_index: pd.DataFrame
    ground_truth: SyntheticGroundTruth


AREA_SPECS: dict[str, dict[str, Any]] = {
    "Palm Jumeirah": {
        "lat": 25.1124,
        "lon": 55.1390,
        "base_ppsqm": 22500.0,
        "yield": 0.050,
        "villa_share": 0.30,
        "metro": "Nakheel",
    },
    "Downtown Dubai": {
        "lat": 25.1972,
        "lon": 55.2744,
        "base_ppsqm": 21500.0,
        "yield": 0.052,
        "villa_share": 0.05,
        "metro": "Burj Khalifa / Dubai Mall",
    },
    "Emaar Beachfront": {
        "lat": 25.0935,
        "lon": 55.1375,
        "base_ppsqm": 23000.0,
        "yield": 0.050,
        "villa_share": 0.02,
        "metro": "Al Khail",
    },
    "City Walk": {
        "lat": 25.2078,
        "lon": 55.2623,
        "base_ppsqm": 19500.0,
        "yield": 0.054,
        "villa_share": 0.05,
        "metro": "Burj Khalifa / Dubai Mall",
    },
    "Dubai Marina": {
        "lat": 25.0805,
        "lon": 55.1403,
        "base_ppsqm": 16500.0,
        "yield": 0.063,
        "villa_share": 0.04,
        "metro": "DMCC",
    },
    "Dubai Creek Harbour": {
        "lat": 25.2044,
        "lon": 55.3441,
        "base_ppsqm": 16000.0,
        "yield": 0.058,
        "villa_share": 0.06,
        "metro": "Creek",
    },
    "Business Bay": {
        "lat": 25.1865,
        "lon": 55.2721,
        "base_ppsqm": 15500.0,
        "yield": 0.065,
        "villa_share": 0.03,
        "metro": "Business Bay",
    },
    "Dubai Hills Estate": {
        "lat": 25.1072,
        "lon": 55.2435,
        "base_ppsqm": 15000.0,
        "yield": 0.058,
        "villa_share": 0.45,
        "metro": "Equiti",
    },
    "Arabian Ranches": {
        "lat": 25.0589,
        "lon": 55.2636,
        "base_ppsqm": 13000.0,
        "yield": 0.055,
        "villa_share": 0.95,
        "metro": "Mall of the Emirates",
    },
    "Jumeirah Lake Towers": {
        "lat": 25.0695,
        "lon": 55.1436,
        "base_ppsqm": 12500.0,
        "yield": 0.070,
        "villa_share": 0.02,
        "metro": "Sobha Realty",
    },
    "Damac Hills": {
        "lat": 25.0255,
        "lon": 55.2485,
        "base_ppsqm": 10500.0,
        "yield": 0.066,
        "villa_share": 0.65,
        "metro": "Mall of the Emirates",
    },
    "Al Furjan": {
        "lat": 25.0279,
        "lon": 55.1433,
        "base_ppsqm": 10200.0,
        "yield": 0.071,
        "villa_share": 0.45,
        "metro": "Al Furjan",
    },
    "Jumeirah Village Circle": {
        "lat": 25.0554,
        "lon": 55.2096,
        "base_ppsqm": 9800.0,
        "yield": 0.076,
        "villa_share": 0.18,
        "metro": "Dubai Internet City",
    },
    "Discovery Gardens": {
        "lat": 25.0419,
        "lon": 55.1397,
        "base_ppsqm": 7800.0,
        "yield": 0.082,
        "villa_share": 0.02,
        "metro": "Discovery Gardens",
    },
    "Dubai South": {
        "lat": 24.8962,
        "lon": 55.1614,
        "base_ppsqm": 7200.0,
        "yield": 0.074,
        "villa_share": 0.35,
        "metro": "Expo 2020",
    },
}

AREA_WEIGHTS: dict[str, float] = {
    "Jumeirah Village Circle": 0.14,
    "Business Bay": 0.12,
    "Dubai Marina": 0.11,
    "Downtown Dubai": 0.10,
    "Dubai Hills Estate": 0.08,
    "Dubai Creek Harbour": 0.07,
    "Jumeirah Lake Towers": 0.06,
    "Al Furjan": 0.06,
    "Dubai South": 0.06,
    "Palm Jumeirah": 0.05,
    "Damac Hills": 0.05,
    "Discovery Gardens": 0.04,
    "Arabian Ranches": 0.03,
    "City Walk": 0.015,
    "Emaar Beachfront": 0.015,
}

DEVELOPER_OFFSETS: dict[str, float] = {
    "Emaar Properties": 0.045,
    "Meraas": 0.035,
    "Sobha Realty": 0.025,
    "Nakheel": 0.015,
    "Ellington Properties": 0.020,
    "Select Group": 0.005,
    "Dubai Properties": -0.005,
    "DAMAC Properties": -0.020,
    "Danube Properties": -0.030,
    "Azizi Developments": -0.040,
}

METRO_STATION_ROWS: list[dict[str, Any]] = [
    {
        "station_name": "Burj Khalifa / Dubai Mall",
        "line": "Red",
        "lat": 25.2014,
        "lon": 55.2696,
        "opening_date": "2010-01-04",
        "is_route_2020": False,
    },
    {
        "station_name": "Business Bay",
        "line": "Red",
        "lat": 25.1912,
        "lon": 55.2603,
        "opening_date": "2010-10-15",
        "is_route_2020": False,
    },
    {
        "station_name": "Equiti",
        "line": "Red",
        "lat": 25.1342,
        "lon": 55.2111,
        "opening_date": "2010-10-15",
        "is_route_2020": False,
    },
    {
        "station_name": "Mall of the Emirates",
        "line": "Red",
        "lat": 25.1202,
        "lon": 55.1983,
        "opening_date": "2009-09-09",
        "is_route_2020": False,
    },
    {
        "station_name": "Dubai Internet City",
        "line": "Red",
        "lat": 25.0956,
        "lon": 55.1726,
        "opening_date": "2010-04-30",
        "is_route_2020": False,
    },
    {
        "station_name": "Al Khail",
        "line": "Red",
        "lat": 25.0866,
        "lon": 55.1588,
        "opening_date": "2010-10-15",
        "is_route_2020": False,
    },
    {
        "station_name": "Sobha Realty",
        "line": "Red",
        "lat": 25.0785,
        "lon": 55.1483,
        "opening_date": "2010-04-30",
        "is_route_2020": False,
    },
    {
        "station_name": "DMCC",
        "line": "Red",
        "lat": 25.0677,
        "lon": 55.1387,
        "opening_date": "2010-10-15",
        "is_route_2020": False,
    },
    {
        "station_name": "Nakheel",
        "line": "Palm Monorail",
        "lat": 25.1095,
        "lon": 55.1502,
        "opening_date": "2009-04-30",
        "is_route_2020": False,
    },
    {
        "station_name": "Creek",
        "line": "Green",
        "lat": 25.2194,
        "lon": 55.3358,
        "opening_date": "2014-03-01",
        "is_route_2020": False,
    },
    {
        "station_name": "The Gardens",
        "line": "Red (Route 2020)",
        "lat": 25.0472,
        "lon": 55.1372,
        "opening_date": "2021-01-01",
        "is_route_2020": True,
    },
    {
        "station_name": "Discovery Gardens",
        "line": "Red (Route 2020)",
        "lat": 25.0427,
        "lon": 55.1378,
        "opening_date": "2021-01-01",
        "is_route_2020": True,
    },
    {
        "station_name": "Al Furjan",
        "line": "Red (Route 2020)",
        "lat": 25.0294,
        "lon": 55.1407,
        "opening_date": "2021-01-01",
        "is_route_2020": True,
    },
    {
        "station_name": "Expo 2020",
        "line": "Red (Route 2020)",
        "lat": 24.9628,
        "lon": 55.1456,
        "opening_date": "2021-06-01",
        "is_route_2020": True,
    },
]


def _haversine_meters(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Compute vectorized great-circle distance in meters."""
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = math.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def market_cycle_log_effect(dates: pd.Series) -> np.ndarray:
    """Smooth piecewise Dubai-like market cycle index in log space anchored at 2019 = 0."""
    ts = pd.to_datetime(dates)
    years = ts.dt.year + (ts.dt.dayofyear - 1) / 365.25
    y = years.to_numpy(dtype=float)

    # 2015 -> 2019 softening, 2020 dip, 2021-2026 rally
    base = np.where(
        y < 2019.0,
        0.16 * (2019.0 - y) / 4.0,
        np.where(
            y < 2020.2,
            0.0,
            np.where(
                y < 2021.0,
                -0.08 * np.sin(np.pi * (y - 2020.2) / 0.8),
                0.085 * np.minimum(y - 2021.0, 5.5) - 0.004 * np.maximum(y - 2024.0, 0.0) ** 2,
            ),
        ),
    )
    return base


def _metro_band_effect(dist_m: np.ndarray, truth: SyntheticGroundTruth) -> np.ndarray:
    return np.where(
        dist_m < 500.0,
        truth.metro_lt500_log_effect,
        np.where(
            dist_m < 1000.0,
            truth.metro_500_1000_log_effect,
            np.where(dist_m < 2000.0, truth.metro_1000_2000_log_effect, 0.0),
        ),
    )


def _generate_projects(
    rng: np.random.Generator, truth: SyntheticGroundTruth, metro_df: pd.DataFrame
) -> pd.DataFrame:
    """Create 75 synthetic projects across Dubai areas with controlled metro distances."""
    developers = list(DEVELOPER_OFFSETS.keys())
    rows: list[dict[str, Any]] = []
    project_idx = 1

    # Distribute 5 projects per area with distinct metro proximity profiles
    for area_name, spec in AREA_SPECS.items():
        base_lat = float(spec["lat"])
        base_lon = float(spec["lon"])
        # Pick 5 distinct metro distance buckets so metro effects are well identified within areas
        target_dists = [280.0, 440.0, 760.0, 1450.0, 2900.0]
        for k, target_d in enumerate(target_dists):
            # Place project near nearest station at roughly target_d if area has a station nearby
            station_row = metro_df.loc[metro_df["station_name"] == spec["metro"]].iloc[0]
            # Convert distance to approximate lat/lon offset (1 deg lat ~ 111.32 km)
            angle = (project_idx * 1.17) % (2.0 * math.pi)
            if k < 4 and area_name in {
                "Downtown Dubai",
                "Dubai Marina",
                "Business Bay",
                "Jumeirah Lake Towers",
                "Discovery Gardens",
                "Al Furjan",
                "Dubai Creek Harbour",
                "Palm Jumeirah",
                "City Walk",
                "Emaar Beachfront",
            }:
                anchor_lat = float(station_row["lat"])
                anchor_lon = float(station_row["lon"])
            else:
                anchor_lat = base_lat
                anchor_lon = base_lon

            d_deg_lat = (target_d * math.cos(angle)) / 111320.0
            d_deg_lon = (target_d * math.sin(angle)) / (
                111320.0 * math.cos(math.radians(anchor_lat))
            )
            lat = anchor_lat + d_deg_lat
            lon = anchor_lon + d_deg_lon

            # Actual distance to nearest station
            dists = _haversine_meters(
                metro_df["lat"].to_numpy(), metro_df["lon"].to_numpy(), lat, lon
            )
            nearest_idx = int(np.argmin(dists))
            nearest_station = metro_df.iloc[nearest_idx]

            dev = developers[(project_idx + k) % len(developers)]
            launch_year = int(rng.integers(2013, 2025))
            completion_year = launch_year + int(rng.integers(2, 5))
            launch_dt = date(launch_year, int(rng.integers(1, 13)), 1)
            completion_dt = date(completion_year, int(rng.integers(1, 13)), 1)
            is_finished = completion_year <= 2025
            completion_pct = 100.0 if is_finished else float(rng.integers(20, 90))

            rows.append(
                {
                    "project_id": f"PRJ-{project_idx:04d}",
                    "project_name": f"{area_name} Residences {k + 1}",
                    "developer_name": dev,
                    "area_name": area_name,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "nearest_metro": str(nearest_station["station_name"]),
                    "metro_distance_m": round(float(dists[nearest_idx]), 1),
                    "metro_opening_date": str(nearest_station["opening_date"]),
                    "is_route_2020_metro": bool(nearest_station["is_route_2020"]),
                    "nearest_mall": "Dubai Mall"
                    if area_name in {"Downtown Dubai", "Business Bay", "City Walk"}
                    else "Mall of the Emirates"
                    if area_name
                    in {"Dubai Hills Estate", "Arabian Ranches", "Jumeirah Village Circle"}
                    else "Dubai Marina Mall",
                    "nearest_landmark": f"{area_name} Central",
                    "project_effect": float(rng.normal(0.0, 0.035)),
                    "status": "Finished" if is_finished else "Under Construction",
                    "completion_pct": completion_pct,
                    "launch_date": launch_dt.isoformat(),
                    "completion_date": completion_dt.isoformat(),
                }
            )
            project_idx += 1

    return pd.DataFrame(rows)


def generate_synthetic_bundle(
    n_sales: int = 8000,
    n_rents: int = 10000,
    seed: int = 42,
    include_dirty_rows: bool = False,
    truth: SyntheticGroundTruth | None = None,
) -> SyntheticBundle:
    """Generate synthetic DLD transactions, Ejari rent contracts, projects, metro stations, and index."""
    gt = truth or SyntheticGroundTruth(seed=seed)
    rng = np.random.default_rng(seed)

    metro_df = pd.DataFrame(METRO_STATION_ROWS)
    projects_df = _generate_projects(rng, gt, metro_df)

    # Sample sales dates from 2015-01-01 to 2026-06-30
    start_ts = pd.Timestamp("2015-01-01")
    end_ts = pd.Timestamp("2026-06-30")
    total_days = (end_ts - start_ts).days

    # Slightly higher transaction density in 2021-2026 (realistic Dubai boom)
    u = rng.uniform(0.0, 1.0, size=n_sales)
    day_offsets = np.floor((u**0.78) * total_days).astype(int)
    tx_dates = start_ts + pd.to_timedelta(day_offsets, unit="D")

    # Sample areas with realistic weights, then pick project inside area
    areas = list(AREA_WEIGHTS.keys())
    weights = np.array(list(AREA_WEIGHTS.values()), dtype=float)
    weights /= weights.sum()
    chosen_areas = rng.choice(areas, size=n_sales, p=weights)

    projects_by_area = {
        area: projects_df.loc[projects_df["area_name"] == area].reset_index(drop=True)
        for area in areas
    }

    proj_records: list[dict[Any, Any]] = []
    for area in chosen_areas:
        pool = projects_by_area[area]
        pick = int(rng.integers(0, len(pool)))
        proj_records.append(pool.iloc[pick].to_dict())
    tx_proj = pd.DataFrame(proj_records)

    # Property type / rooms / area_sqm
    villa_probs = np.array([AREA_SPECS[a]["villa_share"] for a in chosen_areas])
    is_villa = rng.uniform(0.0, 1.0, size=n_sales) < villa_probs
    property_type = np.where(is_villa, "Villa", "Unit")
    property_sub_type = np.where(
        is_villa,
        rng.choice(["Villa", "Townhouse"], size=n_sales, p=[0.6, 0.4]),
        rng.choice(["Flat", "Penthouse"], size=n_sales, p=[0.96, 0.04]),
    )

    rooms_apt = rng.choice([0, 1, 2, 3, 4], size=n_sales, p=[0.18, 0.36, 0.30, 0.12, 0.04])
    rooms_villa = rng.choice([3, 4, 5, 6], size=n_sales, p=[0.35, 0.40, 0.20, 0.05])
    rooms = np.where(is_villa, rooms_villa, rooms_apt)

    base_area_by_room = {0: 42.0, 1: 72.0, 2: 118.0, 3: 178.0, 4: 265.0, 5: 380.0, 6: 520.0}
    expected_area = np.array([base_area_by_room[int(r)] for r in rooms], dtype=float)
    expected_area = np.where(is_villa, expected_area * 1.25, expected_area)
    area_sqm = np.clip(expected_area * np.exp(rng.normal(0.0, 0.14, size=n_sales)), 24.0, 1800.0)

    # Off-plan flag: depends on tx_date relative to project completion_date + random new launches
    comp_dates = pd.to_datetime(tx_proj["completion_date"])
    is_offplan = (tx_dates < comp_dates).astype(int)
    # Force a clean mix (~35% off-plan) across areas/developers
    flip_mask = rng.uniform(0.0, 1.0, size=n_sales) < 0.12
    is_offplan = np.where(flip_mask, 1 - is_offplan, is_offplan)

    months_to_completion = np.where(
        is_offplan == 1,
        np.clip((comp_dates - tx_dates).dt.days / 30.4375, 1.0, 48.0),
        0.0,
    )

    # Compute ground-truth log(price_per_sqm)
    base_ppsqm = np.array([AREA_SPECS[a]["base_ppsqm"] for a in chosen_areas], dtype=float)
    metro_dist = tx_proj["metro_distance_m"].to_numpy(dtype=float)
    metro_eff = _metro_band_effect(metro_dist, gt)

    # Route 2020 opening step for stations opened on/after 2021-01-01
    is_r2020 = tx_proj["is_route_2020_metro"].to_numpy(dtype=bool) & (metro_dist < 1000.0)
    r2020_open_dt = pd.to_datetime(tx_proj["metro_opening_date"])
    post_r2020 = is_r2020 & (tx_dates >= r2020_open_dt)
    r2020_eff = np.where(post_r2020, gt.route_2020_post_open_log_effect, 0.0)

    dev_offsets = tx_proj["developer_name"].map(DEVELOPER_OFFSETS).to_numpy(dtype=float)
    offplan_eff = is_offplan * (gt.offplan_log_effect + dev_offsets)
    cycle_eff = market_cycle_log_effect(pd.Series(tx_dates))
    size_eff = gt.area_size_elasticity * np.log(area_sqm / 100.0)
    villa_eff = is_villa * gt.villa_log_effect
    room_eff = 0.015 * np.maximum(rooms - 1, 0)
    proj_eff = tx_proj["project_effect"].to_numpy(dtype=float)
    eps = rng.normal(0.0, gt.noise_std, size=n_sales)

    log_ppsqm = (
        np.log(base_ppsqm)
        + proj_eff
        + metro_eff
        + r2020_eff
        + offplan_eff
        + cycle_eff
        + size_eff
        + villa_eff
        + room_eff
        + eps
    )
    price_per_sqm = np.exp(log_ppsqm)
    actual_worth = price_per_sqm * area_sqm

    tx_df = pd.DataFrame(
        {
            "transaction_id": [f"SYN-TX-{i + 1:06d}" for i in range(n_sales)],
            "instance_date": tx_dates.strftime("%Y-%m-%d"),
            "procedure_name": "Sales",
            "trans_group": "Sales",
            "property_type": property_type,
            "property_sub_type": property_sub_type,
            "property_usage": "Residential",
            "area_name": chosen_areas,
            "master_project": tx_proj["area_name"].to_numpy(),
            "project_name": tx_proj["project_name"].to_numpy(),
            "developer_name": tx_proj["developer_name"].to_numpy(),
            "rooms": rooms.astype(int),
            "rooms_label": np.where(rooms == 0, "Studio", np.char.add(rooms.astype(str), " B/R")),
            "area_sqm": np.round(area_sqm, 2),
            "area_sqft": np.round(area_sqm * SQFT_PER_SQM, 2),
            "actual_worth": np.round(actual_worth, 2),
            "price_per_sqm": np.round(price_per_sqm, 2),
            "price_per_sqft": np.round(price_per_sqm / SQFT_PER_SQM, 2),
            "is_offplan": is_offplan.astype(int),
            "has_parking": rng.choice([0, 1], size=n_sales, p=[0.08, 0.92]).astype(int),
            "months_to_completion": np.round(months_to_completion, 1),
            "nearest_metro": tx_proj["nearest_metro"].to_numpy(),
            "nearest_mall": tx_proj["nearest_mall"].to_numpy(),
            "nearest_landmark": tx_proj["nearest_landmark"].to_numpy(),
            "metro_distance_m": tx_proj["metro_distance_m"].to_numpy(),
            "lat": np.round(
                tx_proj["lat"].to_numpy(dtype=float) + rng.normal(0.0, 0.0008, size=n_sales), 6
            ),
            "lon": np.round(
                tx_proj["lon"].to_numpy(dtype=float) + rng.normal(0.0, 0.0008, size=n_sales), 6
            ),
            "flag_outlier": False,
            "flag_non_market": False,
            "flag_duplicate": False,
            "data_source": "synthetic",
        }
    )

    if include_dirty_rows:
        dirty_rows = tx_df.iloc[:25].copy()
        dirty_rows.loc[dirty_rows.index[0:5], "procedure_name"] = "Mortgage"
        dirty_rows.loc[dirty_rows.index[0:5], "trans_group"] = "Mortgages"
        dirty_rows.loc[dirty_rows.index[5:10], "property_usage"] = "Commercial"
        dirty_rows.loc[dirty_rows.index[10:13], "area_sqm"] = -15.0
        dirty_rows.loc[dirty_rows.index[13:16], "actual_worth"] = 0.0
        dirty_rows.loc[dirty_rows.index[16:19], "price_per_sqm"] = 950000.0
        dirty_rows.loc[dirty_rows.index[16:19], "actual_worth"] = (
            dirty_rows.loc[dirty_rows.index[16:19], "area_sqm"] * 950000.0
        )
        dirty_rows.loc[dirty_rows.index[19:22], "actual_worth"] = 25000.0
        dirty_rows.loc[dirty_rows.index[19:22], "price_per_sqm"] = 300.0
        # Exact duplicate of row 0
        dup = tx_df.iloc[[0]].copy()
        # Whitespace/case variation
        dirty_rows.loc[dirty_rows.index[22:25], "area_name"] = "  downtown dubai "
        tx_df = pd.concat([tx_df, dirty_rows, dup], ignore_index=True)

    # Generate Ejari rent contracts
    rent_u = rng.uniform(0.0, 1.0, size=n_rents)
    rent_day_offsets = np.floor((rent_u**0.82) * total_days).astype(int)
    rent_dates = start_ts + pd.to_timedelta(rent_day_offsets, unit="D")
    rent_areas = rng.choice(areas, size=n_rents, p=weights)

    r_proj_records: list[dict[Any, Any]] = []
    for area in rent_areas:
        pool = projects_by_area[area]
        pick = int(rng.integers(0, len(pool)))
        r_proj_records.append(pool.iloc[pick].to_dict())
    r_proj = pd.DataFrame(r_proj_records)

    r_villa_probs = np.array([AREA_SPECS[a]["villa_share"] for a in rent_areas])
    r_is_villa = rng.uniform(0.0, 1.0, size=n_rents) < r_villa_probs
    r_type = np.where(r_is_villa, "Villa", "Unit")
    r_sub_type = np.where(r_is_villa, "Villa", "Flat")
    r_rooms = np.where(
        r_is_villa,
        rng.choice([3, 4, 5], size=n_rents, p=[0.4, 0.4, 0.2]),
        rng.choice([0, 1, 2, 3], size=n_rents, p=[0.20, 0.38, 0.30, 0.12]),
    )
    r_expected_area = np.array([base_area_by_room[int(r)] for r in r_rooms], dtype=float)
    r_expected_area = np.where(r_is_villa, r_expected_area * 1.25, r_expected_area)
    r_area_sqm = np.clip(
        r_expected_area * np.exp(rng.normal(0.0, 0.12, size=n_rents)), 24.0, 1500.0
    )

    r_base_ppsqm = np.array([AREA_SPECS[a]["base_ppsqm"] for a in rent_areas], dtype=float)
    r_yield = np.array([AREA_SPECS[a]["yield"] for a in rent_areas], dtype=float)
    r_cycle = market_cycle_log_effect(pd.Series(rent_dates))
    r_metro_eff = 0.7 * _metro_band_effect(r_proj["metro_distance_m"].to_numpy(dtype=float), gt)
    is_renewal = rng.choice([0, 1], size=n_rents, p=[0.55, 0.45])
    # Renewals lag slightly behind new market rents
    renewal_lag = np.where(is_renewal == 1, -0.06, 0.0)
    r_eps = rng.normal(0.0, 0.08, size=n_rents)

    log_rent_ppsqm = (
        np.log(r_base_ppsqm * r_yield)
        + 0.85 * r_cycle
        + r_metro_eff
        + gt.area_size_elasticity * np.log(r_area_sqm / 100.0)
        + renewal_lag
        + r_eps
    )
    rent_per_sqm = np.exp(log_rent_ppsqm)
    annual_amount = rent_per_sqm * r_area_sqm

    rent_df = pd.DataFrame(
        {
            "contract_id": [f"SYN-EJ-{i + 1:06d}" for i in range(n_rents)],
            "contract_start_date": rent_dates.strftime("%Y-%m-%d"),
            "contract_end_date": (rent_dates + pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
            "contract_type": np.where(is_renewal == 1, "Renew", "New"),
            "is_renewal": is_renewal.astype(int),
            "property_type": r_type,
            "property_sub_type": r_sub_type,
            "property_usage": "Residential",
            "area_name": rent_areas,
            "project_name": r_proj["project_name"].to_numpy(),
            "rooms": r_rooms.astype(int),
            "area_sqm": np.round(r_area_sqm, 2),
            "annual_amount": np.round(annual_amount, 2),
            "rent_per_sqm": np.round(rent_per_sqm, 2),
            "nearest_metro": r_proj["nearest_metro"].to_numpy(),
            "metro_distance_m": r_proj["metro_distance_m"].to_numpy(),
            "data_source": "synthetic",
        }
    )

    # Monthly official-style sale price index anchored at 2019 average = 100
    months = pd.date_range("2015-01-01", "2026-06-01", freq="MS")
    idx_cycle = market_cycle_log_effect(pd.Series(months))
    index_values = 100.0 * np.exp(idx_cycle + rng.normal(0.0, 0.008, size=len(months)))
    sale_index_df = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "index_value": np.round(index_values, 2),
            "property_type": "All Residential",
            "data_source": "synthetic",
        }
    )

    return SyntheticBundle(
        transactions=tx_df,
        rent_contracts=rent_df,
        projects=projects_df,
        metro_stations=metro_df,
        sale_index=sale_index_df,
        ground_truth=gt,
    )


def ground_truth_summary(truth: SyntheticGroundTruth | None = None) -> dict[str, Any]:
    """Return serializable dictionary of synthetic ground-truth parameters."""
    return asdict(truth or SyntheticGroundTruth())
