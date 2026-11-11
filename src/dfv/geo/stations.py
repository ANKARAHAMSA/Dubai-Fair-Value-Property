"""Distance calculations to Dubai Metro, Tram, malls, coastline, airports, and POIs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# Curated Dubai reference coordinates (WGS84 lat, lon)
DUBAI_MALLS: list[tuple[str, float, float]] = [
    ("Dubai Mall", 25.1972, 55.2796),
    ("Mall of the Emirates", 25.1181, 55.2006),
    ("Dubai Marina Mall", 25.0765, 55.1404),
    ("Ibn Battuta Mall", 25.0445, 55.1202),
    ("City Centre Deira", 25.2521, 55.3312),
    ("Dubai Hills Mall", 25.1011, 55.2395),
    ("Nakheel Mall", 25.1140, 55.1387),
    ("City Centre Mirdif", 25.2159, 55.4085),
    ("Dubai Festival City Mall", 25.2216, 55.3526),
]

DUBAI_TRAM_STATIONS: list[tuple[str, float, float]] = [
    ("Jumeirah Beach Residence 1", 25.0764, 55.1331),
    ("Jumeirah Beach Residence 2", 25.0715, 55.1304),
    ("Dubai Marina Mall", 25.0757, 55.1394),
    ("Dubai Marina", 25.0804, 55.1456),
    ("Marina Towers", 25.0844, 55.1449),
    ("Mina Seyahi", 25.0906, 55.1497),
    ("Media City", 25.0942, 55.1544),
    ("Palm Jumeirah", 25.1003, 55.1618),
    ("Knowledge Village", 25.1035, 55.1646),
    ("Al Sufouh", 25.1088, 55.1695),
]

DUBAI_COAST_ANCHORS: list[tuple[str, float, float]] = [
    ("JBR Coast", 25.0785, 55.1310),
    ("Palm West", 25.1120, 55.1330),
    ("Umm Suqeim Beach", 25.1480, 55.1930),
    ("Kite Beach", 25.1665, 55.2100),
    ("Jumeirah Beach", 25.2070, 55.2420),
    ("La Mer", 25.2260, 55.2580),
    ("Al Mamzar Beach", 25.3100, 55.3440),
    ("Emaar Beachfront", 25.0935, 55.1360),
]

DUBAI_AIRPORTS: list[tuple[str, float, float]] = [
    ("DXB", 25.2532, 55.3657),
    ("DWC", 24.8964, 55.1614),
]

DOWNTOWN_CENTER: tuple[float, float] = (25.1972, 55.2744)


def haversine_nearest_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
    ref_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Compute vectorized great-circle distance (meters) from query points to nearest reference point."""
    if len(ref_lats) == 0 or len(lat1) == 0:
        empty_d = np.full(len(lat1), np.nan, dtype=float)
        return empty_d, (["Unknown"] * len(lat1) if ref_names is not None else None)

    r = 6371000.0
    lat1_r = np.radians(lat1)[:, None]
    lon1_r = np.radians(lon1)[:, None]
    lat2_r = np.radians(ref_lats)[None, :]
    lon2_r = np.radians(ref_lons)[None, :]

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    dist_mat = 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    nearest_idx = np.argmin(dist_mat, axis=1)
    min_dist = dist_mat[np.arange(len(lat1)), nearest_idx]
    if ref_names is not None:
        names = [ref_names[int(i)] for i in nearest_idx]
        return min_dist, names
    return min_dist, None


def count_within_radius_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
    radius_m: float = 1000.0,
) -> np.ndarray:
    """Count how many reference points fall within radius_m of each query point (chunked for RAM safety)."""
    n = len(lat1)
    if n == 0 or len(ref_lats) == 0:
        return np.zeros(n, dtype=int)

    counts = np.zeros(n, dtype=int)
    r = 6371000.0
    lat2_r = np.radians(ref_lats)[None, :]
    lon2_r = np.radians(ref_lons)[None, :]
    cos_lat2 = np.cos(lat2_r)

    chunk_size = 2000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        lat1_r = np.radians(lat1[start:end])[:, None]
        lon1_r = np.radians(lon1[start:end])[:, None]
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * cos_lat2 * np.sin(dlon / 2.0) ** 2
        dist_mat = 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        counts[start:end] = (dist_mat <= radius_m).sum(axis=1)

    return counts


def metro_distance_band(dist_m: pd.Series | np.ndarray) -> pd.Series:
    """Assign standardized metro proximity distance bands."""
    s = pd.Series(dist_m, dtype="float64")
    bands = np.where(
        s < 500.0,
        "<500m",
        np.where(
            s < 1000.0,
            "500-1000m",
            np.where(s < 2000.0, "1000-2000m", ">2000m"),
        ),
    )
    return pd.Series(bands, index=s.index, dtype="string")


def compute_proximity_features(
    coords_df: pd.DataFrame,
    metro_df: pd.DataFrame,
    pois_df: pd.DataFrame | None = None,
    bus_stops_df: pd.DataFrame | None = None,
    preserve_existing_metro_dist: bool = True,
) -> pd.DataFrame:
    """Compute metro/tram/mall/coast/airport/downtown distances and 1km POI/bus counts on unique coordinates."""
    out = coords_df.copy()
    lats = out["lat"].to_numpy(dtype=float)
    lons = out["lon"].to_numpy(dtype=float)

    # 1. Metro station distance
    m_lats = metro_df["lat"].to_numpy(dtype=float)
    m_lons = metro_df["lon"].to_numpy(dtype=float)
    m_names = metro_df["station_name"].astype(str).tolist()
    calc_metro_dist, calc_metro_names = haversine_nearest_m(lats, lons, m_lats, m_lons, m_names)

    if (
        preserve_existing_metro_dist
        and "metro_distance_m" in out.columns
        and out["metro_distance_m"].notna().any()
    ):
        existing = pd.to_numeric(out["metro_distance_m"], errors="coerce").to_numpy(dtype=float)
        out["dist_nearest_metro_m"] = np.where(np.isfinite(existing), existing, calc_metro_dist)
    else:
        out["dist_nearest_metro_m"] = np.round(calc_metro_dist, 1)

    if "nearest_metro" not in out.columns:
        out["nearest_metro"] = calc_metro_names
    else:
        out["nearest_metro"] = out["nearest_metro"].fillna(
            pd.Series(calc_metro_names, index=out.index)
        )

    out["metro_within_500m"] = (out["dist_nearest_metro_m"] < 500.0).astype(int)
    out["metro_within_1000m"] = (out["dist_nearest_metro_m"] < 1000.0).astype(int)
    out["metro_band"] = metro_distance_band(out["dist_nearest_metro_m"])

    # 2. Tram, mall, coast, airport, downtown
    t_lats = np.array([x[1] for x in DUBAI_TRAM_STATIONS], dtype=float)
    t_lons = np.array([x[2] for x in DUBAI_TRAM_STATIONS], dtype=float)
    tram_d, _ = haversine_nearest_m(lats, lons, t_lats, t_lons)
    out["dist_nearest_tram_m"] = np.round(tram_d, 1)

    if pois_df is not None and not pois_df.empty and "category" in pois_df.columns:
        mall_pts = pois_df[pois_df["category"] == "mall"]
        if not mall_pts.empty:
            mall_lats = mall_pts["lat"].to_numpy(dtype=float)
            mall_lons = mall_pts["lon"].to_numpy(dtype=float)
        else:
            mall_lats = np.array([x[1] for x in DUBAI_MALLS], dtype=float)
            mall_lons = np.array([x[2] for x in DUBAI_MALLS], dtype=float)
    else:
        mall_lats = np.array([x[1] for x in DUBAI_MALLS], dtype=float)
        mall_lons = np.array([x[2] for x in DUBAI_MALLS], dtype=float)

    mall_d, _ = haversine_nearest_m(lats, lons, mall_lats, mall_lons)
    out["dist_nearest_mall_m"] = np.round(mall_d, 1)

    c_lats = np.array([x[1] for x in DUBAI_COAST_ANCHORS], dtype=float)
    c_lons = np.array([x[2] for x in DUBAI_COAST_ANCHORS], dtype=float)
    coast_d, _ = haversine_nearest_m(lats, lons, c_lats, c_lons)
    out["dist_nearest_coast_m"] = np.round(coast_d, 1)

    a_lats = np.array([x[1] for x in DUBAI_AIRPORTS], dtype=float)
    a_lons = np.array([x[2] for x in DUBAI_AIRPORTS], dtype=float)
    airport_d, _ = haversine_nearest_m(lats, lons, a_lats, a_lons)
    out["dist_airport_m"] = np.round(airport_d, 1)

    dt_d, _ = haversine_nearest_m(
        lats, lons, np.array([DOWNTOWN_CENTER[0]]), np.array([DOWNTOWN_CENTER[1]])
    )
    out["dist_downtown_m"] = np.round(dt_d, 1)

    # 3. POI count & bus stop density within 1 km
    if pois_df is not None and not pois_df.empty:
        out["poi_count_1km"] = count_within_radius_m(
            lats,
            lons,
            pois_df["lat"].to_numpy(dtype=float),
            pois_df["lon"].to_numpy(dtype=float),
            radius_m=1000.0,
        )
    else:
        # Fallback: count curated malls + metro stations + tram stations within 1 km
        anchor_lats = np.concatenate([m_lats, t_lats, mall_lats])
        anchor_lons = np.concatenate([m_lons, t_lons, mall_lons])
        out["poi_count_1km"] = count_within_radius_m(
            lats, lons, anchor_lats, anchor_lons, radius_m=1000.0
        )

    if bus_stops_df is not None and not bus_stops_df.empty:
        out["bus_stop_density_1km"] = count_within_radius_m(
            lats,
            lons,
            bus_stops_df["lat"].to_numpy(dtype=float),
            bus_stops_df["lon"].to_numpy(dtype=float),
            radius_m=1000.0,
        )
    else:
        out["bus_stop_density_1km"] = np.maximum(1, (out["poi_count_1km"] * 2).astype(int))

    return out


def haversine_pair_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute scalar great-circle distance in meters."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def stations_table_summary(metro_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize metro station counts and known opening date coverage."""
    has_date = (
        metro_df["opening_date"].notna() & (metro_df["opening_date"].astype(str).str.len() > 0)
        if "opening_date" in metro_df.columns
        else pd.Series(False, index=metro_df.index)
    )
    return {
        "total_stations": int(len(metro_df)),
        "stations_with_opening_date": int(has_date.sum()),
    }
