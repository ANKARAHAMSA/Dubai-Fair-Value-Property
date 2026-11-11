"""Rate-limited Nominatim geocoder with persistent SQLite cache and area-centroid fallback."""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

from dfv.config import PATHS

logger = logging.getLogger("dfv.geo.geocode")

# Authoritative Dubai district/community centroids (WGS84 lat, lon)
DUBAI_AREA_CENTROIDS: dict[str, tuple[float, float]] = {
    "Palm Jumeirah": (25.1124, 55.1390),
    "Downtown Dubai": (25.1972, 55.2744),
    "Emaar Beachfront": (25.0935, 55.1375),
    "City Walk": (25.2078, 55.2623),
    "Dubai Marina": (25.0805, 55.1403),
    "Dubai Creek Harbour": (25.2044, 55.3441),
    "Business Bay": (25.1865, 55.2721),
    "Dubai Hills Estate": (25.1072, 55.2435),
    "Arabian Ranches": (25.0589, 55.2636),
    "Jumeirah Lake Towers": (25.0695, 55.1436),
    "Damac Hills": (25.0255, 55.2485),
    "Al Furjan": (25.0279, 55.1433),
    "Jumeirah Village Circle": (25.0554, 55.2096),
    "Jumeirah Village Triangle": (25.0434, 55.1904),
    "Discovery Gardens": (25.0419, 55.1397),
    "Dubai South": (24.8962, 55.1614),
    "MBR City / Sobha Hartland": (25.1745, 55.3082),
    "Dubai Sports City": (25.0398, 55.2225),
    "Arjan": (25.0607, 55.2389),
    "International City": (25.1636, 55.4071),
    "Dubai Silicon Oasis": (25.1212, 55.3773),
    "Dubai Maritime City": (25.2738, 55.2671),
    "Dubailand Residence Complex": (25.0867, 55.3778),
    "Majan": (25.0915, 55.3137),
    "Me'Aisem First": (25.0338, 55.1932),
    "Al Yelayiss 2": (25.0128, 55.2912),
    "Al Jadaf": (25.2178, 55.3264),
    "Al Thanyah Third": (25.0937, 55.1722),
    "Al Hebiah Fifth": (25.0024, 55.2471),
    "Al Hebiah Sixth": (25.0176, 55.2647),
    "Al Barshaa South Second": (25.0751, 55.2434),
    "Al Thanayah Fourth": (25.0604, 55.1717),
    "Wadi Al Safa 7": (25.0397, 55.2835),
    "Wadi Al Safa 6": (25.0522, 55.2672),
    "Al Yufrah 1": (25.0012, 55.3442),
    "Ras Al Khor Industrial First": (25.1733, 55.3414),
    "Um Suqaim Third": (25.1408, 55.1973),
    "Al Barsha First": (25.1135, 55.1975),
    "Al Barsha South First": (25.0835, 55.2111),
    "Al Sufouh First": (25.1098, 55.1729),
    "Jumeirah First": (25.2302, 55.2651),
    "Za'abeel Second": (25.2141, 55.2878),
    "Nad Al Shiba First": (25.1588, 55.3183),
    "Trade Centre Second": (25.2194, 55.2817),
    "World Trade Centre": (25.2255, 55.2869),
    "Mirdif": (25.2185, 55.4173),
    "Dubai Production City": (25.0338, 55.1932),
}

DUBAI_DEFAULT_COORD: tuple[float, float] = (25.1400, 55.2300)


class GeocodeCache:
    """SQLite-backed cache for Nominatim geocoding results."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS geocache (
                    query TEXT PRIMARY KEY,
                    lat REAL,
                    lon REAL,
                    confidence TEXT,
                    provider TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()

    def get(self, query: str) -> tuple[float, float, str] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT lat, lon, confidence FROM geocache WHERE query = ?", (query,)
            ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return float(row[0]), float(row[1]), str(row[2])

    def put(
        self, query: str, lat: float | None, lon: float | None, confidence: str, provider: str
    ) -> None:
        now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO geocache (query, lat, lon, confidence, provider, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (query, lat, lon, confidence, provider, now_iso),
            )
            conn.commit()


class RateLimitedNominatim:
    """Nominatim geocoder enforcing <= 1 request/second with custom User-Agent."""

    def __init__(
        self,
        cache_path: Path,
        user_agent: str = "DubaiFairValue/0.1 (portfolio-research)",
        min_interval_sec: float = 1.05,
    ) -> None:
        self.cache = GeocodeCache(cache_path)
        self.min_interval_sec = min_interval_sec
        self._last_call_ts = 0.0
        self._geocoder = Nominatim(user_agent=user_agent, timeout=10)

    def geocode(self, query: str) -> tuple[float, float, str] | None:
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self._last_call_ts = time.monotonic()

        try:
            loc = self._geocoder.geocode(query)
        except Exception as exc:
            logger.debug("Nominatim lookup failed for '%s': %s", query, exc)
            return None

        if loc is None:
            self.cache.put(query, None, None, "missing", "nominatim")
            return None

        lat, lon = float(loc.latitude), float(loc.longitude)
        # Validate inside UAE / Dubai bounding box
        if not (24.4 <= lat <= 25.6 and 54.7 <= lon <= 55.8):
            self.cache.put(query, None, None, "out_of_bounds", "nominatim")
            return None

        self.cache.put(query, lat, lon, "nominatim_project", "nominatim")
        return lat, lon, "nominatim_project"


def geocode_transactions(
    df: pd.DataFrame,
    projects_df: pd.DataFrame | None = None,
    cache_path: Path = PATHS.cache / "geocode.sqlite",
    allow_nominatim: bool = False,
    max_nominatim_calls: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach (lat, lon, geocode_confidence) via exact coords -> project join -> SQLite/Nominatim -> area centroid."""
    out = df.copy()
    has_existing = (
        "lat" in out.columns
        and "lon" in out.columns
        and out["lat"].notna().any()
        and out["lon"].notna().any()
    )
    if has_existing:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        if "geocode_confidence" not in out.columns:
            out["geocode_confidence"] = np.where(
                out["lat"].notna() & out["lon"].notna(),
                "exact_or_synthetic",
                "missing",
            )
    else:
        out["lat"] = np.nan
        out["lon"] = np.nan
        out["geocode_confidence"] = "missing"

    # Merge project-level coordinates if provided
    if (
        projects_df is not None
        and not projects_df.empty
        and {"project_name", "lat", "lon"}.issubset(projects_df.columns)
    ):
        proj_coords = (
            projects_df[["project_name", "lat", "lon"]]
            .dropna()
            .drop_duplicates(subset=["project_name"])
            .rename(columns={"lat": "_proj_lat", "lon": "_proj_lon"})
        )
        out = out.merge(proj_coords, on="project_name", how="left")
        fill_proj = out["lat"].isna() & out["_proj_lat"].notna()
        out.loc[fill_proj, "lat"] = out.loc[fill_proj, "_proj_lat"]
        out.loc[fill_proj, "lon"] = out.loc[fill_proj, "_proj_lon"]
        out.loc[fill_proj, "geocode_confidence"] = "project_registry"
        out = out.drop(columns=["_proj_lat", "_proj_lon"])

    # Cache / Nominatim at distinct project level for rows still missing coords
    need_mask = out["lat"].isna() | out["lon"].isna()
    if need_mask.any():
        cache = GeocodeCache(cache_path)
        nominatim = RateLimitedNominatim(cache_path) if allow_nominatim else None
        distinct_pairs = (
            out.loc[need_mask, ["project_name", "area_name"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        resolved_coords: dict[tuple[str, str], tuple[float, float, str]] = {}
        calls_used = 0
        for _, row in distinct_pairs.iterrows():
            proj = str(row["project_name"])
            area = str(row["area_name"])
            query = f"{proj}, {area}, Dubai, UAE"
            hit = cache.get(query)
            if hit is not None:
                resolved_coords[(proj, area)] = hit
                continue
            if nominatim is not None and calls_used < max_nominatim_calls:
                calls_used += 1
                res = nominatim.geocode(query)
                if res is not None:
                    resolved_coords[(proj, area)] = res

        if resolved_coords:
            keys = list(
                zip(
                    out.loc[need_mask, "project_name"].astype(str),
                    out.loc[need_mask, "area_name"].astype(str),
                    strict=True,
                )
            )
            lats_map = [resolved_coords[k][0] if k in resolved_coords else np.nan for k in keys]
            lons_map = [resolved_coords[k][1] if k in resolved_coords else np.nan for k in keys]
            conf_map = [resolved_coords[k][2] if k in resolved_coords else "missing" for k in keys]
            hit_series = pd.Series([np.isfinite(x) for x in lats_map], index=out.index[need_mask])
            hit_idx = hit_series[hit_series].index
            if len(hit_idx) > 0:
                out.loc[hit_idx, "lat"] = [x for x in lats_map if np.isfinite(x)]
                out.loc[hit_idx, "lon"] = [x for x in lons_map if np.isfinite(x)]
                out.loc[hit_idx, "geocode_confidence"] = [c for c in conf_map if c != "missing"]

    # Area centroid fallback for remaining rows
    still_missing = out["lat"].isna() | out["lon"].isna()
    if still_missing.any():
        area_lat_map = {k: v[0] for k, v in DUBAI_AREA_CENTROIDS.items()}
        area_lon_map = {k: v[1] for k, v in DUBAI_AREA_CENTROIDS.items()}
        mapped_lat = out.loc[still_missing, "area_name"].map(area_lat_map)
        mapped_lon = out.loc[still_missing, "area_name"].map(area_lon_map)
        has_centroid = mapped_lat.notna()
        out.loc[still_missing, "lat"] = mapped_lat.fillna(DUBAI_DEFAULT_COORD[0]).to_numpy()
        out.loc[still_missing, "lon"] = mapped_lon.fillna(DUBAI_DEFAULT_COORD[1]).to_numpy()
        out.loc[still_missing, "geocode_confidence"] = np.where(
            has_centroid.to_numpy(), "area_centroid", "dubai_default"
        )

    counts = out["geocode_confidence"].value_counts().to_dict()
    total = max(len(out), 1)
    exact_like = sum(
        counts.get(k, 0) for k in ("exact_or_synthetic", "project_registry", "nominatim_project")
    )
    stats = {
        "total_rows": int(len(out)),
        "exact_or_project_pct": round(100.0 * exact_like / total, 2),
        "area_centroid_pct": round(100.0 * counts.get("area_centroid", 0) / total, 2),
        "dubai_default_pct": round(100.0 * counts.get("dubai_default", 0) / total, 2),
        "breakdown": {str(k): int(v) for k, v in counts.items()},
    }
    return out, stats
