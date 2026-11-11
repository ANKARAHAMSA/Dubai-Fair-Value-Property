"""Community/area boundary loader with GeoJSON cache and centroid-buffer fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from dfv.config import PATHS
from dfv.geo.geocode import DUBAI_AREA_CENTROIDS

logger = logging.getLogger("dfv.geo.boundaries")


def load_area_boundaries(
    raw_dir: Path = PATHS.raw,
    cache_dir: Path = PATHS.cache,
    area_names: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, str]:
    """Load community polygons from raw/cache or synthesize centroid buffers cleanly."""
    for candidate in sorted(raw_dir.glob("*.geojson")) + sorted(raw_dir.glob("*boundaries*.gpkg")):
        try:
            gdf = gpd.read_file(candidate)
            if not gdf.empty:
                return gdf.to_crs("EPSG:4326"), f"raw_official ({candidate.name})"
        except Exception as exc:
            logger.warning("Failed reading boundary file %s: %s", candidate.name, exc)

    cached = cache_dir / "dubai_boundaries.geojson"
    if cached.exists():
        try:
            gdf = gpd.read_file(cached)
            if not gdf.empty:
                return gdf.to_crs("EPSG:4326"), "osm_cached (dubai_boundaries.geojson)"
        except Exception as exc:
            logger.warning("Failed reading cached boundaries: %s", exc)

    # Fallback: centroid circular buffers (~1.5 km radius in degrees ~ 0.0135)
    names = area_names or sorted(DUBAI_AREA_CENTROIDS.keys())
    rows = []
    for area in names:
        lat, lon = DUBAI_AREA_CENTROIDS.get(area, (25.14, 55.23))
        rows.append(
            {
                "area_name": area,
                "lat": lat,
                "lon": lon,
                "geometry": Point(lon, lat).buffer(0.0135),
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf, "centroid_buffer_fallback"


def summarize_area_geometries(df: pd.DataFrame) -> pd.DataFrame:
    """Compute representative area centroids from geocoded transaction coordinates."""
    if df.empty or not {"area_name", "lat", "lon"}.issubset(df.columns):
        return pd.DataFrame(columns=["area_name", "lat", "lon", "tx_count"])
    return (
        df.groupby("area_name", as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"), tx_count=("area_name", "count"))
        .sort_values("tx_count", ascending=False)
        .reset_index(drop=True)
    )
