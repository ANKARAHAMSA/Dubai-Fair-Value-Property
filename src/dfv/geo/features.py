"""High-level geospatial enrichment pipeline and optional OSM Overpass POI caching."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from dfv.config import PATHS
from dfv.geo.geocode import geocode_transactions
from dfv.geo.stations import compute_proximity_features

logger = logging.getLogger("dfv.geo.features")


def load_cached_pois(cache_dir: Path = PATHS.cache) -> pd.DataFrame | None:
    """Load cached OSM POIs if present."""
    poi_path = cache_dir / "osm_pois.parquet"
    if poi_path.exists():
        return pd.read_parquet(poi_path)
    return None


def fetch_and_cache_osm_pois(cache_dir: Path = PATHS.cache) -> pd.DataFrame | None:
    """Fetch Dubai malls/schools/hospitals/parks/beaches via OSMnx Overpass and cache to Parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    poi_path = cache_dir / "osm_pois.parquet"
    if poi_path.exists():
        return pd.read_parquet(poi_path)

    try:
        import osmnx as ox

        ox.settings.requests_timeout = 10
        bbox = (55.02, 24.85, 55.48, 25.34)
        tags: dict[str, Any] = {
            "shop": "mall",
            "amenity": ["school", "hospital"],
            "leisure": "park",
        }
        records: list[dict[str, Any]] = []
        gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
        if not gdf.empty:
            pts = gdf.geometry.centroid
            names = gdf["name"] if "name" in gdf.columns else pd.Series("", index=gdf.index)
            shops = gdf["shop"] if "shop" in gdf.columns else pd.Series("", index=gdf.index)
            amenities = (
                gdf["amenity"] if "amenity" in gdf.columns else pd.Series("", index=gdf.index)
            )
            for idx, (name, pt) in enumerate(zip(names, pts, strict=True)):
                cat = (
                    "mall"
                    if str(shops.iloc[idx]) == "mall"
                    else str(amenities.iloc[idx])
                    if pd.notna(amenities.iloc[idx]) and str(amenities.iloc[idx]) != ""
                    else "park"
                )
                records.append(
                    {
                        "category": cat,
                        "name": str(name) if pd.notna(name) else cat,
                        "lat": float(pt.y),
                        "lon": float(pt.x),
                    }
                )
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(poi_path, index=False)
            logger.info("Cached %d OSM POIs to %s", len(df), poi_path)
            return df
    except Exception as exc:
        logger.warning("OSM Overpass POI fetch skipped (%s); using curated Dubai anchors.", exc)
    return None


def load_gtfs_bus_stops(raw_dir: Path = PATHS.raw) -> pd.DataFrame | None:
    """Load bus stops from MobilityDatabase GTFS zip if downloaded."""
    gtfs_path = raw_dir / "rta_gtfs_mdb904.zip"
    if not gtfs_path.exists():
        return None
    try:
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            with zf.open("stops.txt") as f:
                stops = pd.read_csv(io.BytesIO(f.read()))
        if {"stop_lat", "stop_lon"}.issubset(stops.columns):
            return (
                stops[["stop_lat", "stop_lon"]]
                .rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
                .dropna()
                .drop_duplicates()
                .reset_index(drop=True)
            )
    except Exception as exc:
        logger.warning("Failed extracting GTFS bus stops: %s", exc)
    return None


def enrich_with_geo_features(
    tx_df: pd.DataFrame,
    metro_df: pd.DataFrame,
    projects_df: pd.DataFrame | None = None,
    cache_dir: Path = PATHS.cache,
    raw_dir: Path = PATHS.raw,
    allow_nominatim: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Geocode unique locations and attach proximity/density features efficiently."""
    geocoded, geo_stats = geocode_transactions(
        tx_df,
        projects_df=projects_df,
        cache_path=cache_dir / "geocode.sqlite",
        allow_nominatim=allow_nominatim,
    )

    # Deduplicate purely on immutable coordinate/existing-distance keys
    coord_cols = ["lat", "lon"]
    if "metro_distance_m" in geocoded.columns and geocoded["metro_distance_m"].notna().any():
        coord_cols.append("metro_distance_m")

    unique_coords = geocoded[coord_cols].drop_duplicates().reset_index(drop=True)
    pois_df = load_cached_pois(cache_dir)
    bus_df = load_gtfs_bus_stops(raw_dir)

    enriched_coords = compute_proximity_features(
        unique_coords,
        metro_df=metro_df,
        pois_df=pois_df,
        bus_stops_df=bus_df,
        preserve_existing_metro_dist="metro_distance_m" in coord_cols,
    ).drop_duplicates(subset=coord_cols)

    # Merge proximity features back cleanly; preserve existing nearest_metro if already populated
    added_cols = [c for c in enriched_coords.columns if c not in coord_cols]
    has_orig_metro = "nearest_metro" in geocoded.columns
    if has_orig_metro and "nearest_metro" in added_cols:
        enriched_coords = enriched_coords.rename(columns={"nearest_metro": "_calc_nearest_metro"})
        added_cols = ["_calc_nearest_metro" if c == "nearest_metro" else c for c in added_cols]

    drop_existing = [c for c in added_cols if c in geocoded.columns]
    if drop_existing:
        geocoded = geocoded.drop(columns=drop_existing)

    out = geocoded.merge(enriched_coords[coord_cols + added_cols], on=coord_cols, how="left")
    if has_orig_metro and "_calc_nearest_metro" in out.columns:
        out["nearest_metro"] = out["nearest_metro"].fillna(out["_calc_nearest_metro"])
        out = out.drop(columns=["_calc_nearest_metro"])

    if len(out) != len(geocoded):
        raise RuntimeError(f"Geo merge changed row count: {len(geocoded)} -> {len(out)}")
    return out, geo_stats
