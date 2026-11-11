"""High-level geospatial enrichment pipeline and optional OSM Overpass POI caching."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dfv.config import PATHS
from dfv.geo.geocode import geocode_transactions
from dfv.geo.stations import (
    compute_proximity_features,
    haversine_nearest_m,
    metro_distance_band,
    resolve_station_opening_dates,
)

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


def _attach_time_aware_metro_features(
    df: pd.DataFrame,
    metro_df: pd.DataFrame,
    preserve_existing_metro_dist: bool,
) -> pd.DataFrame:
    """Assign nearest open metro station and distance using only stations with opening_date <= instance_date."""
    out = df.copy()
    if metro_df.empty:
        out["dist_nearest_metro_m"] = np.nan
        out["nearest_metro"] = "Unknown"
        out["metro_within_500m"] = 0
        out["metro_within_1000m"] = 0
        out["metro_band"] = ">2000m"
        return out

    open_dts = resolve_station_opening_dates(metro_df)
    station_work = metro_df.copy()
    station_work["_open_dt"] = open_dts
    open_lookup: dict[str, pd.Timestamp] = {}
    for s_name, s_dt in zip(station_work["station_name"], station_work["_open_dt"], strict=True):
        k = str(s_name).strip().lower()
        if k and (k not in open_lookup or s_dt < open_lookup[k]):
            open_lookup[k] = s_dt

    tx_dt = pd.to_datetime(out["instance_date"], errors="coerce").fillna(pd.Timestamp("2015-01-01"))
    min_tx_dt = tx_dt.min()
    max_tx_dt = tx_dt.max()
    cutoffs = sorted(ts for ts in set(open_dts) if min_tx_dt < ts <= max_tx_dt)
    if cutoffs:
        cutoff_ns = np.array([ts.to_datetime64() for ts in cutoffs], dtype="datetime64[ns]")
        era_ids = np.searchsorted(cutoff_ns, tx_dt.to_numpy(dtype="datetime64[ns]"), side="right")
    else:
        era_ids = np.zeros(len(out), dtype=int)

    out["_metro_era"] = era_ids
    uniq_metro = out[["lat", "lon", "_metro_era"]].drop_duplicates().reset_index(drop=True)
    calc_dist = np.full(len(uniq_metro), np.nan, dtype=float)
    calc_name: list[str] = ["Unknown"] * len(uniq_metro)

    for era_id in sorted(uniq_metro["_metro_era"].unique().tolist()):
        row_mask = uniq_metro["_metro_era"] == era_id
        if era_id < len(cutoffs):
            st_sub = station_work.loc[station_work["_open_dt"] < cutoffs[era_id]]
        else:
            st_sub = station_work.loc[station_work["_open_dt"] <= max_tx_dt]
        if st_sub.empty:
            st_sub = station_work

        d_arr, n_list = haversine_nearest_m(
            uniq_metro.loc[row_mask, "lat"].to_numpy(dtype=float),
            uniq_metro.loc[row_mask, "lon"].to_numpy(dtype=float),
            st_sub["lat"].to_numpy(dtype=float),
            st_sub["lon"].to_numpy(dtype=float),
            st_sub["station_name"].astype(str).tolist(),
        )
        idx_pos = np.flatnonzero(row_mask.to_numpy())
        calc_dist[idx_pos] = np.round(d_arr, 1)
        if n_list is not None:
            for j, name_val in zip(idx_pos, n_list, strict=True):
                calc_name[int(j)] = name_val

    uniq_metro["_calc_dist"] = calc_dist
    uniq_metro["_calc_metro"] = calc_name

    out = out.merge(
        uniq_metro[["lat", "lon", "_metro_era", "_calc_dist", "_calc_metro"]],
        on=["lat", "lon", "_metro_era"],
        how="left",
    )

    if "nearest_metro" in out.columns:
        orig_clean = out["nearest_metro"].astype("string").str.strip()
        orig_open_dt = pd.to_datetime(orig_clean.str.lower().map(open_lookup), errors="coerce")
        orig_is_unopened = orig_open_dt.notna() & (orig_open_dt > tx_dt)
        valid_orig_name = (
            orig_clean.notna() & ~orig_clean.isin(["", "nan", "None", "<NA>"]) & ~orig_is_unopened
        )
        out["nearest_metro"] = np.where(
            valid_orig_name, orig_clean.astype(str), out["_calc_metro"].astype(str)
        )
    else:
        orig_is_unopened = pd.Series(False, index=out.index)
        out["nearest_metro"] = out["_calc_metro"].astype(str)

    if (
        preserve_existing_metro_dist
        and "metro_distance_m" in out.columns
        and out["metro_distance_m"].notna().any()
    ):
        existing_d = pd.to_numeric(out["metro_distance_m"], errors="coerce").to_numpy(dtype=float)
        use_existing = np.isfinite(existing_d) & ~orig_is_unopened.to_numpy(dtype=bool)
        out["dist_nearest_metro_m"] = np.where(
            use_existing, existing_d, out["_calc_dist"].to_numpy(dtype=float)
        )
    else:
        out["dist_nearest_metro_m"] = out["_calc_dist"].to_numpy(dtype=float)

    out["metro_within_500m"] = (out["dist_nearest_metro_m"] < 500.0).astype(int)
    out["metro_within_1000m"] = (out["dist_nearest_metro_m"] < 1000.0).astype(int)
    out["metro_band"] = metro_distance_band(out["dist_nearest_metro_m"])
    return out.drop(columns=["_metro_era", "_calc_dist", "_calc_metro"])


def enrich_with_geo_features(
    tx_df: pd.DataFrame,
    metro_df: pd.DataFrame,
    projects_df: pd.DataFrame | None = None,
    cache_dir: Path = PATHS.cache,
    raw_dir: Path = PATHS.raw,
    allow_nominatim: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Geocode unique locations and attach time-aware metro & immutable spatial proximity features."""
    geocoded, geo_stats = geocode_transactions(
        tx_df,
        projects_df=projects_df,
        cache_path=cache_dir / "geocode.sqlite",
        allow_nominatim=allow_nominatim,
    )

    has_dates = "instance_date" in geocoded.columns
    preserve_dist = bool(
        "metro_distance_m" in geocoded.columns and bool(geocoded["metro_distance_m"].notna().any())
    )

    # Non-metro spatial features depend only on coordinates
    coord_cols = ["lat", "lon"]
    if not has_dates and preserve_dist:
        coord_cols.append("metro_distance_m")

    unique_coords = geocoded[coord_cols].drop_duplicates().reset_index(drop=True)
    pois_df = load_cached_pois(cache_dir)
    bus_df = load_gtfs_bus_stops(raw_dir)

    enriched_coords = compute_proximity_features(
        unique_coords,
        metro_df=metro_df,
        pois_df=pois_df,
        bus_stops_df=bus_df,
        preserve_existing_metro_dist=preserve_dist and not has_dates,
        include_metro=not has_dates,
    ).drop_duplicates(subset=coord_cols)

    added_cols = [c for c in enriched_coords.columns if c not in coord_cols]
    has_orig_metro = not has_dates and "nearest_metro" in geocoded.columns
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

    if has_dates:
        out = _attach_time_aware_metro_features(
            out, metro_df=metro_df, preserve_existing_metro_dist=preserve_dist
        )

    if len(out) != len(geocoded):
        raise RuntimeError(f"Geo merge changed row count: {len(geocoded)} -> {len(out)}")
    return out, geo_stats
