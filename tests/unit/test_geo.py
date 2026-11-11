"""Unit tests for geo distance, Nominatim SQLite cache, and boundary loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dfv.data.synth import SyntheticBundle
from dfv.geo.boundaries import load_area_boundaries, summarize_area_geometries
from dfv.geo.geocode import GeocodeCache, geocode_transactions
from dfv.geo.stations import (
    compute_proximity_features,
    haversine_pair_m,
    metro_distance_band,
    stations_table_summary,
)


def test_haversine_and_metro_band() -> None:
    d_zero = haversine_pair_m(25.1972, 55.2744, 25.1972, 55.2744)
    assert d_zero < 1e-6
    d_burj_to_business = haversine_pair_m(25.2014, 55.2696, 25.1912, 55.2603)
    assert 1000.0 < d_burj_to_business < 2000.0
    bands = metro_distance_band(np.array([250.0, 750.0, 1500.0, 3500.0]))
    assert list(bands) == ["<500m", "500-1000m", "1000-2000m", ">2000m"]


def test_geocode_sqlite_cache_and_fallback(tmp_path: Path) -> None:
    cache = GeocodeCache(tmp_path / "geo.sqlite")
    cache.put(
        "Burj Vista, Downtown Dubai, Dubai, UAE", 25.198, 55.272, "nominatim_project", "nominatim"
    )
    hit = cache.get("Burj Vista, Downtown Dubai, Dubai, UAE")
    assert hit is not None
    assert abs(hit[0] - 25.198) < 1e-6

    df = pd.DataFrame(
        {
            "project_name": ["Burj Vista", "Unknown Bldg"],
            "area_name": ["Downtown Dubai", "Dubai Marina"],
        }
    )
    geocoded, stats = geocode_transactions(
        df, cache_path=tmp_path / "geo.sqlite", allow_nominatim=False
    )
    assert abs(geocoded.loc[0, "lat"] - 25.198) < 1e-6
    assert geocoded.loc[0, "geocode_confidence"] == "nominatim_project"
    assert geocoded.loc[1, "geocode_confidence"] == "area_centroid"
    assert stats["total_rows"] == 2


def test_proximity_and_boundaries(synthetic_bundle: SyntheticBundle, tmp_path: Path) -> None:
    tx_sub = synthetic_bundle.transactions.head(30).copy()
    prox = compute_proximity_features(
        tx_sub[["lat", "lon", "nearest_metro", "metro_distance_m"]],
        synthetic_bundle.metro_stations,
    )
    assert (prox["dist_nearest_mall_m"] > 0).all()
    assert (prox["dist_airport_m"] > 0).all()
    assert prox["metro_within_500m"].isin([0, 1]).all()

    gdf, source = load_area_boundaries(raw_dir=tmp_path, cache_dir=tmp_path)
    assert not gdf.empty
    assert "centroid_buffer" in source
    summary = summarize_area_geometries(tx_sub)
    assert not summary.empty
    st_sum = stations_table_summary(synthetic_bundle.metro_stations)
    assert st_sum["total_stations"] == len(synthetic_bundle.metro_stations)
