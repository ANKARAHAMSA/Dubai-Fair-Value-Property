"""Unit tests for data acquisition, local ingestion, manifest, and synthetic fallback."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import pytest

from dfv.config import PATHS, Paths
from dfv.data.download import (
    _download_with_retries,
    _extract_gtfs_metro_stops,
    _pull_dubai_pulse_api,
    build_sample_dataset,
    compute_sha256,
    ingest_data,
)


def _tmp_paths(tmp_path: Path) -> Paths:
    p = Paths(
        root=tmp_path,
        data=tmp_path / "data",
        raw=tmp_path / "data" / "raw",
        interim=tmp_path / "data" / "interim",
        processed=tmp_path / "data" / "processed",
        sample=tmp_path / "data" / "sample",
        reference=tmp_path / "data" / "reference",
        cache=tmp_path / "data" / "cache",
        docs=tmp_path / "docs",
        reports=tmp_path / "reports",
        figures=tmp_path / "reports" / "figures",
        notebooks=tmp_path / "notebooks",
        models=tmp_path / "data" / "processed" / "models",
        sample_models=tmp_path / "data" / "sample" / "models",
    )
    for d in (p.raw, p.interim, p.processed, p.sample, p.reference, p.cache, p.docs, p.reports):
        d.mkdir(parents=True, exist_ok=True)
    return p


def test_ingest_offline_falls_back_to_synthetic_and_builds_sample(tmp_path: Path) -> None:
    p = _tmp_paths(tmp_path)
    manifest = ingest_data(paths=p, allow_network=False, allow_unofficial_fallback=False)
    assert manifest["sources_summary"]["transactions"] == "synthetic_fallback"
    assert (p.raw / "MANIFEST.json").exists()
    assert (p.docs / "data_dictionary.md").exists()
    assert (p.interim / "transactions_raw.parquet").exists()
    assert (p.sample / "transactions.parquet").exists()


def test_ingest_manual_csv_prioritized_and_hashed(tmp_path: Path) -> None:
    p = _tmp_paths(tmp_path)
    manual_csv = p.raw / "dld_transactions-open.csv"
    pd.DataFrame(
        {
            "trans_group_en": ["Sales"],
            "procedure_name_en": ["Sell"],
            "instance_date": ["15-05-2024"],
            "property_type_en": ["Unit"],
            "property_sub_type_en": ["Flat"],
            "property_usage_en": ["Residential"],
            "area_name_en": ["Dubai Marina"],
            "rooms_en": ["2 B/R"],
            "procedure_area": [110.0],
            "actual_worth": [2200000.0],
        }
    ).to_csv(manual_csv, index=False)

    manifest = ingest_data(paths=p, allow_network=False, allow_unofficial_fallback=False)
    assert manifest["sources_summary"]["transactions"] == "official_dubaipulse_manual"
    saved_manifest = json.loads((p.raw / "MANIFEST.json").read_text(encoding="utf-8"))
    assert saved_manifest["datasets"][0]["sha256"] == compute_sha256(manual_csv)


def test_build_sample_dataset_under_size_limit(tmp_path: Path) -> None:
    p = _tmp_paths(tmp_path)
    out = build_sample_dataset(paths=p, seed=11)
    total_bytes = sum(path.stat().st_size for path in out.values())
    assert total_bytes < 5 * 1024 * 1024


def test_extract_gtfs_metro_stops_normalizes_and_deduplicates(tmp_path: Path) -> None:
    gtfs_zip = tmp_path / "rta_gtfs.zip"
    stops_csv = pd.DataFrame(
        {
            "stop_name": [
                "Expo Metro Station",
                "Jabal Ali Metro Station 3",
                "Jabal Ali Metro Station 4",
                "Union (Green Line ) Metro Station 1",
                "Business Bay Metro Station 1",
                "Mashreq Metro Station",
                "Al Safa Metro Station",
                "Umm Al Sheif Metro Station",
                "Etisalat Metro Station",
                "Burj Khalifa/ Dubai Mall Metro Station 1",
                "Bus Stop Random 12",
            ],
            "stop_lat": [
                24.9635,
                25.0598,
                25.0600,
                25.2661,
                25.1913,
                25.1145,
                25.1820,
                25.1265,
                25.2630,
                25.2010,
                25.10,
            ],
            "stop_lon": [
                55.1461,
                55.1252,
                55.1254,
                55.3136,
                55.2604,
                55.1905,
                55.2380,
                55.2075,
                55.3910,
                55.2690,
                55.20,
            ],
        }
    ).to_csv(index=False)
    with zipfile.ZipFile(gtfs_zip, "w") as zf:
        zf.writestr("stops.txt", stops_csv)

    extracted = _extract_gtfs_metro_stops(gtfs_zip, PATHS.reference / "stations_opening.csv")
    names = set(extracted["station_name"].tolist())
    assert "Expo 2020" in names
    assert "Jebel Ali" in names
    assert "Union" in names
    assert "Business Bay" in names
    assert "InsuranceMarket" in names
    assert "Onpassive" in names
    assert "Equiti" in names
    assert "e&" in names
    assert "Burj Khalifa / Dubai Mall" in names
    assert not any("Bus Stop" in n for n in names)
    expo_row = extracted.loc[extracted["station_name"] == "Expo 2020"].iloc[0]
    assert bool(expo_row["is_route_2020"]) is True
    assert str(expo_row["opening_date"]) == "2021-06-01"


def test_download_with_retries_and_dubai_pulse_api_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "dl.csv"
    payload_bytes = b"col1,col2\n1,2\n"

    class _FakeStreamResp:
        def __init__(self, fail: bool = False) -> None:
            self._fail = fail

        def __enter__(self) -> _FakeStreamResp:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            if self._fail:
                raise httpx.HTTPStatusError(
                    "500",
                    request=httpx.Request("GET", "https://example.com"),
                    response=httpx.Response(500),
                )

        def iter_bytes(self, chunk_size: int = 1024) -> io.BytesIO:  # type: ignore[override]
            return iter([payload_bytes])  # type: ignore[return-value]

    monkeypatch.setattr(
        "dfv.data.download.httpx.stream", lambda *_a, **_kw: _FakeStreamResp(fail=False)
    )
    assert _download_with_retries("https://example.com/file.csv", dest, max_retries=1) is True
    assert dest.read_bytes() == payload_bytes

    monkeypatch.setattr("dfv.data.download.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "dfv.data.download.httpx.stream", lambda *_a, **_kw: _FakeStreamResp(fail=True)
    )
    assert (
        _download_with_retries("https://example.com/bad.csv", tmp_path / "bad.csv", max_retries=2)
        is False
    )
    assert not (tmp_path / "bad.csv.part").exists()

    # Dubai Pulse API OAuth + pagination
    calls = {"get": 0}

    def _fake_post(*_a: object, **_kw: object) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "tok-123"},
            request=httpx.Request("POST", "https://api.dubaipulse.gov.ae/oauth"),
        )

    def _fake_get(*_a: object, **_kw: object) -> httpx.Response:
        calls["get"] += 1
        req = httpx.Request("GET", "https://api.dubaipulse.gov.ae/open/dld/test")
        if calls["get"] == 1:
            return httpx.Response(429, request=req)
        return httpx.Response(200, json={"results": [{"a": 1, "b": 2}]}, request=req)

    monkeypatch.setattr("dfv.data.download.httpx.post", _fake_post)
    monkeypatch.setattr("dfv.data.download.httpx.get", _fake_get)
    api_csv = tmp_path / "api_out.csv"
    assert _pull_dubai_pulse_api("id", "sec", "dld_transactions-open-api", api_csv) is True
    assert api_csv.exists() and len(pd.read_csv(api_csv)) == 1
