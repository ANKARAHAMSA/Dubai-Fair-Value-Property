"""Unit tests for data acquisition, local ingestion, manifest, and synthetic fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dfv.config import Paths
from dfv.data.download import build_sample_dataset, compute_sha256, ingest_data


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
