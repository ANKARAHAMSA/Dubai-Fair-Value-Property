"""Batch runner for all 5 DubaiFairValue market analysis modules (`make analyze`)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from dfv.analysis.developer_premium import analyze_developer_premium
from dfv.analysis.events import analyze_market_events
from dfv.analysis.growth_map import build_quarterly_growth_table, render_growth_timelapse
from dfv.analysis.metro_effect import analyze_metro_effect
from dfv.analysis.yield_score import compute_yield_and_risk
from dfv.config import PATHS, Paths

logger = logging.getLogger("dfv.analysis.run_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_analysis_for_dataset(
    features_path: Path,
    rents_path: Path,
    stations_path: Path,
    sale_index_path: Path,
    events_csv_path: Path,
    out_dir: Path,
    figures_dir: Path | None = None,
    min_group_obs: int = 15,
) -> dict[str, Any]:
    """Run all 5 market analyses on a dataset folder and persist JSON + Parquet outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_df = pd.read_parquet(features_path)
    rent_df = pd.read_parquet(rents_path)
    stations_df = pd.read_parquet(stations_path)
    index_df = pd.read_parquet(sale_index_path)

    dev_res = analyze_developer_premium(feat_df, min_group_obs=min_group_obs)
    metro_res = analyze_metro_effect(feat_df, rent_df, stations_df)
    yield_res = compute_yield_and_risk(
        feat_df, rent_df, min_sales=min_group_obs, min_rents=max(5, min_group_obs // 2)
    )
    events_res = analyze_market_events(feat_df, index_df, events_csv_path)
    growth_df = build_quarterly_growth_table(feat_df, min_tx=max(2, min_group_obs // 5))

    (out_dir / "developer_premium.json").write_text(json.dumps(dev_res, indent=2), encoding="utf-8")
    (out_dir / "metro_effect.json").write_text(json.dumps(metro_res, indent=2), encoding="utf-8")
    (out_dir / "yield_risk.json").write_text(json.dumps(yield_res, indent=2), encoding="utf-8")
    (out_dir / "events_study.json").write_text(json.dumps(events_res, indent=2), encoding="utf-8")
    growth_df.to_parquet(out_dir / "growth_quarterly.parquet", index=False)

    if figures_dir is not None:
        render_growth_timelapse(growth_df, figures_dir)

    summary = {
        "rows_analyzed": int(len(feat_df)),
        "developer_premium_overall": dev_res.get("overall_area_quarter_fe", {}),
        "metro_sales_bands": metro_res.get("sales_distance_bands", {}),
        "metro_rents_bands": metro_res.get("rents_distance_bands", {}),
        "top_yield_areas": yield_res.get("area_ranking", [])[:5],
        "official_index_check": events_res.get("official_index_sanity_check", {}),
        "quarterly_growth_rows": int(len(growth_df)),
    }
    return summary


def run_all_analyses(paths: Paths = PATHS) -> dict[str, Any]:
    """Run market analyses for committed sample data and full processed data if available."""
    out: dict[str, Any] = {}
    events_csv = paths.reference / "events.csv"
    has_processed = (paths.processed / "features.parquet").exists()

    if (paths.sample / "features.parquet").exists():
        sample_summary = run_analysis_for_dataset(
            features_path=paths.sample / "features.parquet",
            rents_path=paths.sample / "rent_contracts.parquet",
            stations_path=paths.sample / "metro_stations.parquet",
            sale_index_path=paths.sample / "sale_index.parquet",
            events_csv_path=events_csv,
            out_dir=paths.sample / "analysis",
            figures_dir=paths.figures if not has_processed else None,
            min_group_obs=8,
        )
        out["sample"] = sample_summary
        logger.info("Sample market analyses complete.")

    if has_processed:
        proc_summary = run_analysis_for_dataset(
            features_path=paths.processed / "features.parquet",
            rents_path=paths.processed / "rent_contracts.parquet",
            stations_path=paths.processed / "metro_stations.parquet",
            sale_index_path=paths.processed / "sale_index.parquet",
            events_csv_path=events_csv,
            out_dir=paths.processed / "analysis",
            figures_dir=paths.figures,
            min_group_obs=20,
        )
        out["processed"] = proc_summary
        (paths.reports / "analysis_summary.json").write_text(
            json.dumps(proc_summary, indent=2), encoding="utf-8"
        )
        logger.info("Processed (full) market analyses complete.")
    elif "sample" in out:
        (paths.reports / "analysis_summary.json").write_text(
            json.dumps(out["sample"], indent=2), encoding="utf-8"
        )

    return out


def main() -> int:
    run_all_analyses(PATHS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
