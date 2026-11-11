"""Unit & planted-effect recovery tests for the 5 market analysis modules."""

from __future__ import annotations

from pathlib import Path

from dfv.analysis.developer_premium import (
    analyze_developer_premium,
    compute_matched_offplan_comparison,
    compute_time_to_catch_up,
)
from dfv.analysis.events import analyze_market_events, build_monthly_price_index
from dfv.analysis.growth_map import build_quarterly_growth_table, render_growth_timelapse
from dfv.analysis.metro_effect import analyze_metro_effect, run_route2020_event_study
from dfv.analysis.run_all import run_all_analyses
from dfv.analysis.yield_score import compute_yield_and_risk, recommend_areas_for_budget
from dfv.config import PATHS, Paths
from dfv.data.synth import generate_synthetic_bundle
from dfv.features.build import build_feature_matrix


def test_planted_metro_and_offplan_recovery() -> None:
    """Verify FE models recover the generator's planted structural effects within tolerance."""
    bundle = generate_synthetic_bundle(n_sales=4500, n_rents=4000, seed=42)
    feat_df, _, _ = build_feature_matrix(
        bundle.transactions,
        bundle.rent_contracts,
        bundle.metro_stations,
        projects_df=bundle.projects,
    )

    # 1. Metro proximity recovery (ground truth: <500m = 0.14 log, 500-1000m = 0.08 log, R2020 = 0.07 log)
    metro_res = analyze_metro_effect(feat_df, bundle.rent_contracts, bundle.metro_stations)
    sales_bands = metro_res["sales_distance_bands"]
    assert abs(sales_bands["band_lt500m"]["log_coef"] - 0.14) < 0.035
    assert abs(sales_bands["band_500_1000m"]["log_coef"] - 0.08) < 0.035
    assert sales_bands["band_lt500m"]["log_coef"] > sales_bands["band_500_1000m"]["log_coef"]

    rent_bands = metro_res["rents_distance_bands"]
    # Synthetic rents plant 0.7 * 0.14 = 0.098 for <500m
    assert abs(rent_bands["band_lt500m"]["log_coef"] - 0.098) < 0.035

    r2020 = run_route2020_event_study(feat_df, bundle.metro_stations, treatment_radius_m=1200.0)
    assert r2020["available"]
    assert r2020["did_post_effect"]["log_coef"] > 0.02
    assert abs(r2020["did_post_effect"]["log_coef"] - 0.07) < 0.045

    # 2. Off-plan vs ready "developer premium" recovery (planted ~0.06 + ~0.005 mean dev offset)
    dev_res = analyze_developer_premium(feat_df, min_group_obs=15)
    area_fe_coef = dev_res["overall_area_quarter_fe"]["log_coef"]
    proj_fe_coef = dev_res["overall_project_quarter_fe"]["log_coef"]
    assert abs(area_fe_coef - 0.065) < 0.025
    assert abs(proj_fe_coef - 0.065) < 0.025
    assert "association" in dev_res["caveat"].lower()

    matched = compute_matched_offplan_comparison(feat_df, min_per_side=4)
    assert matched["matched_cells"] > 0
    assert matched["matched_pct_premium"] is not None
    assert matched["matched_pct_premium"] > 2.0

    catchup = compute_time_to_catch_up(feat_df, min_cohort_rows=5)
    assert len(catchup) > 0
    for row in catchup:
        if row["median_catchup_months"] is not None:
            assert row["median_catchup_months"] > 0.0


def test_yield_risk_ranking_and_budget_recommender() -> None:
    """Verify trailing 12m yield calculation, New-contract preference, and budget filtering."""
    bundle = generate_synthetic_bundle(n_sales=3000, n_rents=3000, seed=7)
    res = compute_yield_and_risk(
        bundle.transactions, bundle.rent_contracts, min_sales=8, min_rents=6
    )

    assert "service charges" in res["caveat"].lower()
    assert "vacancy" in res["caveat"].lower()
    assert len(res["area_ranking"]) >= 5
    assert len(res["segment_ranking"]) >= 10

    first_seg = res["segment_ranking"][0]
    expected_yield = first_seg["median_rent_ppsqm"] / first_seg["median_sale_ppsqm"] * 100.0
    assert abs(first_seg["gross_yield_pct"] - expected_yield) <= 0.15
    assert first_seg["used_new_contracts_only"] is True

    top_budget = recommend_areas_for_budget(
        res["segment_ranking"], budget_aed=2_000_000.0, rooms=1, property_type="Unit", top_k=5
    )
    assert 1 <= len(top_budget) <= 5
    for item in top_budget:
        assert item["median_price_aed"] <= 2_000_000.0
        assert item["rooms"] == 1
        assert item["property_type"] == "Unit"


def test_events_its_and_growth_timelapse(tmp_path: Path) -> None:
    """Verify monthly index coherence, ITS/placebo tests, and GIF/PNG generation."""
    bundle = generate_synthetic_bundle(n_sales=2500, n_rents=1500, seed=11)
    idx_df = build_monthly_price_index(bundle.transactions, min_month_tx=3)
    assert not idx_df.empty
    assert "computed_index" in idx_df.columns

    ev_res = analyze_market_events(
        bundle.transactions, bundle.sale_index, PATHS.reference / "events.csv"
    )
    assert ev_res["official_index_sanity_check"]["available"]
    assert ev_res["official_index_sanity_check"]["pearson_corr"] > 0.70
    assert len(ev_res["events_analysis"]) >= 4
    assert any(e["its"] is not None for e in ev_res["events_analysis"])

    growth_df = build_quarterly_growth_table(bundle.transactions, min_tx=2)
    assert not growth_df.empty
    gif_path, snap_path = render_growth_timelapse(growth_df, tmp_path)
    assert gif_path.exists() and gif_path.stat().st_size > 1000
    assert snap_path.exists() and snap_path.stat().st_size > 1000


def test_run_all_analyses_batch_runner(tmp_path: Path) -> None:
    """Verify analysis/run_all.py batch pipeline writes JSON, parquet, and GIF outputs."""
    bundle = generate_synthetic_bundle(n_sales=1600, n_rents=1200, seed=19)
    feat_df, _, _ = build_feature_matrix(
        bundle.transactions,
        bundle.rent_contracts,
        bundle.metro_stations,
        projects_df=bundle.projects,
    )
    sample_dir = tmp_path / "data" / "sample"
    proc_dir = tmp_path / "data" / "processed"
    ref_dir = tmp_path / "data" / "reference"
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    for d in (sample_dir, proc_dir, ref_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    feat_df.to_parquet(sample_dir / "features.parquet", index=False)
    bundle.rent_contracts.to_parquet(sample_dir / "rent_contracts.parquet", index=False)
    bundle.metro_stations.to_parquet(sample_dir / "metro_stations.parquet", index=False)
    bundle.sale_index.to_parquet(sample_dir / "sale_index.parquet", index=False)
    (ref_dir / "events.csv").write_text(
        (PATHS.reference / "events.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )

    p = Paths(
        root=tmp_path,
        data=tmp_path / "data",
        raw=tmp_path / "data" / "raw",
        interim=tmp_path / "data" / "interim",
        processed=proc_dir,
        sample=sample_dir,
        reference=ref_dir,
        cache=tmp_path / "data" / "cache",
        docs=tmp_path / "docs",
        reports=reports_dir,
        figures=figures_dir,
        notebooks=tmp_path / "notebooks",
        models=proc_dir / "models",
        sample_models=sample_dir / "models",
    )

    summary = run_all_analyses(p)
    assert "sample" in summary
    assert "processed" not in summary
    assert (sample_dir / "analysis" / "developer_premium.json").exists()
    assert (sample_dir / "analysis" / "metro_effect.json").exists()
    assert (sample_dir / "analysis" / "yield_risk.json").exists()
    assert (sample_dir / "analysis" / "events_study.json").exists()
    assert (sample_dir / "analysis" / "growth_quarterly.parquet").exists()
    assert (reports_dir / "analysis_summary.json").exists()
    assert (figures_dir / "price_growth_timelapse.gif").exists()
