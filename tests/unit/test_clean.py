"""Unit tests for every DLD cleaning rule, edge cases, and Dubai realism checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dfv.config import PATHS, SQFT_PER_SQM
from dfv.data.clean import (
    clean_rent_contracts,
    clean_transactions,
    load_name_map,
    parse_rooms_from_label,
    run_realism_checks,
    write_cleaning_report,
)
from dfv.data.synth import SyntheticBundle


def test_parse_rooms_from_label() -> None:
    s = pd.Series(["Studio", "1 B/R", "2 B/R", "PENTHOUSE", "Single Room", None])
    parsed = parse_rooms_from_label(s)
    assert list(parsed.iloc[:5]) == [0.0, 1.0, 2.0, 4.0, 1.0]
    assert pd.isna(parsed.iloc[5])


def test_clean_transactions_removes_dirty_and_flags_outliers(
    dirty_synthetic_bundle: SyntheticBundle,
) -> None:
    name_map = load_name_map(PATHS.reference / "name_map.csv")
    cleaned, logs = clean_transactions(dirty_synthetic_bundle.transactions, name_map=name_map)
    # 626 input: 5 Mortgage + 5 Commercial + 3 neg area + 3 zero worth = 16 dropped -> 610 remain
    assert len(cleaned) == 610
    assert (cleaned["area_sqm"] > 0).all()
    assert (cleaned["actual_worth"] > 0).all()
    assert cleaned["flag_duplicate"].sum() >= 1
    assert cleaned["flag_non_market"].sum() >= 6
    assert cleaned["flag_outlier"].sum() >= 6
    assert abs(cleaned.loc[0, "area_sqft"] - cleaned.loc[0, "area_sqm"] * SQFT_PER_SQM) < 1.0
    assert len(logs) >= 6


def test_clean_transactions_handles_hf_style_columns() -> None:
    raw = pd.DataFrame(
        {
            "procedure_name": ["Sell - Pre registration", "Sell", "Mortgage Registration"],
            "instance_date": ["22-08-2024", "15-05-2024", "15-05-2024"],
            "property_type": ["Unit", "Villa", "Unit"],
            "property_sub_type": ["Flat", "Villa", "Flat"],
            "property_usage": ["Residential", "Residential", "Residential"],
            "area_name": ["Marsa Dubai", "Burj Khalifa", "Marsa Dubai"],
            "master_project": ["Dubai Marina", "Burj Khalifa", "Dubai Marina"],
            "building_name": ["Marina Gate 1", None, "X"],
            "rooms_label": ["2 B/R", None, "1 B/R"],
            "area_sqft": [1200.0, 3500.0, 800.0],
            "actual_worth": [3000000.0, 9500000.0, 1000000.0],
            "has_parking": [1, 1, 1],
        }
    )
    name_map = load_name_map(PATHS.reference / "name_map.csv")
    cleaned, _ = clean_transactions(raw, name_map=name_map)
    assert len(cleaned) == 2
    assert set(cleaned["area_name"]) == {"Dubai Marina", "Downtown Dubai"}
    assert cleaned.loc[cleaned["area_name"] == "Dubai Marina", "is_offplan"].iloc[0] == 1
    assert cleaned.loc[cleaned["property_type"] == "Villa", "rooms_was_imputed"].iloc[0]


def test_clean_rent_contracts_and_realism_checks(
    synthetic_bundle: SyntheticBundle, tmp_path: Path
) -> None:
    name_map = load_name_map(PATHS.reference / "name_map.csv")
    tx_clean, tx_logs = clean_transactions(synthetic_bundle.transactions, name_map=name_map)
    rent_clean, rent_logs = clean_rent_contracts(synthetic_bundle.rent_contracts, name_map=name_map)
    assert len(rent_clean) == len(synthetic_bundle.rent_contracts)
    realism = run_realism_checks(tx_clean)
    assert (
        realism["downtown_vs_dubai_south"]["downtown"]
        > realism["downtown_vs_dubai_south"]["dubai_south"]
    )
    rep = write_cleaning_report(tx_logs, rent_logs, realism, tmp_path / "clean_report.md")
    assert rep.exists()


def test_realism_checks_raises_on_inverted_areas() -> None:
    bad = pd.DataFrame(
        {
            "instance_date": ["2024-06-01", "2024-06-01"],
            "area_name": ["Downtown Dubai", "Dubai South"],
            "area_sqm": [100.0, 100.0],
            "actual_worth": [500000.0, 3000000.0],
            "price_per_sqm": [5000.0, 30000.0],
            "flag_outlier": [False, False],
            "flag_non_market": [False, False],
            "flag_duplicate": [False, False],
        }
    )
    with pytest.raises(ValueError, match="Realism check failed"):
        run_realism_checks(bad)
