"""Unit tests for the explicit schema adapter layer and data dictionary generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dfv.data.schema import (
    TRANSACTIONS_COLUMN_MAP,
    adapt_dataframe,
    canonical_summary_dict,
    to_snake_case,
    write_data_dictionary,
)


def test_to_snake_case() -> None:
    assert to_snake_case("built_up_area_(sqft)") == "built_up_area_sqft"
    assert to_snake_case("Area Name EN") == "area_name_en"
    assert to_snake_case("  Price / SQM ") == "price_sqm"


def test_adapt_dataframe_maps_portal_and_hf_aliases() -> None:
    hf_raw = pd.DataFrame(
        {
            "date": ["22-08-2024"],
            "built_up_area_(sqft)": [1000.0],
            "selling_price": [1500000.0],
            "bedrooms": ["2 B/R"],
            "area_name": ["Marsa Dubai"],
        }
    )
    adapted = adapt_dataframe(hf_raw, TRANSACTIONS_COLUMN_MAP)
    assert "instance_date" in adapted.columns
    assert "area_sqft" in adapted.columns
    assert "actual_worth" in adapted.columns
    assert "rooms_label" in adapted.columns
    assert adapted.loc[0, "actual_worth"] == 1500000.0


def test_write_data_dictionary_creates_markdown(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "area_name": ["Dubai Marina", "Business Bay"],
            "price_per_sqm": [18000.0, None],
            "instance_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        }
    )
    out = tmp_path / "data_dictionary.md"
    write_data_dictionary({"transactions_raw": (df, "unit_test")}, out)
    text = out.read_text(encoding="utf-8")
    assert "transactions_raw" in text
    assert "50.0%" in text
    summary = canonical_summary_dict(df)
    assert summary["rows"] == 2
    assert summary["null_rates"]["price_per_sqm"] == 0.5
