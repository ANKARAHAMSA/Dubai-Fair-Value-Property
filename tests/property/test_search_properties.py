"""Hypothesis property-based tests: filter executor never returns rows violating bounds."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from dfv.config import PATHS
from dfv.search.executor import execute_property_search
from dfv.search.schema import PropertyFilters

SAMPLE_FEATURES = PATHS.sample / "features.parquet"
SAMPLE_YIELD = PATHS.sample / "analysis" / "yield_risk.json"


@given(
    rooms=st.one_of(st.none(), st.integers(min_value=0, max_value=4)),
    max_price=st.one_of(st.none(), st.floats(min_value=600_000.0, max_value=8_000_000.0)),
    near_metro=st.one_of(st.none(), st.floats(min_value=300.0, max_value=4000.0)),
    off_plan=st.one_of(st.none(), st.booleans()),
    limit=st.integers(min_value=1, max_value=40),
)
@settings(max_examples=20, deadline=None)
def test_filter_executor_never_violates_active_bounds(
    rooms: int | None,
    max_price: float | None,
    near_metro: float | None,
    off_plan: bool | None,
    limit: int,
) -> None:
    filters = PropertyFilters(
        rooms=rooms,
        max_price_aed=max_price,
        near_metro_m=near_metro,
        off_plan=off_plan,
        limit=limit,
    )
    out = execute_property_search(filters, SAMPLE_FEATURES, SAMPLE_YIELD)
    rows = out["results"]
    assert len(rows) <= limit
    assert out["returned_count"] == len(rows)
    assert out["total_matches"] >= len(rows)

    for row in rows:
        if rooms is not None:
            assert int(row["rooms"]) == rooms
        if max_price is not None:
            assert float(row["actual_worth"]) <= max_price + 1.0
        if near_metro is not None:
            assert float(row["dist_nearest_metro_m"]) <= near_metro + 1.0
        if off_plan is not None:
            assert int(row["is_offplan"]) == (1 if off_plan else 0)
