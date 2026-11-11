"""Property-based tests for synthetic invariants and unit conversions."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from dfv.config import SQFT_PER_SQM


@given(
    area_sqm=st.floats(min_value=20.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    price_per_sqm=st.floats(
        min_value=2000.0, max_value=200000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=50)
def test_price_area_and_unit_roundtrip(area_sqm: float, price_per_sqm: float) -> None:
    total = price_per_sqm * area_sqm
    recovered_ppsqm = total / area_sqm
    assert abs(recovered_ppsqm - price_per_sqm) / price_per_sqm < 1e-9

    area_sqft = area_sqm * SQFT_PER_SQM
    recovered_sqm = area_sqft / SQFT_PER_SQM
    assert abs(recovered_sqm - area_sqm) / area_sqm < 1e-9
