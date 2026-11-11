"""Property-based tests for interval ordering and verdict monotonicity."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from dfv.models.verdict import compute_verdict


@given(
    median=st.floats(
        min_value=300_000.0, max_value=20_000_000.0, allow_nan=False, allow_infinity=False
    ),
    half_width=st.floats(
        min_value=20_000.0, max_value=2_000_000.0, allow_nan=False, allow_infinity=False
    ),
    p1=st.floats(
        min_value=200_000.0, max_value=25_000_000.0, allow_nan=False, allow_infinity=False
    ),
    p2=st.floats(
        min_value=200_000.0, max_value=25_000_000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=60)
def test_verdict_monotonic_in_asking_price_and_ordered_bounds(
    median: float, half_width: float, p1: float, p2: float
) -> None:
    low = max(100_000.0, median - half_width)
    high = median + half_width
    assert low <= median <= high

    lo_p, hi_p = min(p1, p2), max(p1, p2)
    v_lo = compute_verdict(lo_p, median, low, high, segment_tx_count=30)
    v_hi = compute_verdict(hi_p, median, low, high, segment_tx_count=30)
    rank = {"Underpriced": 0, "Fair": 1, "Overpriced": 2}
    assert rank[v_lo.verdict] <= rank[v_hi.verdict]
