"""Unit tests for verdict boundary logic, thin segments, and confidence labels."""

from __future__ import annotations

import pytest

from dfv.models.verdict import compute_verdict


def test_verdict_boundaries_at_L_and_U() -> None:
    # P == L -> Fair (L <= P <= U)
    v_at_l = compute_verdict(
        asking_price=1_800_000,
        predicted_median=2_000_000,
        low=1_800_000,
        high=2_200_000,
        segment_tx_count=40,
    )
    assert v_at_l.verdict == "Fair"
    assert v_at_l.confidence == "High"

    # P == U -> Fair (L <= P <= U)
    v_at_u = compute_verdict(
        asking_price=2_200_000,
        predicted_median=2_000_000,
        low=1_800_000,
        high=2_200_000,
        segment_tx_count=40,
    )
    assert v_at_u.verdict == "Fair"

    # P < L -> Underpriced
    v_under = compute_verdict(
        asking_price=1_799_999,
        predicted_median=2_000_000,
        low=1_800_000,
        high=2_200_000,
        segment_tx_count=40,
    )
    assert v_under.verdict == "Underpriced"
    assert v_under.gap_pct < 0

    # P > U -> Overpriced
    v_over = compute_verdict(
        asking_price=2_200_001,
        predicted_median=2_000_000,
        low=1_800_000,
        high=2_200_000,
        segment_tx_count=40,
    )
    assert v_over.verdict == "Overpriced"
    assert v_over.gap_pct > 0


def test_verdict_thin_segment_returns_insufficient_data_with_range() -> None:
    # < 15 comparable sales (document threshold in model_card & interview_notes) must abstain
    res = compute_verdict(
        asking_price=1_500_000,
        predicted_median=2_000_000,
        low=1_700_000,
        high=2_300_000,
        segment_tx_count=14,
    )
    assert res.verdict == "insufficient data"
    assert res.low == 1_700_000
    assert res.high == 2_300_000
    assert res.confidence == "Low"

    # At exactly 15 comparable sales -> active directional verdict enabled
    res_15 = compute_verdict(
        asking_price=1_500_000,
        predicted_median=2_000_000,
        low=1_700_000,
        high=2_300_000,
        segment_tx_count=15,
    )
    assert res_15.verdict == "Underpriced"


def test_verdict_high_volume_wide_interval_explains_low_confidence() -> None:
    res = compute_verdict(
        asking_price=2_050_000,
        predicted_median=2_000_000,
        low=1_300_000,
        high=2_700_000,
        segment_tx_count=50,
    )
    assert res.verdict == "Fair"
    assert res.confidence == "Low"
    assert "wide 90% uncertainty band" in res.reason


def test_verdict_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        compute_verdict(0, 1000, 900, 1100)
    with pytest.raises(ValueError):
        compute_verdict(1000, 1000, 1200, 900)
