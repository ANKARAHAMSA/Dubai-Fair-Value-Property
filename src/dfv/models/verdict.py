"""Verdict classification, percentage gap, and prediction confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerdictResult:
    """Structured fair-value verdict for a candidate Dubai property."""

    verdict: str  # Underpriced | Fair | Overpriced | insufficient data
    estimate: float
    low: float
    high: float
    gap_pct: float
    confidence: str  # High | Medium | Low
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "estimate": round(float(self.estimate), 2),
            "low": round(float(self.low), 2),
            "high": round(float(self.high), 2),
            "gap_pct": round(float(self.gap_pct), 2),
            "confidence": self.confidence,
            "reason": self.reason,
        }


def compute_verdict(
    asking_price: float,
    predicted_median: float,
    low: float,
    high: float,
    segment_tx_count: int = 50,
    min_segment_samples: int = 5,
) -> VerdictResult:
    """Evaluate asking price P against calibrated interval [L, U] and median M.

    Guarantees:
    - Never outputs a verdict without an interval [low, high].
    - Returns 'insufficient data' when segment_tx_count < min_segment_samples.
    - Monotonic in asking price: P < L -> Underpriced, L <= P <= U -> Fair, P > U -> Overpriced.
    """
    if predicted_median <= 0 or low <= 0 or high <= 0 or low > high:
        raise ValueError(
            f"Invalid prediction interval [{low}, {high}] or median {predicted_median}."
        )
    if asking_price <= 0:
        raise ValueError("asking_price must be strictly positive.")

    # Ensure median lies inside [low, high] cleanly
    low_c = min(low, predicted_median)
    high_c = max(high, predicted_median)
    gap_pct = 100.0 * (asking_price - predicted_median) / predicted_median
    rel_width = (high_c - low_c) / predicted_median

    if segment_tx_count < min_segment_samples:
        return VerdictResult(
            verdict="insufficient data",
            estimate=predicted_median,
            low=low_c,
            high=high_c,
            gap_pct=gap_pct,
            confidence="Low",
            reason=(
                f"Thin comparable segment ({segment_tx_count} recent sales < minimum {min_segment_samples}); "
                "interval shown for reference only."
            ),
        )

    if rel_width <= 0.35 and segment_tx_count >= 25:
        confidence = "High"
    elif rel_width <= 0.55 and segment_tx_count >= 10:
        confidence = "Medium"
    else:
        confidence = "Low"

    if asking_price < low_c:
        verdict = "Underpriced"
        reason = (
            f"Asking price is {abs(gap_pct):.1f}% below estimated fair value and below the "
            f"calibrated 90% lower bound."
        )
    elif asking_price <= high_c:
        verdict = "Fair"
        reason = f"Asking price sits within the calibrated 90% fair-value range ({gap_pct:+.1f}% vs median)."
    else:
        verdict = "Overpriced"
        reason = (
            f"Asking price is {gap_pct:.1f}% above estimated fair value and exceeds the "
            f"calibrated 90% upper bound."
        )

    return VerdictResult(
        verdict=verdict,
        estimate=predicted_median,
        low=low_c,
        high=high_c,
        gap_pct=gap_pct,
        confidence=confidence,
        reason=reason,
    )
