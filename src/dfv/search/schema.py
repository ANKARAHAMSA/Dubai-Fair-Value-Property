"""Strict Pydantic v2 filter schema and prompt/SQL-injection guardrails."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(?:all\s+)?(?:previous|prior)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:system\s*prompt|reveal\s+(?:your\s+)?(?:prompt|api[_\s]?key|secret))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:drop\s+table|delete\s+from|truncate\s+table|alter\s+table|insert\s+into|union\s+select)\b",
        re.IGNORECASE,
    ),
    re.compile(r";\s*(?:select|drop|delete|update|insert|pragma|attach|copy)\b", re.IGNORECASE),
    re.compile(r"\b(?:__import__|eval\s*\(|exec\s*\(|os\.system|subprocess)\b", re.IGNORECASE),
    re.compile(r"</?(?:system|assistant|tool_call)>", re.IGNORECASE),
)

SQL_AREA_REJECT: re.Pattern[str] = re.compile(
    r"[;\"]|--|/\*|\*/|''|(?<![A-Za-z])'|'(?![A-Za-z])|\b(?:select|drop|union|insert|delete)\b",
    re.I,
)


class SearchGuardrailError(ValueError):
    """Raised when natural-language input contains prompt-injection or SQL-injection patterns."""


def check_query_safety(query: str) -> str:
    """Validate search text against length bounds and prompt/SQL injection patterns."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Search query cannot be empty.")
    if len(cleaned) > 500:
        raise SearchGuardrailError("Search query exceeds 500-character safety limit.")
    for pat in INJECTION_PATTERNS:
        if pat.search(cleaned):
            raise SearchGuardrailError(
                "Query rejected by guardrail: unsupported instruction or SQL syntax detected."
            )
    return cleaned


class PropertyFilters(BaseModel):
    """Validated structured search filters produced by the LLM or rule-based parser."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rooms: int | None = Field(default=None, ge=0, le=10)
    min_rooms: int | None = Field(default=None, ge=0, le=10)
    property_type: Literal["Unit", "Villa"] | None = None
    min_price_aed: float | None = Field(default=None, ge=50_000.0, le=500_000_000.0)
    max_price_aed: float | None = Field(default=None, ge=50_000.0, le=500_000_000.0)
    min_area_sqm: float | None = Field(default=None, ge=15.0, le=10_000.0)
    max_area_sqm: float | None = Field(default=None, ge=15.0, le=10_000.0)
    areas: list[str] = Field(default_factory=list, max_length=15)
    near_metro_m: float | None = Field(default=None, ge=50.0, le=20_000.0)
    min_yield_pct: float | None = Field(default=None, ge=0.0, le=25.0)
    off_plan: bool | None = None
    limit: int = Field(default=25, ge=1, le=100)

    @field_validator("areas")
    @classmethod
    def _validate_area_tokens(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        for val in values:
            token = val.strip()
            if not token:
                continue
            if SQL_AREA_REJECT.search(token):
                raise ValueError(f"Unsafe or malformed area filter token: {token!r}")
            out.append(token[:80])
        return out

    @model_validator(mode="after")
    def _validate_bounds(self) -> PropertyFilters:
        if (
            self.min_price_aed is not None
            and self.max_price_aed is not None
            and self.min_price_aed > self.max_price_aed
        ):
            raise ValueError("min_price_aed cannot exceed max_price_aed.")
        if (
            self.min_area_sqm is not None
            and self.max_area_sqm is not None
            and self.min_area_sqm > self.max_area_sqm
        ):
            raise ValueError("min_area_sqm cannot exceed max_area_sqm.")
        return self


def _fmt_aed_compact(val: float) -> str:
    if val >= 1_000_000:
        return f"AED {val / 1_000_000:.2f}M".rstrip("0").rstrip(".") + (
            "M" if "." not in f"{val / 1_000_000:.2f}".rstrip("0").rstrip(".") else ""
        )
    if val >= 1_000:
        return f"AED {val / 1_000:.0f}k"
    return f"AED {val:,.0f}"


def summarize_filters(filters: PropertyFilters) -> str:
    """Render an explicit, human-readable interpretation of parsed filters."""
    parts: list[str] = []
    if filters.rooms is not None:
        parts.append("Studio" if filters.rooms == 0 else f"{filters.rooms} bed")
    elif filters.min_rooms is not None:
        parts.append(f">= {filters.min_rooms} bed")
    if filters.property_type is not None:
        parts.append(filters.property_type)
    if filters.off_plan is True:
        parts.append("Off-plan")
    elif filters.off_plan is False:
        parts.append("Ready")
    if filters.areas:
        parts.append("in " + ", ".join(filters.areas))
    if filters.min_price_aed is not None and filters.max_price_aed is not None:
        parts.append(
            f"AED {filters.min_price_aed / 1_000_000:.2f}M - {filters.max_price_aed / 1_000_000:.2f}M"
        )
    elif filters.max_price_aed is not None:
        parts.append(f"<= AED {filters.max_price_aed / 1_000_000:.2f}M")
    elif filters.min_price_aed is not None:
        parts.append(f">= AED {filters.min_price_aed / 1_000_000:.2f}M")
    if filters.min_area_sqm is not None:
        parts.append(f">= {filters.min_area_sqm:.0f} sqm")
    if filters.max_area_sqm is not None:
        parts.append(f"<= {filters.max_area_sqm:.0f} sqm")
    if filters.near_metro_m is not None:
        parts.append(f"<= {filters.near_metro_m:.0f} m from metro")
    if filters.min_yield_pct is not None:
        parts.append(f">= {filters.min_yield_pct:.1f}% yield")

    if not parts:
        return "I understood: broad residential market view (no hard filters, showing latest listings)."
    return "I understood: " + ", ".join(parts) + "."
