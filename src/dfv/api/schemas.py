"""Pydantic v2 request and response schemas for DubaiFairValue FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Input payload for POST /predict."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    area_name: str = Field(..., min_length=2, max_length=120)
    property_type: Literal["Unit", "Villa"] = "Unit"
    property_sub_type: str | None = Field(default=None, max_length=60)
    rooms: int = Field(..., ge=0, le=10)
    area_sqm: float = Field(..., ge=15.0, le=5000.0)
    asking_price_aed: float = Field(..., ge=50_000.0, le=500_000_000.0)
    is_offplan: bool = False
    project_name: str | None = Field(default=None, max_length=160)
    developer_name: str | None = Field(default=None, max_length=160)
    has_parking: bool = True
    months_to_completion: float | None = Field(default=None, ge=0.0, le=120.0)
    dist_nearest_metro_m: float | None = Field(default=None, ge=0.0, le=50_000.0)


class DriverItemSchema(BaseModel):
    """Individual SHAP feature contribution converted to AED."""

    feature: str
    aed_impact: float
    text: str


class PredictResponse(BaseModel):
    """Response payload for POST /predict."""

    estimate: float
    low: float
    high: float
    verdict: str
    gap_pct: float
    confidence: str
    reason: str
    drivers: list[DriverItemSchema]
    model_version: str
    data_through: str


class AreaSummarySchema(BaseModel):
    """Summary metrics for a Dubai residential community."""

    area_name: str
    median_ppsqm: float | None = None
    median_price_aed: float | None = None
    gross_yield_pct: float | None = None
    risk_score: float | None = None
    yield_per_risk_score: float | None = None
    yoy_growth_pct: float | None = None
    cumulative_growth_pct: float | None = None
    tx_count: int = 0
    lat: float | None = None
    lon: float | None = None


class AreaDetailResponse(AreaSummarySchema):
    """Community detail view with quarterly historical trajectory."""

    trend: list[dict[str, Any]]
    yield_caveat: str


class YieldRankingResponse(BaseModel):
    """Response payload for GET /yield."""

    budget_aed: float | None = None
    rooms: int | None = None
    property_type: str | None = None
    caveat: str
    results: list[dict[str, Any]]


class SearchRequest(BaseModel):
    """Input payload for POST /search."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=2, max_length=500)
    limit: int | None = Field(default=None, ge=1, le=100)


class SearchResponse(BaseModel):
    """Response payload for POST /search."""

    filters: dict[str, Any]
    results: list[dict[str, Any]]
    explanation: str
    used_llm: bool
    parser_note: str
    total_matches: int
    summary: dict[str, Any]


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str
    model_version: str
    data_date: str
    dataset_mode: str
    known_areas_count: int
