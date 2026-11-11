"""FastAPI application entrypoint (`make api`)."""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dfv.analysis.yield_score import YIELD_CAVEAT, recommend_areas_for_budget
from dfv.api.schemas import (
    AreaDetailResponse,
    AreaSummarySchema,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SearchRequest,
    SearchResponse,
    YieldRankingResponse,
)
from dfv.config import PATHS, get_data_mode
from dfv.models.predict import predict_single_property
from dfv.models.registry import ModelBundle, load_model_bundle
from dfv.search.executor import execute_property_search
from dfv.search.llm import parse_query_nl
from dfv.search.schema import SearchGuardrailError


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


@dataclass
class AppResources:
    """Loaded model, explainer, and cached analytical tables."""

    bundle: ModelBundle
    explainer: Any
    dataset_mode: str
    data_dir: Path
    features_path: Path
    yield_json_path: Path
    yield_data: dict[str, Any]
    growth_df: pd.DataFrame
    areas_list: list[AreaSummarySchema]
    areas_by_key: dict[str, AreaSummarySchema]
    known_areas: list[str]


def build_resources(
    model_dir: Path | None = None,
    data_dir: Path | None = None,
) -> AppResources:
    """Load model bundle (failing cleanly if metadata is missing) and cached data tables."""
    mode = get_data_mode()
    if data_dir is None:
        data_dir = PATHS.processed if mode == "processed" else PATHS.sample
    if model_dir is None:
        cand = data_dir / "models"
        model_dir = (
            cand
            if (cand / "metadata.json").exists() and (cand / "model_bundle.pkl").exists()
            else PATHS.sample_models
        )

    # Strict metadata enforcement: raises ModelMetadataError when invalid/missing
    bundle = load_model_bundle(model_dir)

    if hasattr(bundle.point_model, "compute_shap"):
        explainer: Any = bundle.point_model
    else:
        explainer = shap.TreeExplainer(bundle.point_model)

    features_path = data_dir / "features.parquet"
    if not features_path.exists():
        features_path = PATHS.sample / "features.parquet"

    analysis_dir = data_dir / "analysis"
    if not analysis_dir.exists():
        analysis_dir = PATHS.sample / "analysis"

    yield_json_path = analysis_dir / "yield_risk.json"
    yield_data: dict[str, Any] = {}
    if yield_json_path.exists():
        yield_data = json.loads(yield_json_path.read_text(encoding="utf-8"))

    growth_path = analysis_dir / "growth_quarterly.parquet"
    growth_df = pd.read_parquet(growth_path) if growth_path.exists() else pd.DataFrame()

    # Build area index combining model area_defaults, yield ranking, and latest quarter growth
    yield_by_area = {str(r["area_name"]): r for r in yield_data.get("area_ranking", [])}
    latest_growth: dict[str, dict[str, Any]] = {}
    if not growth_df.empty:
        max_q = str(growth_df["year_quarter"].max())
        for _, row in growth_df.loc[growth_df["year_quarter"] == max_q].iterrows():
            latest_growth[str(row["area_name"])] = {str(k): v for k, v in row.to_dict().items()}

    all_area_names = sorted(
        set(bundle.area_defaults.keys()) | set(yield_by_area.keys()) | set(latest_growth.keys())
    )
    areas_list: list[AreaSummarySchema] = []
    areas_by_key: dict[str, AreaSummarySchema] = {}
    for a_name in all_area_names:
        d = bundle.area_defaults.get(a_name, {})
        y = yield_by_area.get(a_name, {})
        g = latest_growth.get(a_name, {})
        med_ppsqm = (
            y.get("median_sale_ppsqm") or g.get("median_ppsqm") or d.get("area_trailing_12m_ppsqm")
        )
        item = AreaSummarySchema(
            area_name=a_name,
            median_ppsqm=round(float(med_ppsqm), 1) if med_ppsqm is not None else None,
            median_price_aed=round(float(y["median_price_aed"]), 0)
            if y.get("median_price_aed") is not None
            else None,
            gross_yield_pct=round(float(y["gross_yield_pct"]), 2)
            if y.get("gross_yield_pct") is not None
            else (
                round(float(d["area_trailing_12m_yield"]) * 100.0, 2)
                if d.get("area_trailing_12m_yield") is not None
                else None
            ),
            risk_score=round(float(y["risk_score"]), 1)
            if y.get("risk_score") is not None
            else None,
            yield_per_risk_score=round(float(y["yield_per_risk_score"]), 3)
            if y.get("yield_per_risk_score") is not None
            else None,
            yoy_growth_pct=float(g["yoy_growth_pct"])
            if g.get("yoy_growth_pct") is not None and pd.notna(g["yoy_growth_pct"])
            else None,
            cumulative_growth_pct=float(g["cumulative_growth_pct"])
            if g.get("cumulative_growth_pct") is not None and pd.notna(g["cumulative_growth_pct"])
            else None,
            tx_count=int(d.get("area_tx_count") or y.get("sales_12m_count") or 0),
            lat=round(float(d["lat"]), 6) if d.get("lat") is not None else None,
            lon=round(float(d["lon"]), 6) if d.get("lon") is not None else None,
        )
        areas_list.append(item)
        areas_by_key[a_name.lower()] = item
        areas_by_key[_slugify(a_name)] = item

    return AppResources(
        bundle=bundle,
        explainer=explainer,
        dataset_mode=str(bundle.metadata.get("dataset_mode", mode)),
        data_dir=data_dir,
        features_path=features_path,
        yield_json_path=yield_json_path,
        yield_data=yield_data,
        growth_df=growth_df,
        areas_list=areas_list,
        areas_by_key=areas_by_key,
        known_areas=all_area_names,
    )


def create_app(
    model_dir: Path | None = None,
    data_dir: Path | None = None,
) -> FastAPI:
    """Create FastAPI instance with strict metadata validation on startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.resources = build_resources(model_dir=model_dir, data_dir=data_dir)
        yield

    application = FastAPI(
        title="DubaiFairValue API",
        version="0.1.0",
        description="Residential property valuation, calibrated 90% intervals, SHAP drivers, yields, and safe NL search.",
        lifespan=lifespan,
    )

    cors_raw = os.getenv("DFV_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    allow_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    def _res() -> AppResources:
        res: AppResources | None = getattr(application.state, "resources", None)
        if res is None:
            res = build_resources(model_dir=model_dir, data_dir=data_dir)
            application.state.resources = res
        return res

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        res = _res()
        data_through = str(res.bundle.metadata["training_window"]["test_end"])
        return HealthResponse(
            status="ok",
            model_version=str(res.bundle.metadata["model_version"]),
            data_date=data_through,
            dataset_mode=res.dataset_mode,
            known_areas_count=len(res.known_areas),
        )

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        res = _res()
        result = predict_single_property(
            req.model_dump(),
            bundle=res.bundle,
            explainer_or_model=res.explainer,
        )
        return PredictResponse(**result)

    @application.get("/areas", response_model=list[AreaSummarySchema])
    def list_areas() -> list[AreaSummarySchema]:
        return _res().areas_list

    @application.get("/areas/{area_id}", response_model=AreaDetailResponse)
    def get_area(area_id: str) -> AreaDetailResponse:
        res = _res()
        item = res.areas_by_key.get(area_id.strip().lower()) or res.areas_by_key.get(
            _slugify(area_id)
        )
        if item is None:
            raise HTTPException(status_code=404, detail=f"Area {area_id!r} not found.")

        trend: list[dict[str, Any]] = []
        if not res.growth_df.empty:
            sub = (
                res.growth_df.loc[res.growth_df["area_name"] == item.area_name]
                .sort_values("quarter_date")
                .copy()
            )
            trend = [
                {
                    "year_quarter": str(r["year_quarter"]),
                    "quarter_date": str(r["quarter_date"]),
                    "median_ppsqm": float(r["median_ppsqm"]),
                    "tx_count": int(r["tx_count"]),
                    "yoy_growth_pct": float(r["yoy_growth_pct"])
                    if pd.notna(r["yoy_growth_pct"])
                    else None,
                    "cumulative_growth_pct": float(r["cumulative_growth_pct"])
                    if pd.notna(r["cumulative_growth_pct"])
                    else None,
                }
                for _, r in sub.iterrows()
            ]
        return AreaDetailResponse(
            **item.model_dump(),
            trend=trend,
            yield_caveat=res.yield_data.get("caveat", YIELD_CAVEAT),
        )

    @application.get("/yield", response_model=YieldRankingResponse)
    def get_yield(
        budget: float | None = Query(default=None, ge=50_000.0, le=500_000_000.0),
        rooms: int | None = Query(default=None, ge=0, le=10),
        type: str | None = Query(default=None, pattern="^(Unit|Villa)$"),
        top_k: int = Query(default=10, ge=1, le=50),
    ) -> YieldRankingResponse:
        res = _res()
        segs = res.yield_data.get("segment_ranking", [])
        if budget is not None:
            ranked = recommend_areas_for_budget(
                segs,
                budget_aed=budget,
                rooms=rooms,
                property_type=type,
                top_k=top_k,
            )
        elif rooms is not None or type is not None:
            ranked = recommend_areas_for_budget(
                segs,
                budget_aed=500_000_000.0,
                rooms=rooms,
                property_type=type,
                top_k=top_k,
            )
        else:
            ranked = list(res.yield_data.get("area_ranking", []))[:top_k]

        return YieldRankingResponse(
            budget_aed=budget,
            rooms=rooms,
            property_type=type,
            caveat=res.yield_data.get("caveat", YIELD_CAVEAT),
            results=ranked,
        )

    @application.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        res = _res()
        try:
            filters, used_llm, parser_note = parse_query_nl(req.query, known_areas=res.known_areas)
        except SearchGuardrailError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if req.limit is not None:
            filters = filters.model_copy(update={"limit": min(int(req.limit), 100)})

        out = execute_property_search(
            filters=filters,
            features_path=res.features_path,
            yield_json_path=res.yield_json_path,
        )
        return SearchResponse(
            filters=out["filters"],
            results=out["results"],
            explanation=out["explanation"],
            used_llm=used_llm,
            parser_note=parser_note,
            total_matches=out["total_matches"],
            summary=out["summary"],
        )

    return application


app = create_app()
