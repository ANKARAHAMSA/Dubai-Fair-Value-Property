"""Parameterized DuckDB query executor over cleaned/featured Parquet tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dfv.search.schema import PropertyFilters, summarize_filters


def execute_property_search(
    filters: PropertyFilters,
    features_path: Path,
    yield_json_path: Path | None = None,
) -> dict[str, Any]:
    """Execute validated PropertyFilters exclusively via parameterized DuckDB SQL."""
    area_yields: dict[str, float] = {}
    if yield_json_path is not None and yield_json_path.exists():
        yr_payload = json.loads(yield_json_path.read_text(encoding="utf-8"))
        for row in yr_payload.get("area_ranking", []):
            area_yields[str(row["area_name"])] = float(row["gross_yield_pct"])

    clauses: list[str] = [
        "COALESCE(f.flag_outlier, FALSE) = FALSE",
        "COALESCE(f.flag_non_market, FALSE) = FALSE",
        "COALESCE(f.flag_duplicate, FALSE) = FALSE",
    ]
    params: list[Any] = [str(features_path)]

    if filters.rooms is not None:
        clauses.append("f.rooms = ?")
        params.append(int(filters.rooms))
    elif filters.min_rooms is not None:
        clauses.append("f.rooms >= ?")
        params.append(int(filters.min_rooms))

    if filters.property_type is not None:
        clauses.append("f.property_type = ?")
        params.append(str(filters.property_type))

    if filters.off_plan is not None:
        clauses.append("f.is_offplan = ?")
        params.append(1 if filters.off_plan else 0)

    if filters.min_price_aed is not None:
        clauses.append("f.actual_worth >= ?")
        params.append(float(filters.min_price_aed))

    if filters.max_price_aed is not None:
        clauses.append("f.actual_worth <= ?")
        params.append(float(filters.max_price_aed))

    if filters.min_area_sqm is not None:
        clauses.append("f.area_sqm >= ?")
        params.append(float(filters.min_area_sqm))

    if filters.max_area_sqm is not None:
        clauses.append("f.area_sqm <= ?")
        params.append(float(filters.max_area_sqm))

    if filters.near_metro_m is not None:
        clauses.append("f.dist_nearest_metro_m <= ?")
        params.append(float(filters.near_metro_m))

    if filters.areas:
        placeholders = ", ".join(["?"] * len(filters.areas))
        clauses.append(f"f.area_name IN ({placeholders})")
        params.extend([str(a) for a in filters.areas])

    yield_df = pd.DataFrame(
        {
            "area_name": list(area_yields.keys()) if area_yields else ["__none__"],
            "est_area_yield_pct": list(area_yields.values()) if area_yields else [0.0],
        }
    )

    if filters.min_yield_pct is not None:
        clauses.append("COALESCE(y.est_area_yield_pct, f.area_trailing_12m_yield * 100.0) >= ?")
        params.append(float(filters.min_yield_pct))

    where_sql = " AND ".join(clauses)
    limit_val = min(int(filters.limit), 100)

    base_cte = f"""
        WITH base AS (
            SELECT
                f.transaction_id,
                f.instance_date,
                f.area_name,
                f.project_name,
                f.property_type,
                f.rooms,
                ROUND(f.area_sqm, 1) AS area_sqm,
                ROUND(f.actual_worth, 0) AS actual_worth,
                ROUND(f.price_per_sqm, 1) AS price_per_sqm,
                f.is_offplan,
                f.nearest_metro,
                ROUND(f.dist_nearest_metro_m, 0) AS dist_nearest_metro_m,
                ROUND(COALESCE(y.est_area_yield_pct, f.area_trailing_12m_yield * 100.0), 2) AS est_gross_yield_pct
            FROM read_parquet(?) AS f
            LEFT JOIN yield_tbl AS y ON f.area_name = y.area_name
            WHERE {where_sql}
        )
    """

    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("yield_tbl", yield_df)
        stats_row = conn.execute(
            f"""{base_cte}
            SELECT
                COUNT(*)::BIGINT AS total_matches,
                MEDIAN(actual_worth) AS median_price_aed,
                MEDIAN(price_per_sqm) AS median_ppsqm
            FROM base
            """,
            params,
        ).fetchone()

        total_matches = int(stats_row[0]) if stats_row else 0
        med_price = (
            round(float(stats_row[1]), 0)
            if stats_row and stats_row[1] is not None and np.isfinite(float(stats_row[1]))
            else None
        )
        med_ppsqm = (
            round(float(stats_row[2]), 1)
            if stats_row and stats_row[2] is not None and np.isfinite(float(stats_row[2]))
            else None
        )

        rows_df = conn.execute(
            f"""{base_cte}
            SELECT *
            FROM base
            ORDER BY instance_date DESC, actual_worth ASC
            LIMIT ?
            """,
            [*params, limit_val],
        ).df()
    finally:
        conn.close()

    results = [{str(k): v for k, v in rec.items()} for rec in rows_df.to_dict(orient="records")]
    return {
        "filters": filters.model_dump(),
        "explanation": summarize_filters(filters),
        "total_matches": total_matches,
        "returned_count": len(results),
        "summary": {
            "median_price_aed": med_price,
            "median_ppsqm": med_ppsqm,
        },
        "results": results,
    }
