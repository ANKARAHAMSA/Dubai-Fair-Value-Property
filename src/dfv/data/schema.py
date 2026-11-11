"""Explicit schema adapter layer and data-dictionary generator.

All upstream DLD / RTA / fallback column names are mapped here to canonical
snake_case names so portal schema renames require a one-line fix.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# Canonical column adapter dictionaries (raw snake_case -> canonical name)
TRANSACTIONS_COLUMN_MAP: dict[str, str] = {
    # Official Dubai Pulse dld_transactions-open
    "transaction_id": "transaction_id",
    "trans_group_en": "trans_group",
    "trans_group": "trans_group",
    "procedure_name_en": "procedure_name",
    "procedure_name": "procedure_name",
    "instance_date": "instance_date",
    "date": "instance_date",
    "property_type_en": "property_type",
    "property_type": "property_type",
    "property_sub_type_en": "property_sub_type",
    "property_sub_type": "property_sub_type",
    "property_usage_en": "property_usage",
    "property_usage": "property_usage",
    "reg_type_en": "reg_type",
    "reg_type": "reg_type",
    "area_name_en": "area_name",
    "area_name": "area_name",
    "master_project_en": "master_project",
    "master_project": "master_project",
    "project_name_en": "project_name",
    "project_name": "project_name",
    "building_name_en": "building_name",
    "building_name": "building_name",
    "developer_name_en": "developer_name",
    "developer_name": "developer_name",
    "rooms_en": "rooms_label",
    "rooms_label": "rooms_label",
    "bedrooms": "rooms_label",
    "rooms": "rooms",
    "has_parking": "has_parking",
    "parkings": "has_parking",
    "procedure_area": "area_sqm",
    "area_sqm": "area_sqm",
    "built_up_area_sqft": "area_sqft",
    "area_sqft": "area_sqft",
    "actual_worth": "actual_worth",
    "selling_price": "actual_worth",
    "meter_sale_price": "price_per_sqm",
    "price_per_sqm": "price_per_sqm",
    "sale_price_per_foot": "price_per_sqft",
    "price_per_sqft": "price_per_sqft",
    "rent_value": "embedded_rent_value",
    "rent_price_per_foot": "embedded_rent_psf",
    "nearest_metro_en": "nearest_metro",
    "nearest_metro": "nearest_metro",
    "nearest_mall_en": "nearest_mall",
    "nearest_mall": "nearest_mall",
    "nearest_landmark_en": "nearest_landmark",
    "nearest_landmark": "nearest_landmark",
    "is_offplan": "is_offplan",
    "months_to_completion": "months_to_completion",
    "metro_distance_m": "metro_distance_m",
    "lat": "lat",
    "lon": "lon",
    "data_source": "data_source",
    # Arabic pairs preserved if present for name_map enrichment
    "area_name_ar": "area_name_ar",
    "project_name_ar": "project_name_ar",
}

RENTS_COLUMN_MAP: dict[str, str] = {
    "contract_id": "contract_id",
    "contract_start_date": "contract_start_date",
    "contract_end_date": "contract_end_date",
    "contract_reg_type_en": "contract_type",
    "contract_type": "contract_type",
    "is_renewal": "is_renewal",
    "ejari_property_type_en": "property_type",
    "property_type": "property_type",
    "ejari_property_sub_type_en": "property_sub_type",
    "property_sub_type": "property_sub_type",
    "property_usage_en": "property_usage",
    "property_usage": "property_usage",
    "area_name_en": "area_name",
    "area_name": "area_name",
    "project_name_en": "project_name",
    "project_name": "project_name",
    "rooms": "rooms",
    "rooms_en": "rooms_label",
    "actual_area": "area_sqm",
    "area_sqm": "area_sqm",
    "annual_amount": "annual_amount",
    "rent_per_sqm": "rent_per_sqm",
    "nearest_metro_en": "nearest_metro",
    "nearest_metro": "nearest_metro",
    "metro_distance_m": "metro_distance_m",
    "data_source": "data_source",
}

PROJECTS_COLUMN_MAP: dict[str, str] = {
    "project_id": "project_id",
    "project_number": "project_id",
    "project_name_en": "project_name",
    "project_name": "project_name",
    "developer_name_en": "developer_name",
    "developer_name": "developer_name",
    "area_name_en": "area_name",
    "area_name": "area_name",
    "project_status": "status",
    "status": "status",
    "percent_completed": "completion_pct",
    "completion_pct": "completion_pct",
    "project_start_date": "launch_date",
    "launch_date": "launch_date",
    "completion_date": "completion_date",
    "lat": "lat",
    "lon": "lon",
    "nearest_metro": "nearest_metro",
    "metro_distance_m": "metro_distance_m",
}

STATIONS_COLUMN_MAP: dict[str, str] = {
    "station_name": "station_name",
    "stop_name": "station_name",
    "name_en": "station_name",
    "line": "line",
    "line_name": "line",
    "lat": "lat",
    "stop_lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "stop_lon": "lon",
    "longitude": "lon",
    "opening_date": "opening_date",
    "is_route_2020": "is_route_2020",
}

SALE_INDEX_COLUMN_MAP: dict[str, str] = {
    "month": "month",
    "year_month": "month",
    "index_value": "index_value",
    "all_residential_index": "index_value",
    "property_type": "property_type",
    "data_source": "data_source",
}


def to_snake_case(name: str) -> str:
    """Normalize arbitrary portal header into clean snake_case."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def adapt_dataframe(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Normalize headers to snake_case and apply canonical column mapping safely."""
    out = df.copy()
    normalized = [to_snake_case(str(c)) for c in out.columns]
    out.columns = normalized

    rename_dict: dict[str, str] = {}
    seen_targets: set[str] = set()
    # Keep existing canonical columns prioritized if both alias and canonical exist
    for col in normalized:
        target = column_map.get(col, col)
        if target in seen_targets:
            continue
        rename_dict[col] = target
        seen_targets.add(target)

    # If multiple source columns map to same canonical target, coalesce nulls before dropping duplicates
    grouped_cols: dict[str, list[str]] = {}
    for col in normalized:
        target = column_map.get(col, col)
        grouped_cols.setdefault(target, []).append(col)

    result = pd.DataFrame(index=out.index)
    for target, srcs in grouped_cols.items():
        series = out[srcs[0]]
        for extra in srcs[1:]:
            series = series.combine_first(out[extra])
        result[target] = series

    return result


def summarize_table_schema(df: pd.DataFrame, table_name: str, source_label: str) -> list[str]:
    """Generate markdown schema lines from a real or synthetic DataFrame."""
    lines: list[str] = [
        f"## Table: `{table_name}`",
        "",
        f"- **Source**: `{source_label}`",
        f"- **Rows**: `{len(df):,}` | **Columns**: `{len(df.columns)}`",
        "",
        "| Column | Dtype | Null % | Min | Max | Example Values |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for col in df.columns:
        s = df[col]
        null_pct = float(s.isna().mean() * 100.0)
        dtype_str = str(s.dtype)
        non_null = s.dropna()
        if len(non_null) == 0:
            min_v, max_v, ex_v = "n/a", "n/a", "n/a"
        else:
            if pd.api.types.is_numeric_dtype(s):
                min_v = f"{non_null.min():.4g}"
                max_v = f"{non_null.max():.4g}"
            elif pd.api.types.is_datetime64_any_dtype(s):
                min_v = str(non_null.min())[:10]
                max_v = str(non_null.max())[:10]
            else:
                # Sample min/max lexicographically safely
                str_vals = non_null.astype(str)
                min_v = str_vals.min()[:24]
                max_v = str_vals.max()[:24]
            samples = [str(v)[:24].replace("|", "/") for v in non_null.head(3).tolist()]
            ex_v = ", ".join(samples)

        lines.append(
            f"| `{col}` | `{dtype_str}` | {null_pct:.1f}% | `{min_v}` | `{max_v}` | {ex_v} |"
        )
    lines.append("")
    return lines


def write_data_dictionary(
    tables: dict[str, tuple[pd.DataFrame, str]],
    output_path: Path,
) -> Path:
    """Write docs/data_dictionary.md from inspected DataFrames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = [
        "# DubaiFairValue Data Dictionary",
        "",
        "Auto-generated from actual ingested tables (`src/dfv/data/schema.py`).",
        "All upstream column names are normalized to `snake_case` via the adapter layer.",
        "",
    ]
    body: list[str] = []
    for name, (df, src) in tables.items():
        body.extend(summarize_table_schema(df, name, src))

    output_path.write_text("\n".join(header + body), encoding="utf-8")
    return output_path


def canonical_summary_dict(df: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight stats dict for programmatic checks."""
    return {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "null_rates": {str(c): round(float(df[c].isna().mean()), 4) for c in df.columns},
    }
