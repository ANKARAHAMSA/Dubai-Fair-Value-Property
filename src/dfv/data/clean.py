"""Data cleaning and standardization pipeline (`make data` step 2).

Reads normalized interim Parquet files from data/interim/, applies documented
filtering, unit conversion, name normalization, non-market/duplicate/outlier
flagging, Dubai realism checks, and outputs Parquet tables to data/processed/
along with reports/data_cleaning_report.md.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from dfv.config import (
    MAX_ANNUAL_RENT_AED,
    MAX_AREA_SQM,
    MAX_PRICE_PER_SQM_AED,
    MIN_ANNUAL_RENT_AED,
    MIN_AREA_SQM,
    MIN_PRICE_PER_SQM_AED,
    MIN_TOTAL_PRICE_AED,
    PATHS,
    SQFT_PER_SQM,
    Paths,
)

logger = logging.getLogger("dfv.data.clean")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALLOWED_SALES_PROCEDURES: set[str] = {
    "sell",
    "sales",
    "sale",
    "sell - pre registration",
    "delayed sell",
    "sell development",
    "sell - development registration",
}

EXCLUDED_SUBTYPES: set[str] = {
    "office",
    "shop",
    "hotel apartment",
    "hotel rooms",
    "hotel",
    "workshop",
    "store",
    "warehouse",
    "clinic",
    "sized partition",
    "show rooms",
    "building",
}


@dataclass
class CleaningRuleLog:
    """Record of rows remaining and removed/flagged by a single cleaning rule."""

    rule: str
    rows_before: int
    rows_after: int
    rows_affected: int
    action: str  # dropped | flagged | checked
    note: str = ""


def load_name_map(name_map_path: Path) -> dict[str, dict[str, str]]:
    """Load reviewed mapping CSV grouped by entity_type -> {raw_lower: canonical}."""
    mapping: dict[str, dict[str, str]] = {"area": {}, "developer": {}, "metro": {}}
    if not name_map_path.exists():
        return mapping
    df = pd.read_csv(name_map_path)
    for _, row in df.iterrows():
        etype = str(row["entity_type"]).strip().lower()
        raw = re.sub(r"\s+", " ", str(row["raw_name"]).strip().lower())
        canon = str(row["canonical_name"]).strip()
        mapping.setdefault(etype, {})[raw] = canon
    return mapping


def normalize_text_series(series: pd.Series) -> pd.Series:
    """Strip whitespace and collapse internal spacing without losing missingness."""
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s


def map_names_with_lookup(
    series: pd.Series,
    lookup: dict[str, str],
    fuzzy_threshold: int = 92,
) -> pd.Series:
    """Map raw strings via exact casefolded lookup then rapidfuzz on remaining unique values."""
    cleaned = normalize_text_series(series)
    unique_vals = cleaned.dropna().unique().tolist()
    canon_targets = sorted(set(lookup.values()))

    resolved: dict[str, str] = {}
    for val in unique_vals:
        key = val.casefold()
        if key in lookup:
            resolved[val] = lookup[key]
        elif canon_targets:
            match = process.extractOne(
                key, [c.casefold() for c in canon_targets], scorer=fuzz.WRatio
            )
            if match is not None and match[1] >= fuzzy_threshold:
                matched_lower = match[0]
                canon_match = next(c for c in canon_targets if c.casefold() == matched_lower)
                resolved[val] = canon_match
            else:
                resolved[val] = val
        else:
            resolved[val] = val

    return cleaned.map(resolved)


def parse_rooms_from_label(series: pd.Series) -> pd.Series:
    """Parse DLD bedroom strings ('Studio', '1 B/R', 'PENTHOUSE', etc.) into numeric counts."""
    s = series.astype("string").str.strip().str.upper()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out = out.mask(s.isin(["STUDIO", "0 B/R", "0"]), 0.0)
    out = out.mask(s.isin(["SINGLE ROOM"]), 1.0)
    out = out.mask(s.isin(["PENTHOUSE"]), 4.0)

    extracted = s.str.extract(r"^(\d+)", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce").astype("float64")
    out = out.fillna(numeric)
    return out


def filter_residential_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Retain residential unit/villa sales transactions and exclude land/commercial/mortgages."""
    out = df.copy()
    proc = out["procedure_name"].astype("string").str.strip().str.lower()
    is_sale_proc = proc.isin(ALLOWED_SALES_PROCEDURES)
    if "trans_group" in out.columns:
        tg = out["trans_group"].astype("string").str.strip().str.lower()
        is_sale_proc = is_sale_proc & (tg.isna() | tg.isin({"sales", "sell"}))

    ptype = out["property_type"].astype("string").str.strip().str.lower()
    is_res_type = ptype.isin({"unit", "villa"})

    usage = out["property_usage"].astype("string").str.strip().str.lower()
    is_res_usage = usage.eq("residential")

    if "property_sub_type" in out.columns:
        sub = out["property_sub_type"].astype("string").str.strip().str.lower()
        is_allowed_sub = sub.isna() | ~sub.isin(EXCLUDED_SUBTYPES)
    else:
        is_allowed_sub = pd.Series(True, index=out.index)

    mask = is_sale_proc & is_res_type & is_res_usage & is_allowed_sub
    return out.loc[mask].copy()


def clean_transactions(
    df: pd.DataFrame,
    name_map: dict[str, dict[str, str]] | None = None,
    min_date: str = "2015-01-01",
    max_date: str | None = None,
) -> tuple[pd.DataFrame, list[CleaningRuleLog]]:
    """Apply all DLD residential sales cleaning rules and return (cleaned_df, rule_logs)."""
    logs: list[CleaningRuleLog] = []
    work = df.copy()
    n0 = len(work)

    # Rule 1: Keep residential sales only
    work = filter_residential_sales(work)
    logs.append(
        CleaningRuleLog(
            "Filter residential sales (exclude mortgages/gifts/land/commercial)",
            n0,
            len(work),
            n0 - len(work),
            "dropped",
        )
    )

    # Rule 2: Parse dates and enforce plausible date window
    n_before = len(work)
    parsed_dt = pd.to_datetime(
        work["instance_date"], format="mixed", dayfirst=True, errors="coerce"
    )
    min_ts = pd.Timestamp(min_date)
    max_ts = (
        pd.Timestamp(max_date)
        if max_date
        else (pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(days=30))
    )
    valid_dt = parsed_dt.notna() & (parsed_dt >= min_ts) & (parsed_dt <= max_ts)
    work["instance_date"] = parsed_dt
    work = work.loc[valid_dt].copy()
    logs.append(
        CleaningRuleLog(
            f"Parse dates and enforce range [{min_date}, {max_ts.strftime('%Y-%m-%d')}]",
            n_before,
            len(work),
            n_before - len(work),
            "dropped",
        )
    )

    # Rule 3: Remove missing or non-positive price/area and derive sqm/sqft metrics
    n_before = len(work)
    work["actual_worth"] = (
        pd.to_numeric(work["actual_worth"], errors="coerce")
        if "actual_worth" in work.columns
        else pd.Series(np.nan, index=work.index, dtype="float64")
    )
    area_sqm = (
        pd.to_numeric(work["area_sqm"], errors="coerce")
        if "area_sqm" in work.columns
        else pd.Series(np.nan, index=work.index, dtype="float64")
    )
    if "area_sqft" in work.columns:
        area_sqft = pd.to_numeric(work["area_sqft"], errors="coerce")
        area_sqm = area_sqm.fillna(area_sqft / SQFT_PER_SQM)
    else:
        area_sqft = area_sqm * SQFT_PER_SQM

    work["area_sqm"] = area_sqm
    work["area_sqft"] = area_sqft.fillna(area_sqm * SQFT_PER_SQM)

    pos_mask = (
        work["actual_worth"].notna()
        & (work["actual_worth"] > 0.0)
        & work["area_sqm"].notna()
        & (work["area_sqm"] > 0.0)
    )
    work = work.loc[pos_mask].copy()
    work["price_per_sqm"] = work["actual_worth"] / work["area_sqm"]
    work["price_per_sqft"] = work["price_per_sqm"] / SQFT_PER_SQM
    logs.append(
        CleaningRuleLog(
            "Remove non-positive/missing price or area; compute price_per_sqm & price_per_sqft",
            n_before,
            len(work),
            n_before - len(work),
            "dropped",
        )
    )

    # Rule 4: Standardize areas, projects, developers, metro names
    nmap = name_map or {"area": {}, "developer": {}, "metro": {}}
    work["area_name"] = map_names_with_lookup(work["area_name"], nmap.get("area", {}))

    # Disambiguate master_project overrides (e.g. Discovery Gardens inside Jabal Ali First)
    if "master_project" in work.columns:
        mp_norm = normalize_text_series(work["master_project"])
        work["master_project"] = mp_norm.fillna(work["area_name"])
        mp_low = work["master_project"].astype("string").str.lower()
        work["area_name"] = np.where(
            mp_low.str.contains("discovery gardens", na=False),
            "Discovery Gardens",
            np.where(
                mp_low.str.contains("arabian ranches", na=False),
                "Arabian Ranches",
                work["area_name"],
            ),
        )
    else:
        work["master_project"] = work["area_name"]

    if "project_name" in work.columns and work["project_name"].notna().any():
        proj_series = normalize_text_series(work["project_name"])
    elif "building_name" in work.columns:
        proj_series = normalize_text_series(work["building_name"])
    else:
        proj_series = pd.Series(pd.NA, index=work.index, dtype="string")

    if "building_name" in work.columns:
        proj_series = proj_series.fillna(normalize_text_series(work["building_name"]))
    proj_series = proj_series.fillna(work["master_project"]).fillna(work["area_name"] + " Project")
    work["project_name"] = proj_series

    if "developer_name" in work.columns and work["developer_name"].notna().any():
        work["developer_name"] = map_names_with_lookup(
            work["developer_name"].fillna("Unknown"), nmap.get("developer", {})
        )
    else:
        work["developer_name"] = "Unknown"

    if "nearest_metro" in work.columns:
        work["nearest_metro"] = map_names_with_lookup(
            work["nearest_metro"], nmap.get("metro", {}), fuzzy_threshold=90
        )
    else:
        work["nearest_metro"] = pd.NA

    for opt_col in ("nearest_mall", "nearest_landmark"):
        if opt_col in work.columns:
            work[opt_col] = normalize_text_series(work[opt_col])
        else:
            work[opt_col] = pd.NA

    logs.append(
        CleaningRuleLog(
            "Standardize area, project, developer, and metro names via name_map + rapidfuzz",
            len(work),
            len(work),
            int(work["area_name"].nunique()),
            "checked",
            note=f"{work['area_name'].nunique()} distinct canonical areas",
        )
    )

    # Rule 5: Rooms parsing and Villa missing-bedroom imputation
    if "rooms" in work.columns and pd.api.types.is_numeric_dtype(work["rooms"]):
        rooms_num = pd.to_numeric(work["rooms"], errors="coerce")
    else:
        rooms_num = pd.Series(np.nan, index=work.index, dtype="float64")
    if "rooms_label" in work.columns:
        rooms_num = rooms_num.fillna(parse_rooms_from_label(work["rooms_label"]).astype("float64"))

    work["rooms_was_imputed"] = rooms_num.isna()
    # Impute missing bedrooms cleanly from area brackets (common for DLD Villa rows)
    imputed_rooms = np.select(
        [
            work["area_sqm"] < 55.0,
            work["area_sqm"] < 95.0,
            work["area_sqm"] < 150.0,
            work["area_sqm"] < 240.0,
            work["area_sqm"] < 380.0,
        ],
        [0, 1, 2, 3, 4],
        default=5,
    )
    rooms_filled = (
        rooms_num.fillna(pd.Series(imputed_rooms, index=work.index, dtype="float64"))
        .clip(0, 10)
        .astype(int)
    )
    work["rooms"] = rooms_filled
    fallback_rooms_label = pd.Series(
        np.where(work["rooms"] == 0, "Studio", work["rooms"].astype(str) + " B/R"),
        index=work.index,
        dtype="string",
    )
    if "rooms_label" not in work.columns:
        work["rooms_label"] = fallback_rooms_label
    else:
        work["rooms_label"] = normalize_text_series(work["rooms_label"]).fillna(
            fallback_rooms_label
        )

    # Rule 6: Off-plan flag & auxiliary fields
    if "is_offplan" in work.columns and work["is_offplan"].notna().all():
        work["is_offplan"] = (
            pd.to_numeric(work["is_offplan"], errors="coerce").fillna(0).astype(int)
        )
    else:
        proc_low = work["procedure_name"].astype("string").str.lower()
        reg_low = (
            work["reg_type"].astype("string").str.lower()
            if "reg_type" in work.columns
            else pd.Series("", index=work.index, dtype="string")
        )
        work["is_offplan"] = (
            proc_low.str.contains("pre registration|pre-registration|off-plan", na=False)
            | reg_low.str.contains("off-plan|off plan", na=False)
        ).astype(int)

    if "has_parking" in work.columns:
        work["has_parking"] = (
            pd.to_numeric(work["has_parking"], errors="coerce").fillna(0).gt(0).astype(int)
        )
    else:
        work["has_parking"] = 0

    if "property_sub_type" not in work.columns:
        work["property_sub_type"] = work["property_type"]
    else:
        work["property_sub_type"] = normalize_text_series(work["property_sub_type"]).fillna(
            work["property_type"]
        )

    if "transaction_id" not in work.columns or work["transaction_id"].isna().any():
        work["transaction_id"] = [f"DLD-TX-{i + 1:07d}" for i in range(len(work))]

    # Rule 7: Flag duplicates (do not silently drop; keep flag_duplicate column)
    dup_subset = [
        "instance_date",
        "area_name",
        "project_name",
        "property_type",
        "rooms",
        "area_sqm",
        "actual_worth",
    ]
    work["flag_duplicate"] = work.duplicated(subset=dup_subset, keep="first")
    logs.append(
        CleaningRuleLog(
            "Flag exact duplicate transactions",
            len(work),
            len(work),
            int(work["flag_duplicate"].sum()),
            "flagged",
        )
    )

    # Rule 8: Flag non-market / sanity-bound violations
    non_market = (
        (work["actual_worth"] < MIN_TOTAL_PRICE_AED)
        | (work["area_sqm"] < MIN_AREA_SQM)
        | (work["area_sqm"] > MAX_AREA_SQM)
        | (work["price_per_sqm"] < MIN_PRICE_PER_SQM_AED)
        | (work["price_per_sqm"] > MAX_PRICE_PER_SQM_AED)
    )
    work["flag_non_market"] = non_market
    logs.append(
        CleaningRuleLog(
            "Flag non-market & hard sanity-bound violations",
            len(work),
            len(work),
            int(non_market.sum()),
            "flagged",
        )
    )

    # Rule 9: Flag statistical outliers via MAD on log(price_per_sqm) within area x property_type x year
    work["year"] = work["instance_date"].dt.year.astype(int)
    work["log_ppsqm"] = np.log(work["price_per_sqm"].clip(lower=1.0))

    valid_for_stats = ~work["flag_non_market"] & ~work["flag_duplicate"]
    grp_cols = ["area_name", "property_type", "year"]
    grp_sizes = work.loc[valid_for_stats].groupby(grp_cols)["log_ppsqm"].transform("count")
    med_fine = work.loc[valid_for_stats].groupby(grp_cols)["log_ppsqm"].transform("median")
    abs_dev_fine = (work.loc[valid_for_stats, "log_ppsqm"] - med_fine).abs()
    mad_fine = abs_dev_fine.groupby([work.loc[valid_for_stats, c] for c in grp_cols]).transform(
        "median"
    )

    # Fallback to property_type x year if group has < 20 rows or MAD == 0
    fb_cols = ["property_type", "year"]
    med_fb = work.loc[valid_for_stats].groupby(fb_cols)["log_ppsqm"].transform("median")
    abs_dev_fb = (work.loc[valid_for_stats, "log_ppsqm"] - med_fb).abs()
    mad_fb = abs_dev_fb.groupby([work.loc[valid_for_stats, c] for c in fb_cols]).transform("median")

    use_fine = (grp_sizes >= 20) & (mad_fine > 1e-6)
    med = np.where(use_fine, med_fine, med_fb)
    mad = np.where(use_fine, mad_fine, np.maximum(mad_fb, 0.05))
    z_mad = 0.6745 * (work.loc[valid_for_stats, "log_ppsqm"] - med).abs() / mad

    outlier_series = pd.Series(False, index=work.index)
    outlier_series.loc[valid_for_stats] = z_mad > 3.5
    outlier_series = outlier_series | work["flag_non_market"]
    work["flag_outlier"] = outlier_series

    logs.append(
        CleaningRuleLog(
            "Flag statistical outliers (|MAD z-score| > 3.5 within area x type x year)",
            len(work),
            len(work),
            int(work["flag_outlier"].sum()),
            "flagged",
        )
    )

    work = work.drop(columns=["log_ppsqm", "year"])
    work["property_type"] = work["property_type"].astype("string").str.strip()
    work["property_usage"] = "Residential"
    work["instance_date"] = work["instance_date"].dt.strftime("%Y-%m-%d")
    if "data_source" not in work.columns:
        work["data_source"] = "unknown"

    # Sort chronologically for clean downstream processing
    work = work.sort_values("instance_date").reset_index(drop=True)
    return work, logs


def clean_rent_contracts(
    df: pd.DataFrame,
    name_map: dict[str, dict[str, str]] | None = None,
    min_date: str = "2015-01-01",
) -> tuple[pd.DataFrame, list[CleaningRuleLog]]:
    """Clean Ejari residential rent contracts."""
    logs: list[CleaningRuleLog] = []
    work = df.copy()
    n0 = len(work)

    if "property_usage" in work.columns:
        usage = work["property_usage"].astype("string").str.strip().str.lower()
        work = work.loc[usage.isna() | usage.eq("residential")].copy()
    logs.append(
        CleaningRuleLog(
            "Filter residential rent contracts", n0, len(work), n0 - len(work), "dropped"
        )
    )

    n_before = len(work)
    start_dt = pd.to_datetime(
        work["contract_start_date"], format="mixed", dayfirst=True, errors="coerce"
    )
    end_dt = pd.to_datetime(
        work["contract_end_date"], format="mixed", dayfirst=True, errors="coerce"
    )
    valid_dt = start_dt.notna() & (start_dt >= pd.Timestamp(min_date))
    work["contract_start_date"] = start_dt
    work["contract_end_date"] = end_dt.fillna(start_dt + pd.Timedelta(days=365))
    work = work.loc[valid_dt].copy()
    logs.append(
        CleaningRuleLog(
            f"Parse rent dates >= {min_date}", n_before, len(work), n_before - len(work), "dropped"
        )
    )

    n_before = len(work)
    work["annual_amount"] = pd.to_numeric(work["annual_amount"], errors="coerce")
    work["area_sqm"] = pd.to_numeric(work["area_sqm"], errors="coerce")
    valid_num = work["annual_amount"].between(MIN_ANNUAL_RENT_AED, MAX_ANNUAL_RENT_AED) & work[
        "area_sqm"
    ].between(MIN_AREA_SQM, MAX_AREA_SQM)
    work = work.loc[valid_num].copy()
    work["rent_per_sqm"] = work["annual_amount"] / work["area_sqm"]
    logs.append(
        CleaningRuleLog(
            "Enforce positive rent & area bounds; compute rent_per_sqm",
            n_before,
            len(work),
            n_before - len(work),
            "dropped",
        )
    )

    nmap = name_map or {"area": {}, "developer": {}, "metro": {}}
    work["area_name"] = map_names_with_lookup(work["area_name"], nmap.get("area", {}))
    work["property_type"] = work["property_type"].astype("string").str.strip()
    if "is_renewal" not in work.columns:
        ctype = (
            work.get("contract_type", pd.Series("New", index=work.index))
            .astype("string")
            .str.lower()
        )
        work["is_renewal"] = ctype.str.contains("renew", na=False).astype(int)
    else:
        work["is_renewal"] = (
            pd.to_numeric(work["is_renewal"], errors="coerce").fillna(0).astype(int)
        )

    if "rooms" not in work.columns and "rooms_label" in work.columns:
        work["rooms"] = parse_rooms_from_label(work["rooms_label"]).fillna(1).astype(int)
    else:
        work["rooms"] = pd.to_numeric(work["rooms"], errors="coerce").fillna(1).astype(int)

    if "contract_id" not in work.columns:
        work["contract_id"] = [f"EJ-{i + 1:07d}" for i in range(len(work))]

    work["contract_start_date"] = work["contract_start_date"].dt.strftime("%Y-%m-%d")
    work["contract_end_date"] = work["contract_end_date"].dt.strftime("%Y-%m-%d")
    return work.sort_values("contract_start_date").reset_index(drop=True), logs


def run_realism_checks(tx_clean: pd.DataFrame) -> dict[str, Any]:
    """Run Dubai-specific economic sanity checks on unflagged cleaned sales."""
    unflagged = tx_clean.loc[
        ~tx_clean["flag_outlier"] & ~tx_clean["flag_non_market"] & ~tx_clean["flag_duplicate"]
    ].copy()
    if unflagged.empty:
        raise ValueError("No unflagged transactions remain after cleaning.")
    if (unflagged["area_sqm"] <= 0).any() or (unflagged["actual_worth"] <= 0).any():
        raise ValueError("Found non-positive area or price in unflagged transactions.")

    dt = pd.to_datetime(unflagged["instance_date"])
    recent_cutoff = dt.max() - pd.Timedelta(days=730)
    recent = unflagged.loc[dt >= recent_cutoff]
    med_by_area = recent.groupby("area_name")["price_per_sqm"].median()

    checks: dict[str, Any] = {}
    if "Downtown Dubai" in med_by_area.index and "Dubai South" in med_by_area.index:
        dt_med = float(med_by_area["Downtown Dubai"])
        ds_med = float(med_by_area["Dubai South"])
        if not (dt_med > ds_med):
            raise ValueError(
                f"Realism check failed: Downtown Dubai ({dt_med:.0f}) <= Dubai South ({ds_med:.0f})"
            )
        checks["downtown_vs_dubai_south"] = {
            "downtown": round(dt_med, 1),
            "dubai_south": round(ds_med, 1),
        }

    if "Palm Jumeirah" in med_by_area.index and "Discovery Gardens" in med_by_area.index:
        pj_med = float(med_by_area["Palm Jumeirah"])
        dg_med = float(med_by_area["Discovery Gardens"])
        if not (pj_med > dg_med):
            raise ValueError(
                f"Realism check failed: Palm Jumeirah ({pj_med:.0f}) <= Discovery Gardens ({dg_med:.0f})"
            )
        checks["palm_vs_discovery_gardens"] = {
            "palm": round(pj_med, 1),
            "discovery_gardens": round(dg_med, 1),
        }

    return checks


def write_cleaning_report(
    tx_logs: list[CleaningRuleLog],
    rent_logs: list[CleaningRuleLog],
    realism: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write reports/data_cleaning_report.md summarizing every cleaning rule's impact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# DubaiFairValue Data Cleaning Report",
        "",
        "## Sales Transactions Cleaning Rules",
        "",
        "| Rule | Rows Before | Rows After | Rows Affected | Action | Notes |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in tx_logs:
        lines.append(
            f"| {r.rule} | {r.rows_before:,} | {r.rows_after:,} | {r.rows_affected:,} | `{r.action}` | {r.note} |"
        )
    lines.extend(
        [
            "",
            "## Rent Contracts Cleaning Rules",
            "",
            "| Rule | Rows Before | Rows After | Rows Affected | Action | Notes |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for r in rent_logs:
        lines.append(
            f"| {r.rule} | {r.rows_before:,} | {r.rows_after:,} | {r.rows_affected:,} | `{r.action}` | {r.note} |"
        )
    lines.extend(["", "## Dubai-Specific Realism Checks", ""])
    for k, v in realism.items():
        lines.append(f"- **{k}**: `{v}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_cleaning_pipeline(paths: Paths = PATHS) -> dict[str, Any]:
    """Read data/interim/, clean all tables, run realism checks, and write data/processed/."""
    paths.processed.mkdir(parents=True, exist_ok=True)
    name_map = load_name_map(paths.reference / "name_map.csv")

    tx_raw = pd.read_parquet(paths.interim / "transactions_raw.parquet")
    rent_raw = pd.read_parquet(paths.interim / "rent_contracts_raw.parquet")
    proj_raw = pd.read_parquet(paths.interim / "projects_raw.parquet")
    metro_raw = pd.read_parquet(paths.interim / "metro_stations_raw.parquet")
    idx_raw = pd.read_parquet(paths.interim / "sale_index_raw.parquet")

    tx_clean, tx_logs = clean_transactions(tx_raw, name_map=name_map)
    rent_clean, rent_logs = clean_rent_contracts(rent_raw, name_map=name_map)
    realism = run_realism_checks(tx_clean)

    # Also standardize area_name in projects
    if "area_name" in proj_raw.columns:
        proj_raw["area_name"] = map_names_with_lookup(
            proj_raw["area_name"], name_map.get("area", {})
        )

    tx_clean.to_parquet(paths.processed / "transactions.parquet", index=False)
    rent_clean.to_parquet(paths.processed / "rent_contracts.parquet", index=False)
    proj_raw.to_parquet(paths.processed / "projects.parquet", index=False)
    metro_raw.to_parquet(paths.processed / "metro_stations.parquet", index=False)
    idx_raw.to_parquet(paths.processed / "sale_index.parquet", index=False)

    write_cleaning_report(tx_logs, rent_logs, realism, paths.reports / "data_cleaning_report.md")
    logger.info(
        "Cleaned %s transactions (%s unflagged) and %s rent contracts.",
        f"{len(tx_clean):,}",
        f"{int((~tx_clean['flag_outlier'] & ~tx_clean['flag_non_market'] & ~tx_clean['flag_duplicate']).sum()):,}",
        f"{len(rent_clean):,}",
    )
    return {"tx_rows": len(tx_clean), "rent_rows": len(rent_clean), "realism": realism}


def main() -> int:
    run_cleaning_pipeline(paths=PATHS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
