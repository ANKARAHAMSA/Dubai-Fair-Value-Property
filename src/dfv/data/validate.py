"""Pandera data contracts for cleaned and sample tables (`make data` step 3)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from dfv.config import PATHS, Paths

logger = logging.getLogger("dfv.data.validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TransactionsContract(pa.DataFrameModel):
    """Schema contract for cleaned DLD residential sales transactions."""

    transaction_id: Series[str] = pa.Field(nullable=False, unique=True)
    instance_date: Series[str] = pa.Field(nullable=False, str_matches=r"^\d{4}-\d{2}-\d{2}$")
    area_name: Series[str] = pa.Field(nullable=False, str_length={"min_value": 2})
    project_name: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    developer_name: Series[str] = pa.Field(nullable=False)
    property_type: Series[str] = pa.Field(isin=["Unit", "Villa"])
    rooms: Series[int] = pa.Field(ge=0, le=10)
    area_sqm: Series[float] = pa.Field(gt=0.0)
    actual_worth: Series[float] = pa.Field(gt=0.0)
    price_per_sqm: Series[float] = pa.Field(gt=0.0)
    is_offplan: Series[int] = pa.Field(isin=[0, 1])
    flag_outlier: Series[bool]
    flag_non_market: Series[bool]
    flag_duplicate: Series[bool]

    class Config:
        coerce = True
        strict = False


class RentsContract(pa.DataFrameModel):
    """Schema contract for cleaned Ejari residential rent contracts."""

    contract_id: Series[str] = pa.Field(nullable=False, unique=True)
    contract_start_date: Series[str] = pa.Field(nullable=False, str_matches=r"^\d{4}-\d{2}-\d{2}$")
    area_name: Series[str] = pa.Field(nullable=False, str_length={"min_value": 2})
    property_type: Series[str] = pa.Field(isin=["Unit", "Villa"])
    rooms: Series[int] = pa.Field(ge=0, le=10)
    area_sqm: Series[float] = pa.Field(gt=0.0)
    annual_amount: Series[float] = pa.Field(gt=0.0)
    rent_per_sqm: Series[float] = pa.Field(gt=0.0)
    is_renewal: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        coerce = True
        strict = False


class ProjectsContract(pa.DataFrameModel):
    """Schema contract for project metadata table."""

    project_id: Series[str] = pa.Field(nullable=False, unique=True)
    project_name: Series[str] = pa.Field(nullable=False)
    area_name: Series[str] = pa.Field(nullable=False)
    lat: Series[float] = pa.Field(ge=24.4, le=25.6)
    lon: Series[float] = pa.Field(ge=54.7, le=55.8)

    class Config:
        coerce = True
        strict = False


class MetroStationsContract(pa.DataFrameModel):
    """Schema contract for Dubai Metro station locations."""

    station_name: Series[str] = pa.Field(nullable=False)
    lat: Series[float] = pa.Field(ge=24.4, le=25.8)
    lon: Series[float] = pa.Field(ge=54.7, le=56.5)

    class Config:
        coerce = True
        strict = False


class SaleIndexContract(pa.DataFrameModel):
    """Schema contract for DLD sale-price index."""

    month: Series[str] = pa.Field(nullable=False)
    index_value: Series[float] = pa.Field(gt=0.0)

    class Config:
        coerce = True
        strict = False


def validate_tables(
    transactions: pd.DataFrame,
    rent_contracts: pd.DataFrame,
    projects: pd.DataFrame,
    metro_stations: pd.DataFrame,
    sale_index: pd.DataFrame,
) -> dict[str, int]:
    """Validate all core tables against Pandera contracts and return validated row counts."""
    tx_v = TransactionsContract.validate(transactions)
    rent_v = RentsContract.validate(rent_contracts)
    proj_v = ProjectsContract.validate(projects)
    metro_v = MetroStationsContract.validate(metro_stations)
    idx_v = SaleIndexContract.validate(sale_index)
    return {
        "transactions": int(len(tx_v)),
        "rent_contracts": int(len(rent_v)),
        "projects": int(len(proj_v)),
        "metro_stations": int(len(metro_v)),
        "sale_index": int(len(idx_v)),
    }


def validate_directory(data_dir: Path) -> dict[str, int]:
    """Validate Parquet files inside a directory (data/processed or data/sample)."""
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    rents = pd.read_parquet(data_dir / "rent_contracts.parquet")
    projs = pd.read_parquet(data_dir / "projects.parquet")
    metro = pd.read_parquet(data_dir / "metro_stations.parquet")
    idx = pd.read_parquet(data_dir / "sale_index.parquet")
    return validate_tables(tx, rents, projs, metro, idx)


def run_validation(paths: Paths = PATHS) -> dict[str, int]:
    """Validate processed directory if present, otherwise validate committed sample directory."""
    target_dir = (
        paths.processed if (paths.processed / "transactions.parquet").exists() else paths.sample
    )
    logger.info("Validating Pandera data contracts on %s", target_dir)
    counts = validate_directory(target_dir)
    logger.info("Contracts passed: %s", counts)
    return counts


def main() -> int:
    run_validation(PATHS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
