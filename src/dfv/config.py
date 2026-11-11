"""Central configuration and path management for DubaiFairValue."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

SQFT_PER_SQM: float = 10.7639
USD_AED_PEG: float = 3.6725

# Sanity bounds for Dubai residential transactions (AED, sqm)
MIN_PRICE_PER_SQM_AED: float = 1500.0
MAX_PRICE_PER_SQM_AED: float = 250000.0
MIN_AREA_SQM: float = 15.0
MAX_AREA_SQM: float = 5000.0
MIN_TOTAL_PRICE_AED: float = 100000.0
MIN_ANNUAL_RENT_AED: float = 10000.0
MAX_ANNUAL_RENT_AED: float = 5000000.0

# Major area ordering sanity band (recent median price/sqm ordering)
AREA_ORDERING_CHECKS: tuple[tuple[str, str], ...] = (
    ("Downtown Dubai", "Dubai South"),
    ("Palm Jumeirah", "Discovery Gardens"),
)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout relative to the repository root."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    interim: Path = PROJECT_ROOT / "data" / "interim"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    sample: Path = PROJECT_ROOT / "data" / "sample"
    reference: Path = PROJECT_ROOT / "data" / "reference"
    cache: Path = PROJECT_ROOT / "data" / "cache"
    docs: Path = PROJECT_ROOT / "docs"
    reports: Path = PROJECT_ROOT / "reports"
    figures: Path = PROJECT_ROOT / "reports" / "figures"
    notebooks: Path = PROJECT_ROOT / "notebooks"
    models: Path = PROJECT_ROOT / "data" / "processed" / "models"
    sample_models: Path = PROJECT_ROOT / "data" / "sample" / "models"


PATHS = Paths()


def get_data_mode() -> str:
    """Return active data mode ('processed' or 'sample') based on env and files."""
    override = os.getenv("DFV_DATA_MODE", "auto").strip().lower()
    if override in {"sample", "processed"}:
        return override
    if (PATHS.processed / "transactions.parquet").exists():
        return "processed"
    return "sample"


def ensure_dirs() -> None:
    """Create expected writable project directories if missing."""
    for directory in (
        PATHS.raw,
        PATHS.interim,
        PATHS.processed,
        PATHS.sample,
        PATHS.reference,
        PATHS.cache,
        PATHS.docs,
        PATHS.reports,
        PATHS.figures,
        PATHS.models,
        PATHS.sample_models,
    ):
        directory.mkdir(parents=True, exist_ok=True)
