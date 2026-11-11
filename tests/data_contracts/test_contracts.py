"""Pandera data contract tests (run on processed data if present, else committed sample)."""

from __future__ import annotations

import pandera.errors as pa_errors
import pytest

from dfv.config import PATHS
from dfv.data.synth import SyntheticBundle
from dfv.data.validate import TransactionsContract, run_validation, validate_tables


def test_data_contracts_on_active_dataset() -> None:
    counts = run_validation(PATHS)
    assert counts["transactions"] > 0
    assert counts["rent_contracts"] > 0
    assert counts["projects"] > 0
    assert counts["metro_stations"] > 0
    assert counts["sale_index"] > 0


def test_data_contracts_on_synthetic_bundle(synthetic_bundle: SyntheticBundle) -> None:
    counts = validate_tables(
        synthetic_bundle.transactions,
        synthetic_bundle.rent_contracts,
        synthetic_bundle.projects,
        synthetic_bundle.metro_stations,
        synthetic_bundle.sale_index,
    )
    assert counts["transactions"] == len(synthetic_bundle.transactions)


def test_contract_rejects_negative_area(synthetic_bundle: SyntheticBundle) -> None:
    bad = synthetic_bundle.transactions.head(5).copy()
    bad.loc[bad.index[0], "area_sqm"] = -10.0
    with pytest.raises(pa_errors.SchemaError):
        TransactionsContract.validate(bad)
