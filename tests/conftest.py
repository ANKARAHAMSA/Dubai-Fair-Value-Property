"""Shared pytest fixtures for DubaiFairValue."""

from __future__ import annotations

import pytest

from dfv.data.synth import SyntheticBundle, generate_synthetic_bundle


@pytest.fixture(scope="session")
def synthetic_bundle() -> SyntheticBundle:
    """Small deterministic synthetic bundle for fast tests."""
    return generate_synthetic_bundle(n_sales=1200, n_rents=1500, seed=42, include_dirty_rows=False)


@pytest.fixture(scope="session")
def dirty_synthetic_bundle() -> SyntheticBundle:
    """Synthetic bundle containing dirty/non-sales/duplicate rows for cleaning tests."""
    return generate_synthetic_bundle(n_sales=600, n_rents=600, seed=99, include_dirty_rows=True)
