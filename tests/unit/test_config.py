"""Unit tests for configuration and path utilities."""

from __future__ import annotations

from dfv.config import PATHS, SQFT_PER_SQM, ensure_dirs, get_data_mode


def test_paths_exist_and_constants() -> None:
    assert (PATHS.root / "pyproject.toml").exists()
    assert (PATHS.root / "src" / "dfv").is_dir()
    assert abs(SQFT_PER_SQM - 10.7639) < 1e-6
    ensure_dirs()
    assert PATHS.sample.exists()
    assert PATHS.reports.exists()


def test_get_data_mode_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DFV_DATA_MODE", "sample")
    assert get_data_mode() == "sample"
    monkeypatch.setenv("DFV_DATA_MODE", "processed")
    assert get_data_mode() == "processed"
    monkeypatch.setenv("DFV_DATA_MODE", "auto")
    mode = get_data_mode()
    assert mode in {"sample", "processed"}
