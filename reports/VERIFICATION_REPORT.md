# DubaiFairValue Verification Report

- **Timestamp (UTC)**: `2026-11-11T06:40:11Z`
- **Git SHA**: `eaf3cf0`
- **Data Manifest Hash**: `946ce591cc0c55b7`
- **Test Count**: `73`
- **Code Coverage (`src/dfv`)**: `85%`
- **Overall Status**: `PASS`

## Check Summary

| Check | Status | Details |
| --- | --- | --- |
| Lint & mypy | **PASS** | ruff check/format and mypy passed |
| Secrets & .env gitignore | **PASS** | Scanned 128 files cleanly |
| Pytest & coverage | **PASS** | 73 passed, coverage 85% |
| Data contracts (Pandera) | **PASS** | Contracts passed |
| Model acceptance gates | **PASS** | Model MdAPE=0.06891, Best baseline MdAPE=0.09484, 90% coverage=0.9012 |
| API contract & latency tests | **PASS** | API tests passed (warm /predict median=12.32 ms) |
| Streamlit app smoke tests | **PASS** | App smoke tests passed |
| Notebook execution | **PASS** | 5 notebooks executed |

## Measured Model & System Metrics

- Model MdAPE: `0.06891`
- Best Baseline MdAPE: `0.09484`
- Nominal 90% Interval Coverage: `0.9012`
- Spatial Hold-out MdAPE: `0.19908`
- API `/predict` Warm Latency (ms): `12.32`

## Skipped Checks & Justifications

- None (all checks active and evaluated).
