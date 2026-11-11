# DubaiFairValue Verification Report

- **Timestamp (UTC)**: `2026-11-11T07:33:46Z`
- **Data Mode**: `processed`
- **Git SHA**: `65f4e17`
- **Data Manifest Hash**: `6580090e6cf82820`
- **Test Count**: `83`
- **Code Coverage (`src/dfv`)**: `89%`
- **Overall Status**: `PASS`

## Check Summary

| Check | Status | Details |
| --- | --- | --- |
| Lint & mypy | **PASS** | ruff check/format and mypy passed |
| Secrets & .env gitignore | **PASS** | Scanned 142 files cleanly |
| Pytest & coverage | **PASS** | 83 passed, coverage 89% |
| Data contracts (Pandera) | **PASS** | Contracts passed (3 checks) |
| Model acceptance gates | **PASS** | Model MdAPE=0.06936, Best baseline MdAPE=0.09501, 90% coverage=0.8993 |
| API contract & latency tests | **PASS** | API tests passed (warm /predict median=13.6 ms) |
| Streamlit app smoke tests | **PASS** | App smoke tests passed (8 tests) |
| Notebook execution | **PASS** | 5 notebooks executed |

## Measured Model & System Metrics

- Model MdAPE: `0.06936`
- Best Baseline MdAPE: `0.09501`
- Nominal 90% Interval Coverage: `0.8993`
- Spatial Hold-out MdAPE: `0.20216`
- API `/predict` Warm Latency (ms): `13.6`

## Skipped Checks & Justifications

- None (all checks active and evaluated).
