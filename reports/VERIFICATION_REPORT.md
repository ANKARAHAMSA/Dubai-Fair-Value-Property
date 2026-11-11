# DubaiFairValue Verification Report

- **Timestamp (UTC)**: `2026-11-11T06:33:58Z`
- **Git SHA**: `b156f36`
- **Data Manifest Hash**: `946ce591cc0c55b7`
- **Test Count**: `65`
- **Code Coverage (`src/dfv`)**: `84%`
- **Overall Status**: `PASS`

## Check Summary

| Check | Status | Details |
| --- | --- | --- |
| Lint & mypy | **PASS** | ruff check/format and mypy passed |
| Secrets & .env gitignore | **PASS** | Scanned 113 files cleanly |
| Pytest & coverage | **PASS** | 65 passed, coverage 84% |
| Data contracts (Pandera) | **PASS** | Contracts passed |
| Model acceptance gates | **PASS** | Model MdAPE=0.06891, Best baseline MdAPE=0.09484, 90% coverage=0.9012 |
| API contract & latency tests | **PASS** | API tests passed (warm /predict median=14.65 ms) |
| Streamlit app smoke tests | **SKIPPED** | Scheduled for Phase 6 |
| Notebook execution | **PASS** | 5 notebooks executed |

## Measured Model & System Metrics

- Model MdAPE: `0.06891`
- Best Baseline MdAPE: `0.09484`
- Nominal 90% Interval Coverage: `0.9012`
- Spatial Hold-out MdAPE: `0.19908`
- API `/predict` Warm Latency (ms): `14.65`

## Skipped Checks & Justifications

- **Streamlit app smoke tests**: Scheduled for Phase 6
