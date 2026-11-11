# DubaiFairValue Verification Report

- **Timestamp (UTC)**: `2026-11-11T06:17:56Z`
- **Git SHA**: `840439a`
- **Data Manifest Hash**: `946ce591cc0c55b7`
- **Test Count**: `43`
- **Code Coverage (`src/dfv`)**: `83%`
- **Overall Status**: `PASS`

## Check Summary

| Check | Status | Details |
| --- | --- | --- |
| Lint & mypy | **PASS** | ruff check/format and mypy passed |
| Secrets & .env gitignore | **PASS** | Scanned 78 files cleanly |
| Pytest & coverage | **PASS** | 43 passed, coverage 83% |
| Data contracts (Pandera) | **PASS** | Contracts passed |
| Model acceptance gates | **PASS** | Model MdAPE=0.06891, Best baseline MdAPE=0.09484, 90% coverage=0.9012 |
| API contract & latency tests | **SKIPPED** | Scheduled for Phase 5 |
| Streamlit app smoke tests | **SKIPPED** | Scheduled for Phase 6 |
| Notebook execution | **SKIPPED** | Scheduled for Phase 4 |

## Measured Model & System Metrics

- Model MdAPE: `0.06891`
- Best Baseline MdAPE: `0.09484`
- Nominal 90% Interval Coverage: `0.9012`
- Spatial Hold-out MdAPE: `0.19908`
- API `/predict` Warm Latency (ms): `n/a`

## Skipped Checks & Justifications

- **API contract & latency tests**: Scheduled for Phase 5
- **Streamlit app smoke tests**: Scheduled for Phase 6
- **Notebook execution**: Scheduled for Phase 4
