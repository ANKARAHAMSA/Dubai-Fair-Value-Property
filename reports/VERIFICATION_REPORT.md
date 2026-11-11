# DubaiFairValue Verification Report

- **Timestamp (UTC)**: `2026-11-11T06:01:56Z`
- **Git SHA**: `3a1b88a`
- **Data Manifest Hash**: `946ce591cc0c55b7`
- **Test Count**: `36`
- **Code Coverage (`src/dfv`)**: `80%`
- **Overall Status**: `PASS`

## Check Summary

| Check | Status | Details |
| --- | --- | --- |
| Lint & mypy | **PASS** | ruff check/format and mypy passed |
| Secrets & .env gitignore | **PASS** | Scanned 60 files cleanly |
| Pytest & coverage | **PASS** | 36 passed, coverage 80% |
| Data contracts (Pandera) | **PASS** | Contracts passed |
| Model acceptance gates | **SKIPPED** | Scheduled for Phase 3 |
| API contract & latency tests | **SKIPPED** | Scheduled for Phase 5 |
| Streamlit app smoke tests | **SKIPPED** | Scheduled for Phase 6 |
| Notebook execution | **SKIPPED** | Scheduled for Phase 4 |

## Measured Model & System Metrics

- Model metrics not yet generated (Phase 3 pending).

## Skipped Checks & Justifications

- **Model acceptance gates**: Scheduled for Phase 3
- **API contract & latency tests**: Scheduled for Phase 5
- **Streamlit app smoke tests**: Scheduled for Phase 6
- **Notebook execution**: Scheduled for Phase 4
