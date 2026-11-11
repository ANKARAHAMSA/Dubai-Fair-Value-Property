# Model Card (`DubaiFairValue`)

## 1. Model Overview & Intended Use

- **Task:** Predict residential property transaction `log(price_per_sqm)` (and total price `AED = exp(y_hat) * area_sqm`) with calibrated 90% conformal uncertainty bounds and plain-English per-property cost drivers.
- **Intended Users & Use Cases:** Buyers, renters, data science reviewers, and analysts screening Dubai residential asking prices against historical DLD sales benchmarks, exploring metro/yield/cycle patterns, and testing calibrated uncertainty quantification.
- **Out-of-Scope / Prohibited Use:** Do **not** use as a replacement for RERA/RICS-certified appraisals, mortgage underwriting, legal disputes, or binding financial advice.

---

## 2. Training Data, Architecture & Leakage Controls

### Dataset Modes & Model Architecture (ADR-008)

| Mode | Rows (`train / val / cal / test`) | Date Window | Selected Architecture | Reason |
| --- | --- | --- | --- | --- |
| **Processed Real DLD Fallback** (`lightgbm_tuned`) | `459,145` total (`332,060 / 27,938 / 28,539 / 70,608`) | `2015-01-01` -> `2024-08-22` (Test: last 6m `2024-02-23` -> `2024-08-22`) | Optuna-tuned LightGBM (`objective="regression_l1"`, 45 features) | Beats Ridge baseline & Ridge-residual hybrid on strictly future test slice (`6.94%` vs `8.14%` vs `9.50%` MdAPE) |
| **Sample CI Bundle** (`data/sample/`) | `5,000` total (`4,471 / 130 / 135 / 264`, `2015-01-12` -> `2026-06-29`) | Test: `2025-12-30` -> `2026-06-29` | `ResidualBoostedLGBM` (Stage-1 Ridge + Stage-2 Optuna LightGBM) | Handles small-N trend extrapolation beyond date cutoff cleanly (`6.45%` vs `6.72%` Ridge baseline) |

### Leakage Prevention Guarantees (`src/dfv/features/leakage.py`)
- Strict chronological split with zero boundary-date overlap (`max(train_date) < min(val_date) <= max(val_date) < min(cal_date) <= max(cal_date) < min(test_date)`).
- Time-aware station opening gating (`opening_date <= instance_date`) prevents Route 2020 (2021) stations from appearing on pre-2021 sales, and `completion_pct` on historical off-plan rows is interpolated from launch/completion spans rather than static snapshots (`0%` pre-2020 off-plan `completion_pct == 100`).
- Trailing `3m/6m/12m` area, project, segment, and yield medians use exclusively transactions strictly prior to each row's `instance_date` (`future-deletion invariant` verified by unit tests).
- High-cardinality `developer_name` and `project_name` out-of-fold (`OOF`) target encodings use fold-specific priors computed strictly inside `train` folds (`5-fold OOF`) and mapped onto `val/cal/test` using only training targets.
- Target price columns (`actual_worth`, `price_per_sqm`, `price_per_sqft`) are explicitly blacklisted from model feature inputs.

---

## 3. Quantitative Performance vs Baselines

### Processed Full DLD Model (`lightgbm_tuned`, Test Set n = `70,608`)

| Model | Test MdAPE | Test MAPE | Log RMSE | Log R^2 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_global_median` | 33.32% | 35.20% | 0.5496 | -0.5899 | Global median price/sqm x area |
| `baseline_trailing_12m_segment` | 13.07% | 20.01% | 0.2758 | 0.5995 | Trailing 12m `area x type x rooms` |
| `baseline_ridge_log` | 9.50% | 13.05% | 0.1791 | 0.8312 | Best baseline (L2 linear on log price/sqm) |
| `ridge_residual_lightgbm` | 8.14% | 11.14% | 0.1555 | 0.8727 | Evaluated alternative hybrid |
| **`lightgbm_tuned` (Primary Served)** | **`6.94%`** | **`10.00%`** | **`0.1454`** | **`0.8887`** | **+27.00% relative MdAPE reduction vs best baseline** |

- **Optuna Hyperparameters (10 trials, seed `42`):** `n_estimators=202`, `learning_rate=0.0774`, `num_leaves=99`, `min_child_samples=71`, `subsample=0.9314`, `colsample_bytree=0.6759`.
- **Deterministic Seed Reproducibility:** Max absolute log-prediction diff across identical reruns = `0.0`.
- **FastAPI `/predict` Latency:** Warm median latency = `14.67 ms` (< 500 ms budget).

---

## 4. Conformal Interval Calibration & Verdict Logic

- **Method:** Quantile LightGBM (`0.05` / `0.95` quantiles) wrapped with split-conformalized quantile regression (`CQR`) fitted on the strictly separated `28,539`-row calibration period (`2023-11-21` -> `2024-02-22`, strictly after `val_tune` `2023-08-23` -> `2023-11-20`).
- **Measured Test Coverage (Nominal 90% Target):** **`89.93%`** (gate `[85.0%, 95.0%]` **PASS**).
- **Median Interval Width (Test):** `43.75%` relative width (`AED 592,299` median absolute span).
- **Sample CI Bundle CQR Coverage:** **`91.67%`** (`35.52%` median relative width on `264` test rows).
- **Verdict Abstention (`src/dfv/models/verdict.py`):** Segments with `< 15` trailing historical transactions or unrecognized/unknown areas return `"insufficient data"` alongside the calibrated interval rather than forcing a buy/overpriced label.

---

## 5. Honest Segment Performance, Spatial Hold-Out & Temporal Drift

### Segment Error Breakdown ([`reports/error_analysis.md`](../reports/error_analysis.md))

| Segment | Test Rows | MdAPE | 90% CQR Coverage |
| --- | ---: | ---: | ---: |
| All Test Transactions | 70,608 | 6.94% | 89.9% |
| Apartments (`Unit`) | 62,613 | 6.41% | 90.4% |
| Off-Plan Sales | 48,345 | 6.29% | 89.4% |
| Thin-Data Areas (`<150` trailing sales) | 1,077 | 7.42% | 92.9% |
| Ready / Secondary Sales | 22,263 | 8.68% | 91.0% |
| **Luxury (`>= AED 5M`)** | **3,231** | **13.18%** | **81.0%** |
| Villas & Townhouses | 7,995 | 13.53% | 86.0% |

### Spatial (Unseen-Project) Hold-Out Degradation (Important)
Holding out `680` entire residential projects (`88,057` test rows, recomputing area/segment trailing priors strictly excluding held-out project sales) increases MdAPE sharply from **`6.94%`** to **`20.22%`** (`log R^2 = 0.6095`). Why:
1. Trailing project-level features (`project_trailing_6m_ppsqm`, `project_trailing_12m_ppsqm`, `project_name_oof_enc`) fall back to community medians on brand-new projects.
2. Fallback DLD records lack building-level coordinates and developer metadata (`developer_name == "Unknown"`), removing developer tier and micro-location signals for unseen towers.

### Temporal Drift & Rolling Cross-Validation
- **Rolling-Origin 6m Folds (MdAPE):** `11.26%` (`2022-08`), `10.47%` (`2023-02`), `9.65%` (`2023-08`).
- **Monthly Test Drift (2024):** Widens gradually across the 6-month holdout from `6.52%` (`2024-02`, `90.9%` coverage) to `7.74%` (`2024-08`, `87.9%` coverage).

---

## 6. SHAP Log-to-AED Approximation & Limitations

`TreeExplainer` produces exact additive contributions in `log(AED/sqm)`. Converting each log contribution `phi_j` into AED impacts uses Mean Value Theorem (MVT) scaling anchored around total predicted delta: `Delta_total = area_sqm * (exp(base + sum(phi)) - exp(base))`. This guarantees exact additivity (`sum(AED impacts) == exp(y_hat)*area - exp(base)*area`) while smoothing nonlinear exponential curvature. Because tree features correlate strongly (e.g., `area_trailing_12m_ppsqm` vs `segment_trailing_12m_ppsqm`), individual drivers describe **model attribution**, not independent causal effects.

---

## 7. Retraining & Maintenance Procedure

1. Drop fresh `dld_transactions-open.csv`, `dld_rent_contracts-open.csv`, `dld_projects-open.csv`, and index extracts into `data/raw/`.
2. Execute full pipeline:
   ```bash
   make data features train analyze render verify
   ```
3. Confirm in `reports/VERIFICATION_REPORT.md` that:
   - Out-of-time `lightgbm_tuned` beats all baselines,
   - Nominal 90% CQR coverage stays within `[85%, 95%]`,
   - Zero leakage or data-contract tests fail,
   - Git SHA, timestamp, and data manifest hash update inside `data/processed/models/metadata.json`.
