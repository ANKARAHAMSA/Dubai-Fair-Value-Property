# Architectural & Engineering Decisions (`DubaiFairValue`)

This log records technical decisions made during implementation so choices remain transparent and reproducible.

## ADR-001: Project Root, Python Runtime, and Package Management
- **Decision**: Use Python `3.12` managed via `uv` (`pyproject.toml` + `uv.lock`) on macOS Apple Silicon (ARM64), with Homebrew `libomp` installed for LightGBM OpenMP support and zero CUDA dependencies.
- **Rationale**: Matches the target macOS Apple Silicon environment and tech-stack spec; `uv` gives deterministic, fast virtual environment creation and locked dependency resolution.

## ADR-002: Repository Layout and Packaging
- **Decision**: Use `src/dfv/` (`src`-layout) installed in editable mode (`uv sync`), with `tests/` split into `unit/`, `integration/`, `property/`, `data_contracts/`, `app/`, `api/`, `llm/`, and `fixtures/`.
- **Rationale**: Prevents accidental imports from the repo root instead of the installed package and cleanly separates test categories for CI and `make verify`.

## ADR-003: Phased Verification Gate (`make verify`)
- **Decision**: Implement a dedicated verification runner (`src/dfv/verify.py` invoked by `make verify`) starting in Phase 0 that runs lint, static checks, secret/gitignore checks, pytest with coverage (`>= 80%` on `src/dfv`), and phase-appropriate gates, writing `reports/VERIFICATION_REPORT.md`.
- **Rationale**: Ensures every phase passes `make verify` before committing. Checks for modules scheduled for future phases are explicitly marked `SKIPPED` with phase justification until their phase is completed, at which point all checks must pass.

## ADR-004: Synthetic-First + Explicit Schema Adapter Layer
- **Decision**: Build `src/dfv/data/synth.py` in Phase 0 to generate realistic DLD-like transactions, Ejari rent contracts, project metadata, and metro station tables with known planted structural effects (e.g., metro proximity premium, off-plan developer/area adjustments, area yield structure). All downstream code accesses canonical columns defined in `src/dfv/data/schema.py`.
- **Rationale**: Allows full end-to-end development, CI, and planted-effect recovery testing while waiting for gated Dubai Pulse approvals, and keeps open-data column renames isolated to a one-line mapping change in `schema.py`.

## ADR-005: Conformal Prediction Intervals
- **Decision**: Pair quantile LightGBM (`0.05`, `0.50`, `0.95` quantiles) with split-conformal calibration (conformalized quantile regression on a held-out temporal calibration slice) in `src/dfv/models/intervals.py`.
- **Rationale**: Raw quantile trees frequently mis-cover under temporal shift; CQR provides finite-sample marginal coverage guarantees within the required `90% ± 5%` test-set gate while preserving heteroscedastic interval widths.

## ADR-006: Observational Causality & Uncertainty Framing
- **Decision**: Label all observational analyses (off-plan developer premium, metro-proximity premium, market-cycle event study) explicitly as associations with confidence intervals and caveats unless quasi-experimental assumptions (pre-trend checks, placebo tests) are satisfied and documented.
- **Rationale**: Observational transaction data cannot rule out omitted confounding (e.g., unobserved building quality, floor height, payment plan terms).

## ADR-007: Natural-Language Search Safety & LLM Fallback
- **Decision**: Constrain `src/dfv/search/` so an optional LLM (env `LLM_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`, defaulting to Anthropic) only emits JSON matching `PropertyFilters` (Pydantic v2), retries once on validation failure, falls back deterministically to `rules.py`, and executes exclusively via parameterized DuckDB queries in `executor.py`.
- **Rationale**: Prevents SQL injection and prompt injection, guarantees zero-secret local execution and deployment, and allows fast offline mocked testing.

## ADR-008: Model Selection & Small-Sample Ridge + Residual LightGBM (`ResidualBoostedLGBM`)
- **Decision**: Use Optuna-tuned standalone `LightGBM (objective="regression_l1")` as the primary model on large datasets (`>= 20,000` training rows, such as the 459k DLD dataset), and use Stage-1 `Ridge` + Stage-2 Optuna-tuned `LightGBM` (`ResidualBoostedLGBM`) on small sample datasets (`< 20,000` training rows, such as the committed `data/sample/` CI bundle) while always reporting both standalone baselines and conformal CQR intervals.
- **Rationale**: On the full 459k DLD dataset, standalone Optuna LightGBM outperformed the Ridge-residual hybrid on strictly future 6-month test data (`MdAPE = 0.0689` vs `0.0805` hybrid and `0.0948` Ridge baseline) because nonlinear building/neighborhood interactions dominate and global linear time trends from 2021–2023 overshoot 2024 moderation. Conversely, on small sample slices (`3,500–5,000` rows), standalone trees overfit and cannot extrapolate synthetic trends beyond the training date cutoff, whereas regularized Ridge + Residual LightGBM beats Ridge while preserving exact additive Linear SHAP + TreeSHAP attribution.

## ADR-009: Two-Way Additive Fixed Effects for Route 2020 DiD & Centroid Fallback Disclosure
- **Decision**: Use additive two-way fixed effects ($\alpha_{\text{area}} + \gamma_{\text{quarter}}$ via alternating projections) in `run_route2020_event_study`, prune near-collinear regressors via incremental Gram-Schmidt QR before HC1 OLS in `src/dfv/analysis/_stats.py`, and explicitly flag `within_area_identifiable` in `analyze_metro_effect`.
- **Rationale**: Interacted `area_name × year_quarter` fixed effects completely absorb area-level `Treated_a × Post_t` DiD indicators under centroid-level geocoding. Similarly, area-centroid real fallback coordinates carry zero within-area metro distance variance, so within-area sales distance bands cannot be identified until project-level coordinates/`dld_projects-open` are supplied; surfacing cross-area bands with an explicit confounding caveat alongside synthetic/rent bands avoids opaque empty outputs.
