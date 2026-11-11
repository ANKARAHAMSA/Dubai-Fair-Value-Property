# LinkedIn Showcase Draft (`DubaiFairValue`)

## Post Copy (Verified <= 250 Words)

Buying or pricing a home in Dubai often means guessing off portal asking prices with zero transparency on uncertainty.

So I built **DubaiFairValue** — an end-to-end open-data valuation, conformal uncertainty, and market-intelligence engine trained on **459,145 unflagged Dubai Land Department (DLD) residential sales** (cleaned from 1.28M public records through August 2024).

Three findings that stood out when the numbers ran:

1. **Uncertainty matters more than point predictions:** On a strict 6-month future hold-out (`70,608` sales), tuned LightGBM reached **6.94% MdAPE** (27.0% lower error than log-Ridge at 9.50%), while Split-Conformalized Quantile Regression hit **89.93%** empirical 90% coverage — yet holding out **680 unseen projects** jumped error to **20.22%**, proving spatial metadata drives new-launch accuracy.
2. **Same-building controls shrink the off-plan gap:** Across Area x Quarter fixed effects, off-plan sales traded at a **+28.27% association** over ready units — dropping to **+5.48%** inside the same Project x Quarter, highlighting mix and unobserved payment-plan effects.
3. **Rentals & liquidity reward mid-market hubs:** On calibrated synthetic Ejari fallback rents pending DLD rent approval, units within 500m of Metro showed **+10.72% higher rents** vs >=2 km, with JVC leading gross yield-per-risk at **8.36%** (pre-fee).

Built with Python, DuckDB, Pandera, LightGBM, SHAP, FastAPI (`14.7 ms` warm `/predict`), parameterized NL search, and a 10-page Streamlit app.

Explore it here: [Live Dashboard Link Placeholder] | Code: https://github.com/ANKARAHAMSA/Dubai-Fair-Value-Property

#DubaiRealEstate #DataScience #MachineLearning #PropTech #MLOps #ConformalPrediction #UAE

---

## 6-Slide Carousel Outline

- **Slide 1 (Hook & Problem):** *"Is this Dubai Marina 2-bed actually overpriced?"* — Problem: headline asking prices lack uncertainty intervals, transparent comparables, and leakage-free benchmarks. Show Fair-Value Gauge (`estimate`, `[low, high]`, `Underpriced / Fair / Overpriced`).
- **Slide 2 (Data Hygiene & Realism Checks):** `1,276,960` public DLD extracts -> `459,145` unflagged residential sales (`2015–2024`). Show Pandera contracts, rule attrition, and Dubai realism bounds (`Downtown AED 26.8k/sqm > Dubai South AED 10.9k/sqm`).
- **Slide 3 (Leakage-Safe Valuation & Conformal Uncertainty):** Strict chronological splits (`max train < min val < min cal < min test`), LightGBM `6.94%` MdAPE vs `9.50%` Ridge baseline, `89.93%` CQR coverage, and log-to-AED MVT SHAP drivers.
- **Slide 4 (Honest Failure Analysis — What the Model Gets Wrong):** Why unseen projects jump to `20.22%` MdAPE (missing building coordinates + developer registry) and why luxury `>= AED 5M` (`13.18%`) & villas (`13.53%`) lag due to unobserved views & floor height.
- **Slide 5 (Dubai Market Analytics & Time-Travel):** Area x Quarter (`+28.27%`) vs Project x Quarter (`+5.48%`) off-plan association; on calibrated synthetic Ejari fallback rents, `+10.72%` `<500m` Metro band & JVC `8.36%` gross yield, plus `price_growth_timelapse.gif`.
- **Slide 6 (Engineering & Guarded Serving):** FastAPI (`14.7 ms` warm `/predict`), NL-to-Pydantic + parameterized DuckDB search (zero raw LLM SQL, deterministic rule fallback without keys), `83` automated tests at `89%` coverage, open-source repo & live app link.
