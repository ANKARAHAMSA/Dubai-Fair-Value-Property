# Data Card (`DubaiFairValue`)

## 1. Sources, Access Mode, and Stated Terms / Licenses

`DubaiFairValue` uses **official open-data portals or transparently labeled fallback extracts only**. Consumer listing portals (Bayut, Property Finder, Dubizzle) are never scraped.

| Dataset Role | Source & Portal URL | Portal License / Terms | Mode in Current Workspace |
| --- | --- | --- | --- |
| **Residential Sales Transactions** (`dld_transactions-open`) | Primary target: [Dubai Pulse / DLD](https://www.dubaipulse.gov.ae) & [Dubai Land Department](https://dubailand.gov.ae/en/) · Fallback copy: [`viewit-ai/full-dubai-pulse`](https://huggingface.co/datasets/viewit-ai/full-dubai-pulse) (`Transactions_clean.csv`) | **License not specified on portal / mirror** (Dubai Open Data terms; attribution required) | `unofficial_hf_fallback` (`1,276,960` raw rows through `2024-08-22`, SHA-256 `077c0027f6be4b8c...`) |
| **RTA Dubai Metro Stations & GTFS** (`rta_gtfs-open` / `rta_metro_stations-open`) | Primary target: [Dubai Pulse / RTA](https://www.dubaipulse.gov.ae) · Fallback GTFS mirror: [MobilityDatabase `mdb-904`](https://mobilitydatabase.org/feeds/mdb-904) (`53` deduplicated metro stations) paired with curated official opening dates ([`stations_opening.csv`](../data/reference/stations_opening.csv)) | **License not specified on RTA portal**; MobilityDatabase feed redistributed as public transit feed metadata | `unofficial_mobilitydatabase_gtfs (mdb-904)` (`53` stations) |
| **Ejari Rent Contracts** (`dld_rent_contracts-open`) | [Dubai Pulse / DLD](https://www.dubaipulse.gov.ae) (permission-gated) | License not specified on portal (requires portal approval) | `synthetic_fallback` (`10,000` structured contracts calibrated to Dubai communities) |
| **Project Registry Metadata** (`dld_projects-open`) | [Dubai Pulse / DLD](https://www.dubaipulse.gov.ae) (permission-gated) | License not specified on portal (requires portal approval) | `synthetic_fallback` (`75` seeded community projects; real DLD sales lack project coordinates/developers until granted) |
| **Official Sale Price Index** (`dld_residential_sale_index-open`) | [Dubai Pulse / DLD](https://www.dubaipulse.gov.ae) (permission-gated) | License not specified on portal (requires portal approval) | `synthetic_fallback` (`138` monthly points, `2015-01` to `2026-06`) |
| **Macro & Policy Event Calendar** | Manually curated citations ([`events.csv`](../data/reference/events.csv)): [u.ae](https://u.ae/en/), [rta.ae](https://www.rta.ae), [bie-paris.org](https://www.bie-paris.org/site/en/2020-dubai), [dubailand.gov.ae](https://dubailand.gov.ae/en/) | Public official press releases with per-row source URLs | Verified reference CSV (`6` major policy/infrastructure events) |
| **Points of Interest & Landmarks** | OpenStreetMap (via Nominatim/OSMnx) | Open Database License (**ODbL 1.0**) | Cached community centroids + landmark/coast coordinates |

For step-by-step instructions to swap in permission-granted official CSVs, see [`docs/DATA_ACCESS.md`](DATA_ACCESS.md).

---

## 2. Cleaning Steps & Measured Row Attrition

All rules execute deterministically in [`src/dfv/data/clean.py`](../src/dfv/data/clean.py) and log exact counts to [`reports/data_cleaning_report.md`](../reports/data_cleaning_report.md):

| Cleaning Step (Sales) | Rows Before | Rows After | Rows Affected | Action |
| --- | ---: | ---: | ---: | --- |
| Filter residential sales (exclude mortgages, gifts, land, commercial) | 1,276,960 | 710,801 | 566,159 | `dropped` |
| Parse dates & enforce plausible range (`2015-01-01` to `2026-12-11`) | 710,801 | 490,288 | 220,513 | `dropped` |
| Remove missing/non-positive price or area; compute AED/sqm & AED/sqft | 490,288 | 490,288 | 0 | `dropped` |
| Normalize Arabic/English community & metro labels (`name_map.csv` + `rapidfuzz`) | 490,288 | 490,288 | 125 canonical areas | `checked` |
| Flag exact duplicate transactions | 490,288 | 490,288 | 19,362 | `flagged` (`flag_duplicate`) |
| Flag non-market values & hard AED/sqm bounds (`1,500–250,000` AED/sqm) | 490,288 | 490,288 | 1,025 | `flagged` (`flag_non_market`) |
| Flag statistical outliers (`\|MAD z-score\| > 3.5` within `area x type x year`) | 490,288 | 490,288 | 11,918 | `flagged` (`flag_outlier`) |
| **Clean unflagged training/evaluation pool** | `490,288` | **`459,145`** | `31,143 flagged total` | Retained unflagged rows |

**Dubai-Specific Realism Checks (Verified Pass):**
- `Downtown Dubai` vs `Dubai South` median price/sqm: `AED 26,786.4 / sqm` > `AED 10,945.8 / sqm`.
- `Palm Jumeirah` vs `Discovery Gardens` median price/sqm: `AED 31,999.2 / sqm` > `AED 9,183.7 / sqm`.

---

## 3. Committed CI / Sample Bundle (`data/sample/`)

To allow zero-secret, clean-clone testing and Streamlit Cloud deployment without redistributing raw DLD extracts (`209 MB`):
- `data/sample/` (`2.18 MB` total on disk including sample analysis JSONs and `ResidualBoostedLGBM` CI model) contains `5,000` sales transactions (`2015-01-12` to `2026-06-29`), `5,000` synthetic rent contracts, `14` core corridor metro stations, and precomputed sample analysis tables.
- All Pandera schema contracts (`TransactionsContract`, `RentsContract`, `ProjectsContract`, `MetroStationsContract`, `SaleIndexContract`) pass on both `data/processed/` and `data/sample/`.

---

## 4. Known Data Gaps & Methodological Caveats

1. **No unit-level coordinates (area-centroid fallback):** Neither DLD transaction CSVs nor the Hugging Face extract publish latitude/longitude per apartment/building. `dist_nearest_metro_m` relies on dataset-provided nearest metro labels paired with `41` geocoded area centroids across `122` active communities (`within_area_identifiable = false` on sales metro distance bands until `dld_projects-open` / project coordinates are linked).
2. **Missing developer identities in sales extract:** `developer_name == "Unknown"` for `100.0%` of fallback sales transactions, preventing named-developer league tables until `dld_projects-open` is approved.
3. **Off-plan registration timing lag & payment-plan omissions:** DLD Oqood registrations can reflect developer launch dates rather than contract signing dates, and post-handover payment plans (e.g., 60/40 vs 5-year post-handover) are absent from tabular registries despite moving headline prices.
4. **Rental yield & renewal caveats:** Ejari synthetic fallback distinguishes `New` vs `Renew` contracts (yield ranking uses `New` contracts), while gross rental yield (`annual rent / purchase price`) excludes Dubai service charges (`AED 10–35/sqft/yr`), vacancy periods, furnishing, and DLD/agency closing fees (`~6–7%`).
5. **Unobserved unit features:** Floor number, panoramic sea/Burj Khalifa/golf views, terrace/upgrade condition, and corner layouts are unrecorded, driving high tail error on luxury units (`>= AED 5M`).
