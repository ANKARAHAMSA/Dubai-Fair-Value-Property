# Data Access & Provenance Guide (`DubaiFairValue`)

`DubaiFairValue` uses **official open data only** (or explicitly labeled fallback extracts where portal access is permission-gated). Listing portals (Bayut, Property Finder, Dubizzle, etc.) are **never scraped**.

## 1. Primary Official Data: Dubai Land Department (DLD) via Dubai Pulse

- Portal: <https://www.dubaipulse.gov.ae> (Organisation: `DLD`) and <https://dubailand.gov.ae/en/>
- Registration: Required (free account; some datasets require a permission request that may take up to ~14 days).

| Dataset ID | Purpose | Access Mode |
| --- | --- | --- |
| `dld_transactions-open` | Sales/mortgage/gift transactions | Manual CSV/Parquet in `data/raw/` or API |
| `dld_transactions-open-api` | Incremental API refresh | Dubai Pulse Data API (`DUBAI_PULSE_API_KEY` + `SECRET`) |
| `dld_rent_contracts-open` | Ejari residential rent contracts | Manual CSV/Parquet in `data/raw/` or API |
| `dld_projects-open` | Project metadata, developer, completion % | Manual CSV/Parquet in `data/raw/` or API |
| `dld_land_registry-open` | Land/plot zoning & area registry | Manual CSV/Parquet in `data/raw/` |
| `dld_residential_sale_index-open` | Official DLD sale-price index | Manual CSV/Parquet in `data/raw/` |
| `dld_real_estate_licenses-open` | Optional licensed agencies | Manual CSV/Parquet in `data/raw/` |

## 2. Transport Data: Roads & Transport Authority (RTA)

- Portal: <https://www.dubaipulse.gov.ae> (Organisation: `RTA`)
- Datasets: `rta_metro_stations-open`, `rta_metro_stations_gis-open`, `rta_tram_stations_gis-open`, `rta_bus_stops_gis-open`, `rta_gtfs-open`, `rta_metro_ridership-open`.
- **Fallback**: Community GTFS mirror on MobilityDatabase (<https://mobilitydatabase.org/feeds/mdb-904>, RTA Dubai Feed `mdb-904`), treated as **unofficial** and labeled accordingly in `MANIFEST.json`.
- Station opening dates are curated in `data/reference/stations_opening.csv` with source URLs; unknown dates are left null and excluded from timing-based analyses.

## 3. Geography, Context & Fallback Copies

- **OpenStreetMap**: Overpass/OSMNX queries for malls, schools, hospitals, beaches, parks, airports, major roads, and community boundaries; Nominatim geocoding at <= 1 req/s with persistent SQLite cache (`data/cache/geocode.sqlite`).
- **Curated Event Calendar**: `data/reference/events.csv` (every row includes a verifiable source URL).
- **Unofficial Community Fallback (Section 5.5)**: Hugging Face `viewit-ai/full-dubai-pulse` (`Transactions_clean.csv`). Used only when official `dld_transactions-open` is absent from `data/raw/` and explicitly labeled as an **unofficial convenience fallback**.
- **Synthetic Fallback**: `src/dfv/data/synth.py` generates realistic DLD/Ejari/project/station tables whenever gated datasets (`dld_rent_contracts-open`, `dld_projects-open`, etc.) are not yet present in `data/raw/`.

## 4. How to Run Ingestion (`make data`)

1. **Manual official download (recommended once approved)**: Place downloaded CSV/Parquet files in `data/raw/` (e.g., `dld_transactions-open.csv`, `dld_rent_contracts-open.csv`, `dld_projects-open.csv`, `rta_metro_stations-open.csv`).
2. **Optional API credentials**: Copy `.env.example` to `.env` and populate `DUBAI_PULSE_API_KEY` and `DUBAI_PULSE_API_SECRET`. Never commit `.env`.
3. Run:
   ```bash
   make data
   ```
   This ingests `data/raw/` files (or pulls configured API / fallback files if missing), records SHA-256 checksums and provenance in `data/raw/MANIFEST.json`, normalizes column names to `snake_case`, writes `docs/data_dictionary.md`, cleans transactions/rents into `data/processed/`, and validates Pandera contracts.
