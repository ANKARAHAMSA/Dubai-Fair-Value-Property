# DubaiFairValue Data Dictionary

Auto-generated from actual ingested tables (`src/dfv/data/schema.py`).
All upstream column names are normalized to `snake_case` via the adapter layer.

## Table: `transactions_raw`

- **Source**: `unofficial_hf_fallback (viewit-ai/full-dubai-pulse)`
- **Rows**: `1,276,960` | **Columns**: `19`

| Column | Dtype | Null % | Min | Max | Example Values |
| --- | --- | --- | --- | --- | --- |
| `procedure_name` | `str` | 0.0% | `Adding Land By Sell` | `Transfer Development Mor` | Sell - Pre registration, Delayed Sell, Delayed Sell |
| `instance_date` | `str` | 0.0% | `01-01-1975` | `31-12-2021` | 22-08-2024, 22-08-2024, 22-08-2024 |
| `property_type` | `str` | 0.0% | `Building` | `Villa` | Unit, Land, Unit |
| `property_sub_type` | `str` | 22.0% | `Building` | `Workshop` | Flat, Hotel Apartment, Flat |
| `property_usage` | `str` | 0.0% | `Agricultural` | `Storage` | Residential, Residential, Hospitality  |
| `area_name` | `str` | 0.0% | `Abu Hail` | `Zareeba Duviya` | Al Khairan First, Hadaeq Sheikh Mohammed B, Jabal Ali First |
| `building_name` | `str` | 30.7% | `  PALLADIUM` | `canceled building` | Palace Residences Creek , CANDACE ACACIA HOTEL APA, Marina Views Tower 1 |
| `master_project` | `str` | 17.9% | `800 Villas` | `Wasl Gate` | The Lagoons, DUBAI HILLS, Al Furjan |
| `nearest_landmark` | `str` | 14.8% | `Al Makhtoum Internationa` | `Sports City Swimming Aca` | IMG World Adventures, Sports City Swimming Aca, Sports City Swimming Aca |
| `nearest_metro` | `str` | 25.5% | `ADCB Metro Station` | `Union Metro Station` | Noor Bank Metro Station, Ibn Battuta Metro Statio, Nakheel Metro Station |
| `nearest_mall` | `str` | 25.9% | `City Centre Mirdif` | `Marina Mall` | Dubai Mall, Ibn-e-Battuta Mall, Marina Mall |
| `rooms_label` | `str` | 23.3% | `1 B/R` | `Studio` | 1 B/R, 1 B/R, 3 B/R |
| `has_parking` | `int64` | 0.0% | `0` | `1` | 1, 0, 1 |
| `area_sqft` | `float64` | 0.0% | `0.11` | `3.682e+09` | 742.82, 13273.83, 765.31 |
| `actual_worth` | `float64` | 0.0% | `1` | `1.379e+10` | 2048888.0, 40500000.0, 593340.0 |
| `price_per_sqft` | `float64` | 0.0% | `0` | `7.548e+06` | 2758.27, 3051.12, 775.29 |
| `embedded_rent_value` | `float64` | 97.3% | `2` | `2.621e+08` | 964600.0, 2662400.0, 900000.0 |
| `embedded_rent_psf` | `float64` | 97.3% | `0` | `1.031e+06` | 671.77, 2162.3, 931.72 |
| `data_source` | `str` | 0.0% | `unofficial_hf_fallback (` | `unofficial_hf_fallback (` | unofficial_hf_fallback (, unofficial_hf_fallback (, unofficial_hf_fallback ( |

## Table: `rent_contracts_raw`

- **Source**: `synthetic_fallback`
- **Rows**: `10,000` | **Columns**: `17`

| Column | Dtype | Null % | Min | Max | Example Values |
| --- | --- | --- | --- | --- | --- |
| `contract_id` | `str` | 0.0% | `SYN-EJ-000001` | `SYN-EJ-010000` | SYN-EJ-000001, SYN-EJ-000002, SYN-EJ-000003 |
| `contract_start_date` | `str` | 0.0% | `2015-01-01` | `2026-06-29` | 2026-06-17, 2024-04-08, 2020-11-19 |
| `contract_end_date` | `str` | 0.0% | `2016-01-01` | `2027-06-29` | 2027-06-17, 2025-04-08, 2021-11-19 |
| `contract_type` | `str` | 0.0% | `New` | `Renew` | Renew, Renew, Renew |
| `is_renewal` | `int64` | 0.0% | `0` | `1` | 1, 1, 1 |
| `property_type` | `str` | 0.0% | `Unit` | `Villa` | Unit, Unit, Villa |
| `property_sub_type` | `str` | 0.0% | `Flat` | `Villa` | Flat, Flat, Villa |
| `property_usage` | `str` | 0.0% | `Residential` | `Residential` | Residential, Residential, Residential |
| `area_name` | `str` | 0.0% | `Al Furjan` | `Palm Jumeirah` | Dubai Marina, Jumeirah Lake Towers, Palm Jumeirah |
| `project_name` | `str` | 0.0% | `Al Furjan Residences 1` | `Palm Jumeirah Residences` | Dubai Marina Residences , Jumeirah Lake Towers Res, Palm Jumeirah Residences |
| `rooms` | `int64` | 0.0% | `0` | `5` | 1, 2, 3 |
| `area_sqm` | `float64` | 0.0% | `28.78` | `683.5` | 63.12, 124.23, 191.0 |
| `annual_amount` | `float64` | 0.0% | `1.593e+04` | `8.78e+05` | 97131.65, 164267.08, 211410.69 |
| `rent_per_sqm` | `float64` | 0.0% | `381.8` | `2162` | 1538.75, 1322.32, 1106.87 |
| `nearest_metro` | `str` | 0.0% | `Al Furjan` | `The Gardens` | DMCC, Sobha Realty, Nakheel |
| `metro_distance_m` | `float64` | 0.0% | `133` | `1.302e+04` | 439.5, 759.1, 439.5 |
| `data_source` | `str` | 0.0% | `synthetic_fallback` | `synthetic_fallback` | synthetic_fallback, synthetic_fallback, synthetic_fallback |

## Table: `projects_raw`

- **Source**: `synthetic_fallback`
- **Rows**: `75` | **Columns**: `17`

| Column | Dtype | Null % | Min | Max | Example Values |
| --- | --- | --- | --- | --- | --- |
| `project_id` | `str` | 0.0% | `PRJ-0001` | `PRJ-0075` | PRJ-0001, PRJ-0002, PRJ-0003 |
| `project_name` | `str` | 0.0% | `Al Furjan Residences 1` | `Palm Jumeirah Residences` | Palm Jumeirah Residences, Palm Jumeirah Residences, Palm Jumeirah Residences |
| `developer_name` | `str` | 0.0% | `Azizi Developments` | `Sobha Realty` | Meraas, Nakheel, Select Group |
| `area_name` | `str` | 0.0% | `Al Furjan` | `Palm Jumeirah` | Palm Jumeirah, Palm Jumeirah, Palm Jumeirah |
| `lat` | `float64` | 0.0% | `24.89` | `25.22` | 25.110481, 25.106751, 25.103131 |
| `lon` | `float64` | 0.0% | `55.11` | `55.34` | 55.152758, 55.153336, 55.147485 |
| `nearest_metro` | `str` | 0.0% | `Al Furjan` | `The Gardens` | Nakheel, Nakheel, Nakheel |
| `metro_distance_m` | `float64` | 0.0% | `133` | `1.302e+04` | 279.7, 439.5, 759.1 |
| `metro_opening_date` | `str` | 0.0% | `2009-04-30` | `2021-06-01` | 2009-04-30, 2009-04-30, 2009-04-30 |
| `is_route_2020_metro` | `bool` | 0.0% | `0` | `1` | False, False, False |
| `nearest_mall` | `str` | 0.0% | `Dubai Mall` | `Mall of the Emirates` | Dubai Marina Mall, Dubai Marina Mall, Dubai Marina Mall |
| `nearest_landmark` | `str` | 0.0% | `Al Furjan Central` | `Palm Jumeirah Central` | Palm Jumeirah Central, Palm Jumeirah Central, Palm Jumeirah Central |
| `project_effect` | `float64` | 0.0% | `-0.07462` | `0.102` | 0.026265791853226007, -0.045576282740181134, -0.0005880405126501079 |
| `status` | `str` | 0.0% | `Finished` | `Under Construction` | Finished, Finished, Finished |
| `completion_pct` | `float64` | 0.0% | `24` | `100` | 100.0, 100.0, 100.0 |
| `launch_date` | `str` | 0.0% | `2013-06-01` | `2024-10-01` | 2014-08-01, 2014-03-01, 2021-09-01 |
| `completion_date` | `str` | 0.0% | `2015-07-01` | `2027-10-01` | 2018-06-01, 2018-02-01, 2025-10-01 |

## Table: `metro_stations_raw`

- **Source**: `unofficial_mobilitydatabase_gtfs (mdb-904)`
- **Rows**: `53` | **Columns**: `7`

| Column | Dtype | Null % | Min | Max | Example Values |
| --- | --- | --- | --- | --- | --- |
| `station_name` | `str` | 0.0% | `ADCB` | `e&` | ADCB, Abu Baker Al Siddique, Abu Hail |
| `lat` | `float64` | 0.0% | `24.96` | `25.28` | 25.24476136728795, 25.270885629782548, 25.275317383272 |
| `lon` | `float64` | 0.0% | `55.09` | `55.4` | 55.2982809418214, 55.332846872977754, 55.346378886791854 |
| `line` | `str` | 0.0% | `Green` | `Red/Green` | Red, Green, Green |
| `opening_date` | `str` | 0.0% | `2009-09-09` | `2021-06-01` | 2010-04-30, 2011-09-09, 2011-09-09 |
| `is_route_2020` | `bool` | 0.0% | `0` | `1` | False, False, False |
| `data_source` | `str` | 0.0% | `unofficial_mobilitydatab` | `unofficial_mobilitydatab` | unofficial_mobilitydatab, unofficial_mobilitydatab, unofficial_mobilitydatab |

## Table: `sale_index_raw`

- **Source**: `synthetic_fallback`
- **Rows**: `138` | **Columns**: `4`

| Column | Dtype | Null % | Min | Max | Example Values |
| --- | --- | --- | --- | --- | --- |
| `month` | `str` | 0.0% | `2015-01-01` | `2026-06-01` | 2015-01-01, 2015-02-01, 2015-03-01 |
| `index_value` | `float64` | 0.0% | `91.55` | `154.8` | 116.73, 117.12, 114.01 |
| `property_type` | `str` | 0.0% | `All Residential` | `All Residential` | All Residential, All Residential, All Residential |
| `data_source` | `str` | 0.0% | `synthetic` | `synthetic` | synthetic, synthetic, synthetic |
