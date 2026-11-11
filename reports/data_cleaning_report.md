# DubaiFairValue Data Cleaning Report

## Sales Transactions Cleaning Rules

| Rule | Rows Before | Rows After | Rows Affected | Action | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| Filter residential sales (exclude mortgages/gifts/land/commercial) | 1,276,960 | 710,801 | 566,159 | `dropped` |  |
| Parse dates and enforce range [2015-01-01, 2026-12-11] | 710,801 | 490,288 | 220,513 | `dropped` |  |
| Remove non-positive/missing price or area; compute price_per_sqm & price_per_sqft | 490,288 | 490,288 | 0 | `dropped` |  |
| Standardize area, project, developer, and metro names via name_map + rapidfuzz | 490,288 | 490,288 | 125 | `checked` | 125 distinct canonical areas |
| Flag exact duplicate transactions | 490,288 | 490,288 | 19,362 | `flagged` |  |
| Flag non-market & hard sanity-bound violations | 490,288 | 490,288 | 1,025 | `flagged` |  |
| Flag statistical outliers (|MAD z-score| > 3.5 within area x type x year) | 490,288 | 490,288 | 11,918 | `flagged` |  |

## Rent Contracts Cleaning Rules

| Rule | Rows Before | Rows After | Rows Affected | Action | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| Filter residential rent contracts | 10,000 | 10,000 | 0 | `dropped` |  |
| Parse rent dates >= 2015-01-01 | 10,000 | 10,000 | 0 | `dropped` |  |
| Enforce positive rent & area bounds; compute rent_per_sqm | 10,000 | 10,000 | 0 | `dropped` |  |

## Dubai-Specific Realism Checks

- **downtown_vs_dubai_south**: `{'downtown': 26786.4, 'dubai_south': 10945.8}`
- **palm_vs_discovery_gardens**: `{'palm': 31999.2, 'discovery_gardens': 9183.7}`
