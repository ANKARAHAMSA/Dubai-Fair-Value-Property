"""Data acquisition pipeline (`make data`).

Priority order:
1. Official CSV/Parquet files manually placed in data/raw/
2. Official Dubai Pulse API if DUBAI_PULSE_API_KEY & DUBAI_PULSE_API_SECRET exist in .env
3. Documented unofficial open mirrors (Hugging Face viewit-ai/full-dubai-pulse & MobilityDatabase mdb-904)
4. Synthetic fallback (src/dfv/data/synth.py) for any missing/gated dataset, logged clearly
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from dfv.config import PATHS, Paths, ensure_dirs
from dfv.data.schema import (
    PROJECTS_COLUMN_MAP,
    RENTS_COLUMN_MAP,
    SALE_INDEX_COLUMN_MAP,
    STATIONS_COLUMN_MAP,
    TRANSACTIONS_COLUMN_MAP,
    adapt_dataframe,
    write_data_dictionary,
)
from dfv.data.synth import generate_synthetic_bundle

logger = logging.getLogger("dfv.data.download")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_TRANSACTIONS_URL = (
    "https://huggingface.co/datasets/viewit-ai/full-dubai-pulse/resolve/main/Transactions_clean.csv"
)
MDB_GTFS_URL = (
    "https://files.mobilitydatabase.org/mdb-904/mdb-904-202406071531/mdb-904-202406071531.zip"
)
DUBAI_PULSE_TOKEN_URL = "https://api.dubaipulse.gov.ae/oauth/client_credential/accesstoken"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a local file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_retries(
    url: str, dest: Path, max_retries: int = 3, timeout: float = 120.0
) -> bool:
    """Stream a remote file to disk with exponential backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 512):
                        out.write(chunk)
            tmp.replace(dest)
            return True
        except Exception as exc:
            wait_s = 2**attempt
            logger.warning("Download attempt %d failed for %s: %s", attempt, url, exc)
            if tmp.exists():
                tmp.unlink()
            if attempt < max_retries:
                time.sleep(wait_s)
    return False


def _pull_dubai_pulse_api(api_key: str, api_secret: str, dataset_slug: str, dest: Path) -> bool:
    """Pull a dataset from Dubai Pulse Data API using OAuth client credentials (no secret logging)."""
    try:
        token_resp = httpx.post(
            DUBAI_PULSE_TOKEN_URL,
            params={"grant_type": "client_credentials"},
            data={"client_id": api_key, "client_secret": api_secret},
            timeout=20.0,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json().get("access_token")
        if not access_token:
            return False

        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        all_rows: list[dict[str, Any]] = []
        offset = 0
        limit = 1000
        base_url = f"https://api.dubaipulse.gov.ae/open/dld/{dataset_slug}"
        while True:
            for attempt in range(1, 4):
                resp = httpx.get(
                    base_url,
                    headers=headers,
                    params={"limit": limit, "offset": offset},
                    timeout=30.0,
                )
                if resp.status_code == 429:
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                rows = payload.get("results") or payload.get("data") or []
                if not isinstance(rows, list) or not rows:
                    if all_rows:
                        pd.DataFrame(all_rows).to_csv(dest, index=False)
                        return True
                    return False
                all_rows.extend(rows)
                if len(rows) < limit:
                    pd.DataFrame(all_rows).to_csv(dest, index=False)
                    return True
                offset += limit
                time.sleep(0.25)
                break
    except Exception as exc:
        logger.warning(
            "Dubai Pulse API pull unavailable for %s (%s)", dataset_slug, type(exc).__name__
        )
    return False


def _find_local_raw_file(raw_dir: Path, prefixes: tuple[str, ...]) -> Path | None:
    """Return the first matching CSV or Parquet file placed in data/raw/."""
    for ext in ("*.parquet", "*.csv"):
        for path in sorted(raw_dir.glob(ext)):
            stem = path.stem.lower()
            if any(stem.startswith(p) for p in prefixes):
                return path
    return None


def _read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _extract_gtfs_metro_stops(gtfs_zip: Path, reference_csv: Path) -> pd.DataFrame:
    """Extract Dubai Metro stations from GTFS zip, deduplicate platforms, and join opening dates."""
    import re

    ref_df = pd.read_csv(reference_csv) if reference_csv.exists() else pd.DataFrame()
    with zipfile.ZipFile(gtfs_zip, "r") as zf:
        with zf.open("stops.txt") as f:
            stops_df = pd.read_csv(io.BytesIO(f.read()))

    stops_adapted = adapt_dataframe(stops_df, STATIONS_COLUMN_MAP)
    metro_mask = stops_adapted["station_name"].astype(str).str.contains(
        "Metro Station", case=False, na=False
    ) & ~stops_adapted["station_name"].astype(str).str.contains("Bus", case=False, na=False)
    metro_raw = stops_adapted.loc[metro_mask, ["station_name", "lat", "lon"]].copy()
    if metro_raw.empty:
        metro_raw = stops_adapted[["station_name", "lat", "lon"]].copy()

    def _clean_station_label(raw: str) -> str:
        s = re.sub(r"\s+", " ", str(raw)).strip()
        s = re.sub(r"\s*Metro\s*Station(?:\s*[12])?$", "", s, flags=re.IGNORECASE).strip()
        aliases = {
            "deira city centre": "City Centre Deira",
            "burj khalifa/ dubai mall": "Burj Khalifa / Dubai Mall",
            "centrepoint": "Centrepoint",
            "max": "Max",
            "etisalat by e&": "e&",
        }
        low = s.lower()
        return aliases.get(low, s)

    metro_raw["station_name"] = metro_raw["station_name"].map(_clean_station_label)
    metro_stops = (
        metro_raw.groupby("station_name", as_index=False)
        .agg(lat=("lat", "mean"), lon=("lon", "mean"))
        .sort_values("station_name")
        .reset_index(drop=True)
    )

    if not ref_df.empty:
        ref_clean = ref_df.copy()
        ref_clean["key"] = ref_clean["station_name"].astype(str).str.lower().str.strip()
        metro_stops["key"] = metro_stops["station_name"].astype(str).str.lower().str.strip()
        merged = metro_stops.merge(
            ref_clean[["key", "line", "opening_date", "is_route_2020"]],
            on="key",
            how="left",
        ).drop(columns=["key"])
        merged["line"] = merged["line"].fillna("Red/Green")
        merged["is_route_2020"] = merged["is_route_2020"].fillna(False).astype(bool)
        merged["data_source"] = "unofficial_mobilitydatabase_gtfs"
        return merged.reset_index(drop=True)

    metro_stops["line"] = "Unknown"
    metro_stops["opening_date"] = None
    metro_stops["is_route_2020"] = False
    metro_stops["data_source"] = "unofficial_mobilitydatabase_gtfs"
    return metro_stops.reset_index(drop=True)


def build_sample_dataset(paths: Paths = PATHS, seed: int = 42) -> dict[str, Path]:
    """Create small committed sample dataset (< 5 MB) in data/sample/ for clean-clone CI/demo."""
    paths.sample.mkdir(parents=True, exist_ok=True)
    bundle = generate_synthetic_bundle(
        n_sales=3500, n_rents=4000, seed=seed, include_dirty_rows=False
    )
    out_paths = {
        "transactions": paths.sample / "transactions.parquet",
        "rent_contracts": paths.sample / "rent_contracts.parquet",
        "projects": paths.sample / "projects.parquet",
        "metro_stations": paths.sample / "metro_stations.parquet",
        "sale_index": paths.sample / "sale_index.parquet",
    }
    bundle.transactions.to_parquet(out_paths["transactions"], index=False)
    bundle.rent_contracts.to_parquet(out_paths["rent_contracts"], index=False)
    bundle.projects.to_parquet(out_paths["projects"], index=False)
    bundle.metro_stations.to_parquet(out_paths["metro_stations"], index=False)
    bundle.sale_index.to_parquet(out_paths["sale_index"], index=False)
    return out_paths


def ingest_data(
    paths: Paths = PATHS,
    allow_network: bool = True,
    allow_unofficial_fallback: bool = True,
    synthetic_only: bool = False,
) -> dict[str, Any]:
    """Run full acquisition & interim normalization, writing MANIFEST.json and data_dictionary.md."""
    ensure_dirs()
    manifest_entries: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    api_key = os.getenv("DUBAI_PULSE_API_KEY", "").strip()
    api_secret = os.getenv("DUBAI_PULSE_API_SECRET", "").strip()
    has_api_creds = bool(api_key and api_secret)

    # Always generate deterministic synthetic fallback tables as backup for missing/gated tables
    synth_bundle = generate_synthetic_bundle(
        n_sales=8000, n_rents=10000, seed=42, include_dirty_rows=False
    )

    # 1. Transactions
    tx_source_label = "synthetic_fallback"
    tx_df: pd.DataFrame | None = None
    if not synthetic_only:
        local_tx = _find_local_raw_file(paths.raw, ("dld_transactions", "transactions"))
        if local_tx is not None and local_tx.name != "Transactions_clean.csv":
            tx_df = _read_tabular(local_tx)
            tx_source_label = "official_dubaipulse_manual"
            manifest_entries.append(
                {
                    "dataset": "dld_transactions",
                    "file": local_tx.name,
                    "source_url": "https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open",
                    "source_type": tx_source_label,
                    "retrieved_at": now_iso,
                    "rows": int(len(tx_df)),
                    "size_bytes": local_tx.stat().st_size,
                    "sha256": compute_sha256(local_tx),
                }
            )
        elif has_api_creds and allow_network:
            api_dest = paths.raw / "dld_transactions-api.csv"
            if _pull_dubai_pulse_api(api_key, api_secret, "dld_transactions-open-api", api_dest):
                tx_df = _read_tabular(api_dest)
                tx_source_label = "official_dubaipulse_api"
                manifest_entries.append(
                    {
                        "dataset": "dld_transactions",
                        "file": api_dest.name,
                        "source_url": "https://api.dubaipulse.gov.ae/open/dld/dld_transactions-open-api",
                        "source_type": tx_source_label,
                        "retrieved_at": now_iso,
                        "rows": int(len(tx_df)),
                        "size_bytes": api_dest.stat().st_size,
                        "sha256": compute_sha256(api_dest),
                    }
                )

        if tx_df is None and allow_unofficial_fallback:
            hf_path = paths.raw / "Transactions_clean.csv"
            if not hf_path.exists() and allow_network:
                logger.info(
                    "Official dld_transactions missing; downloading unofficial fallback from Hugging Face (%s)",
                    HF_TRANSACTIONS_URL,
                )
                _download_with_retries(HF_TRANSACTIONS_URL, hf_path, timeout=300.0)
            if hf_path.exists():
                tx_df = _read_tabular(hf_path)
                tx_source_label = "unofficial_hf_fallback (viewit-ai/full-dubai-pulse)"
                manifest_entries.append(
                    {
                        "dataset": "dld_transactions_fallback",
                        "file": hf_path.name,
                        "source_url": HF_TRANSACTIONS_URL,
                        "source_type": tx_source_label,
                        "retrieved_at": now_iso,
                        "rows": int(len(tx_df)),
                        "size_bytes": hf_path.stat().st_size,
                        "sha256": compute_sha256(hf_path),
                    }
                )

    if tx_df is None:
        logger.warning("Using synthetic fallback for transactions (no raw file present).")
        tx_df = synth_bundle.transactions.copy()
    else:
        tx_df = adapt_dataframe(tx_df, TRANSACTIONS_COLUMN_MAP)
        tx_df["data_source"] = tx_source_label

    # 2. Rent contracts (Ejari)
    rent_source_label = "synthetic_fallback"
    rent_df: pd.DataFrame | None = None
    if not synthetic_only:
        local_rent = _find_local_raw_file(
            paths.raw, ("dld_rent_contracts", "rent_contracts", "ejari")
        )
        if local_rent is not None:
            rent_df = adapt_dataframe(_read_tabular(local_rent), RENTS_COLUMN_MAP)
            rent_source_label = "official_dubaipulse_manual"
            rent_df["data_source"] = rent_source_label
            manifest_entries.append(
                {
                    "dataset": "dld_rent_contracts",
                    "file": local_rent.name,
                    "source_url": "https://www.dubaipulse.gov.ae/data/dld-registration/dld_rent_contracts-open",
                    "source_type": rent_source_label,
                    "retrieved_at": now_iso,
                    "rows": int(len(rent_df)),
                    "size_bytes": local_rent.stat().st_size,
                    "sha256": compute_sha256(local_rent),
                }
            )
    if rent_df is None:
        logger.warning(
            "dld_rent_contracts-open not found in data/raw/; using synthetic fallback for rents."
        )
        rent_df = synth_bundle.rent_contracts.copy()
        rent_df["data_source"] = rent_source_label

    # 3. Projects
    proj_source_label = "synthetic_fallback"
    proj_df: pd.DataFrame | None = None
    if not synthetic_only:
        local_proj = _find_local_raw_file(paths.raw, ("dld_projects", "projects"))
        if local_proj is not None:
            proj_df = adapt_dataframe(_read_tabular(local_proj), PROJECTS_COLUMN_MAP)
            proj_source_label = "official_dubaipulse_manual"
            manifest_entries.append(
                {
                    "dataset": "dld_projects",
                    "file": local_proj.name,
                    "source_url": "https://www.dubaipulse.gov.ae/data/dld-registration/dld_projects-open",
                    "source_type": proj_source_label,
                    "retrieved_at": now_iso,
                    "rows": int(len(proj_df)),
                    "size_bytes": local_proj.stat().st_size,
                    "sha256": compute_sha256(local_proj),
                }
            )
    if proj_df is None:
        logger.warning(
            "dld_projects-open not found in data/raw/; using synthetic fallback for projects."
        )
        proj_df = synth_bundle.projects.copy()

    # 4. Metro stations (official manual -> MobilityDatabase GTFS -> curated/synthetic)
    metro_source_label = "curated_reference"
    metro_df: pd.DataFrame | None = None
    if not synthetic_only:
        local_metro = _find_local_raw_file(paths.raw, ("rta_metro_stations", "metro_stations"))
        if local_metro is not None:
            metro_df = adapt_dataframe(_read_tabular(local_metro), STATIONS_COLUMN_MAP)
            metro_source_label = "official_rta_dubaipulse_manual"
            manifest_entries.append(
                {
                    "dataset": "rta_metro_stations",
                    "file": local_metro.name,
                    "source_url": "https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_stations-open",
                    "source_type": metro_source_label,
                    "retrieved_at": now_iso,
                    "rows": int(len(metro_df)),
                    "size_bytes": local_metro.stat().st_size,
                    "sha256": compute_sha256(local_metro),
                }
            )
        elif allow_unofficial_fallback:
            gtfs_path = paths.raw / "rta_gtfs_mdb904.zip"
            if not gtfs_path.exists() and allow_network:
                logger.info("Downloading unofficial RTA GTFS fallback (%s)", MDB_GTFS_URL)
                _download_with_retries(MDB_GTFS_URL, gtfs_path, timeout=120.0)
            if gtfs_path.exists():
                metro_df = _extract_gtfs_metro_stops(
                    gtfs_path, paths.reference / "stations_opening.csv"
                )
                metro_source_label = "unofficial_mobilitydatabase_gtfs (mdb-904)"
                manifest_entries.append(
                    {
                        "dataset": "rta_gtfs_fallback",
                        "file": gtfs_path.name,
                        "source_url": MDB_GTFS_URL,
                        "source_type": metro_source_label,
                        "retrieved_at": now_iso,
                        "rows": int(len(metro_df)),
                        "size_bytes": gtfs_path.stat().st_size,
                        "sha256": compute_sha256(gtfs_path),
                    }
                )
    if metro_df is None:
        ref_stations = paths.reference / "stations_opening.csv"
        if ref_stations.exists():
            metro_df = pd.read_csv(ref_stations)
            metro_source_label = "curated_reference (stations_opening.csv)"
        else:
            metro_df = synth_bundle.metro_stations.copy()
            metro_source_label = "synthetic_fallback"

    # 5. Official sale price index
    index_source_label = "synthetic_fallback"
    index_df: pd.DataFrame | None = None
    if not synthetic_only:
        local_idx = _find_local_raw_file(
            paths.raw, ("dld_residential_sale_index", "sale_index", "residential_sale_index")
        )
        if local_idx is not None:
            index_df = adapt_dataframe(_read_tabular(local_idx), SALE_INDEX_COLUMN_MAP)
            index_source_label = "official_dubaipulse_manual"
            index_df["data_source"] = index_source_label
            manifest_entries.append(
                {
                    "dataset": "dld_residential_sale_index",
                    "file": local_idx.name,
                    "source_url": "https://www.dubaipulse.gov.ae/data/dld-transactions/dld_residential_sale_index-open",
                    "source_type": index_source_label,
                    "retrieved_at": now_iso,
                    "rows": int(len(index_df)),
                    "size_bytes": local_idx.stat().st_size,
                    "sha256": compute_sha256(local_idx),
                }
            )
    if index_df is None:
        logger.warning(
            "dld_residential_sale_index-open not found in data/raw/; using synthetic index fallback."
        )
        index_df = synth_bundle.sale_index.copy()

    # Save interim Parquet tables
    tx_df.to_parquet(paths.interim / "transactions_raw.parquet", index=False)
    rent_df.to_parquet(paths.interim / "rent_contracts_raw.parquet", index=False)
    proj_df.to_parquet(paths.interim / "projects_raw.parquet", index=False)
    metro_df.to_parquet(paths.interim / "metro_stations_raw.parquet", index=False)
    index_df.to_parquet(paths.interim / "sale_index_raw.parquet", index=False)

    # Always keep compact sample dataset in data/sample/
    build_sample_dataset(paths=paths)

    # Write MANIFEST.json
    manifest = {
        "generated_at": now_iso,
        "datasets": manifest_entries,
        "sources_summary": {
            "transactions": tx_source_label,
            "rent_contracts": rent_source_label,
            "projects": proj_source_label,
            "metro_stations": metro_source_label,
            "sale_index": index_source_label,
        },
    }
    manifest_path = paths.raw / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate docs/data_dictionary.md from actual ingested schemas
    write_data_dictionary(
        tables={
            "transactions_raw": (tx_df, tx_source_label),
            "rent_contracts_raw": (rent_df, rent_source_label),
            "projects_raw": (proj_df, proj_source_label),
            "metro_stations_raw": (metro_df, metro_source_label),
            "sale_index_raw": (index_df, index_source_label),
        },
        output_path=paths.docs / "data_dictionary.md",
    )

    logger.info("Ingestion complete. Summary: %s", manifest["sources_summary"])
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire and normalize DubaiFairValue datasets.")
    parser.add_argument("--offline", action="store_true", help="Do not make network downloads.")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip raw/network files and build synthetic interim + sample tables.",
    )
    args = parser.parse_args(argv)
    ingest_data(
        paths=PATHS,
        allow_network=not args.offline,
        synthetic_only=args.synthetic_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
