"""Model bundle persistence and strict metadata verification (`registry.py`)."""

from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "model_version",
    "feature_list",
    "categorical_features",
    "training_window",
    "git_sha",
    "metrics",
    "data_manifest_hash",
)


class ModelMetadataError(RuntimeError):
    """Raised when a model bundle is missing required provenance metadata."""


@dataclass
class ModelBundle:
    """Serialized model artifacts + strict metadata needed for inference."""

    point_model: Any
    interval_model: Any
    oof_mappings: dict[str, dict[str, float]]
    global_prior_log: float
    area_defaults: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def current_git_sha(root: Path) -> str:
    """Return short git SHA or 'uncommitted'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def save_model_bundle(bundle: ModelBundle, target_dir: Path) -> Path:
    """Save pickled models and validated JSON metadata to target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    validate_metadata(bundle.metadata)

    with (target_dir / "model_bundle.pkl").open("wb") as f:
        pickle.dump(
            {
                "point_model": bundle.point_model,
                "interval_model": bundle.interval_model,
                "oof_mappings": bundle.oof_mappings,
                "global_prior_log": bundle.global_prior_log,
                "area_defaults": bundle.area_defaults,
            },
            f,
        )

    meta_path = target_dir / "metadata.json"
    meta_path.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")
    return target_dir


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Raise ModelMetadataError if any mandatory provenance field is missing or empty."""
    missing = [
        k for k in REQUIRED_METADATA_FIELDS if k not in metadata or metadata[k] in (None, "", [])
    ]
    if missing:
        raise ModelMetadataError(f"Model metadata missing required fields: {missing}")


def load_model_bundle(target_dir: Path) -> ModelBundle:
    """Load model bundle and enforce metadata presence; API refuses to start otherwise."""
    meta_path = target_dir / "metadata.json"
    pkl_path = target_dir / "model_bundle.pkl"
    if not meta_path.exists():
        raise ModelMetadataError(f"Missing model metadata file: {meta_path}")
    if not pkl_path.exists():
        raise ModelMetadataError(f"Missing model artifact file: {pkl_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)

    with pkl_path.open("rb") as f:
        payload = pickle.load(f)

    return ModelBundle(
        point_model=payload["point_model"],
        interval_model=payload["interval_model"],
        oof_mappings=payload["oof_mappings"],
        global_prior_log=float(payload["global_prior_log"]),
        area_defaults=payload["area_defaults"],
        metadata=metadata,
    )


def prepare_inference_frame(
    df_row: pd.DataFrame,
    bundle: ModelBundle,
) -> pd.DataFrame:
    """Apply saved OOF encodings and categorical dtypes to match training feature order."""
    work = df_row.copy()
    for col, mapping in bundle.oof_mappings.items():
        enc_col = f"{col}_oof_enc"
        if col in work.columns:
            work[enc_col] = work[col].astype(str).map(mapping).fillna(bundle.global_prior_log)
        else:
            work[enc_col] = bundle.global_prior_log

    cat_cols = set(bundle.metadata["categorical_features"])
    feature_list = list(bundle.metadata["feature_list"])
    for col in feature_list:
        if col not in work.columns:
            work[col] = "Unknown" if col in cat_cols else 0.0
        if col in cat_cols:
            work[col] = work[col].astype("category")
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return work[feature_list]


def build_metadata(
    model_version: str,
    feature_list: list[str],
    categorical_features: list[str],
    training_window: dict[str, str],
    git_sha: str,
    metrics: dict[str, Any],
    data_manifest_hash: str,
    dataset_mode: str,
) -> dict[str, Any]:
    """Construct validated metadata dictionary."""
    return {
        "model_version": model_version,
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_list": feature_list,
        "categorical_features": categorical_features,
        "training_window": training_window,
        "git_sha": git_sha,
        "metrics": metrics,
        "data_manifest_hash": data_manifest_hash,
        "dataset_mode": dataset_mode,
    }
