"""Quarterly geospatial price-growth timelapse dataset and animated GIF/PNG export."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def build_quarterly_growth_table(sales_features_df: pd.DataFrame, min_tx: int = 3) -> pd.DataFrame:
    """Compute quarterly area price/sqm, volume, cumulative growth vs earliest base, and YoY growth."""
    work = sales_features_df.copy()
    if "flag_outlier" in work.columns:
        work = work.loc[
            ~work["flag_outlier"] & ~work["flag_non_market"] & ~work["flag_duplicate"]
        ].copy()

    work["dt"] = pd.to_datetime(work["instance_date"])
    work["year_quarter"] = work["dt"].dt.to_period("Q").astype(str)
    work["quarter_start"] = work["dt"].dt.to_period("Q").dt.start_time

    agg = work.groupby(["year_quarter", "quarter_start", "area_name"], as_index=False).agg(
        median_ppsqm=("price_per_sqm", "median"),
        tx_count=("price_per_sqm", "count"),
        lat=("lat", "median"),
        lon=("lon", "median"),
    )
    agg = (
        agg.loc[agg["tx_count"] >= min_tx]
        .sort_values(["area_name", "quarter_start"])
        .reset_index(drop=True)
    )

    # YoY and Cumulative growth relative to area's earliest observed quarter (or 2019Q1 baseline if present)
    agg["cumulative_growth_pct"] = np.nan
    agg["yoy_growth_pct"] = np.nan
    for _area, grp in agg.groupby("area_name"):
        idx = grp.index
        base_2019 = grp.loc[grp["year_quarter"].str.startswith("2019"), "median_ppsqm"]
        base_val = (
            float(base_2019.mean()) if not base_2019.empty else float(grp["median_ppsqm"].iloc[0])
        )
        agg.loc[idx, "cumulative_growth_pct"] = np.round(
            100.0 * (grp["median_ppsqm"] / max(base_val, 1.0) - 1.0), 1
        )
        yoy = grp["median_ppsqm"].pct_change(periods=4) * 100.0
        agg.loc[idx, "yoy_growth_pct"] = np.round(yoy, 1)

    agg["quarter_date"] = agg["quarter_start"].dt.strftime("%Y-%m-%d")
    agg["median_ppsqm"] = np.round(agg["median_ppsqm"], 1)
    return (
        agg.drop(columns=["quarter_start"])
        .sort_values(["quarter_date", "area_name"])
        .reset_index(drop=True)
    )


def render_growth_timelapse(
    quarterly_df: pd.DataFrame,
    figures_dir: Path,
    gif_name: str = "price_growth_timelapse.gif",
    snapshots_name: str = "price_growth_snapshots.png",
) -> tuple[Path, Path]:
    """Export animated quarterly GIF and 4-panel comparison PNG from Dubai area bubbles."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    gif_path = figures_dir / gif_name
    snap_path = figures_dir / snapshots_name

    if quarterly_df.empty:
        return gif_path, snap_path

    # Focus on top 25 most liquid communities cleanly visible on the Dubai metro/coastal canvas
    top_areas = (
        quarterly_df.groupby("area_name")["tx_count"]
        .sum()
        .sort_values(ascending=False)
        .head(25)
        .index.tolist()
    )
    sub = quarterly_df.loc[quarterly_df["area_name"].isin(top_areas)].copy()
    sub = sub.loc[sub["lat"].between(24.80, 25.32) & sub["lon"].between(55.02, 55.45)].copy()

    quarters = sorted(sub["year_quarter"].unique())
    vmin = float(sub["median_ppsqm"].quantile(0.05))
    vmax = float(sub["median_ppsqm"].quantile(0.95))

    frames: list[Image.Image] = []
    # Cap GIF frame count to ~40 evenly spaced quarters so file stays lightweight for Git/LinkedIn
    step = max(1, len(quarters) // 36)
    selected_q = quarters[::step]
    if quarters[-1] not in selected_q:
        selected_q.append(quarters[-1])

    lon_min, lon_max = float(sub["lon"].min() - 0.03), float(sub["lon"].max() + 0.03)
    lat_min, lat_max = float(sub["lat"].min() - 0.03), float(sub["lat"].max() + 0.03)

    for q_str in selected_q:
        q_df = sub.loc[sub["year_quarter"] == q_str]
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=110)
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")
        sc = ax.scatter(
            q_df["lon"],
            q_df["lat"],
            s=np.clip(q_df["tx_count"] * 1.4, 55, 650),
            c=q_df["median_ppsqm"],
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            alpha=0.88,
            edgecolors="#F8FAFC",
            linewidth=0.9,
        )
        for _, r in q_df.sort_values("tx_count", ascending=False).head(6).iterrows():
            ax.annotate(
                f"{r['area_name']}\n{r['median_ppsqm']:,.0f}",
                (float(r["lon"]), float(r["lat"])),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color="#F8FAFC",
            )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude", color="#CBD5E1", fontsize=8)
        ax.set_ylabel("Latitude", color="#CBD5E1", fontsize=8)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Median AED / sqm", color="#E2E8F0", fontsize=8)
        cbar.ax.tick_params(colors="#CBD5E1", labelsize=7)
        mkt_med = float(q_df["median_ppsqm"].median()) if not q_df.empty else 0.0
        ax.set_title(
            f"Dubai Residential Price Evolution — {q_str} (Median: AED {mkt_med:,.0f}/sqm)",
            color="#F8FAFC",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=380,
            loop=0,
            optimize=True,
        )

    # Multi-panel static snapshots (4 key quarters)
    snap_idx = np.linspace(0, len(quarters) - 1, min(4, len(quarters)), dtype=int)
    snap_quarters = [quarters[i] for i in snap_idx]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=130)
    fig.patch.set_facecolor("#0F172A")
    for ax_i, q_str in zip(axes.flat, snap_quarters, strict=False):
        q_df = sub.loc[sub["year_quarter"] == q_str]
        ax_i.set_facecolor("#1E293B")
        ax_i.scatter(
            q_df["lon"],
            q_df["lat"],
            s=np.clip(q_df["tx_count"] * 1.1, 40, 420),
            c=q_df["median_ppsqm"],
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            alpha=0.9,
            edgecolors="#F8FAFC",
            linewidth=0.7,
        )
        ax_i.set_xlim(lon_min, lon_max)
        ax_i.set_ylim(lat_min, lat_max)
        ax_i.tick_params(colors="#94A3B8", labelsize=7)
        mkt_med = float(q_df["median_ppsqm"].median()) if not q_df.empty else 0.0
        ax_i.set_title(f"{q_str} (AED {mkt_med:,.0f}/sqm)", color="#F8FAFC", fontsize=10)
    fig.suptitle(
        "Dubai Price Growth Snapshots (Area Centroids)", color="#F8FAFC", fontsize=13, y=0.98
    )
    plt.tight_layout()
    fig.savefig(snap_path, dpi=140, facecolor="#0F172A")
    plt.close(fig)

    return gif_path, snap_path
