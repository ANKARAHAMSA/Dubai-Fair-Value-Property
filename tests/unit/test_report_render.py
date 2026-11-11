"""Unit tests verifying README auto-rendering stays synced with JSON metrics."""

from __future__ import annotations

import json
from pathlib import Path

from dfv.config import PATHS
from dfv.report_render import build_readme_markdown


def test_readme_matches_rendered_output_and_metrics() -> None:
    """Confirm README.md on disk matches build_readme_markdown(PATHS) and pulls exact metrics."""
    rendered = build_readme_markdown(PATHS)
    readme_on_disk = (PATHS.root / "README.md").read_text(encoding="utf-8")
    assert readme_on_disk == rendered
    metrics = json.loads((PATHS.reports / "metrics.json").read_text(encoding="utf-8"))
    tm = metrics["test_metrics"]

    assert f"{tm['model_mdape'] * 100:.2f}%" in rendered
    assert f"{tm['best_baseline_mdape'] * 100:.2f}%" in rendered
    assert f"{tm['interval_coverage'] * 100:.2f}%" in rendered
    assert f"{tm['spatial_mdape'] * 100:.2f}%" in rendered
    assert "hero_banner.jpg" in rendered
    assert "Engineering Build Phases (Phases 0" in rendered
    assert "What the Model Gets Wrong" in rendered
    assert "make setup && make data && make app" in rendered
    assert "Not Financial or Appraisal Advice" in rendered


def test_readme_degrades_cleanly_without_reports(tmp_path: Path) -> None:
    """Confirm build_readme_markdown degrades cleanly even with empty reports directory."""
    from dfv.config import Paths

    empty_paths = Paths(
        root=tmp_path,
        data=tmp_path / "data",
        raw=tmp_path / "data" / "raw",
        interim=tmp_path / "data" / "interim",
        processed=tmp_path / "data" / "processed",
        sample=tmp_path / "data" / "sample",
        reference=tmp_path / "data" / "reference",
        cache=tmp_path / "data" / "cache",
        docs=tmp_path / "docs",
        reports=tmp_path / "reports",
        figures=tmp_path / "reports" / "figures",
        notebooks=tmp_path / "notebooks",
        models=tmp_path / "data" / "processed" / "models",
        sample_models=tmp_path / "data" / "sample" / "models",
    )
    fallback_md = build_readme_markdown(empty_paths)
    assert "# DubaiFairValue" in fallback_md
    assert "n/a" in fallback_md
