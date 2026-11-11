"""Unit tests for verification report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from dfv.config import Paths
from dfv.verify import (
    _manifest_hash,
    check_secrets_and_gitignore,
    parse_pytest_summary,
    run_verification,
)


def _tmp_paths(tmp_path: Path) -> Paths:
    return Paths(
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


def test_parse_pytest_summary() -> None:
    sample = "12 passed in 1.23s\nTOTAL 100 10 90%\n"
    cnt, cov = parse_pytest_summary(sample)
    assert cnt == "12"
    assert cov == "90%"


def test_check_secrets_detects_clean_repo(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / "safe.py").write_text("x = 1\n", encoding="utf-8")
    res = check_secrets_and_gitignore(tmp_path)
    assert res.status == "PASS"


def test_check_secrets_flags_secret_pattern(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    fake_secret = "sk-ant-" + "secretvalue123456"
    (tmp_path / "bad.py").write_text(f'API_KEY = "{fake_secret}"\n', encoding="utf-8")
    res = check_secrets_and_gitignore(tmp_path)
    assert res.status == "FAIL"


def test_check_secrets_flags_missing_env_ignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    res = check_secrets_and_gitignore(tmp_path)
    assert res.status == "FAIL"


def test_manifest_hash_and_report_only(tmp_path: Path) -> None:
    p = _tmp_paths(tmp_path)
    p.raw.mkdir(parents=True)
    p.reports.mkdir(parents=True)
    p.notebooks.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")

    assert "synthetic" in _manifest_hash(p.raw)
    (p.raw / "MANIFEST.json").write_text('{"files": []}', encoding="utf-8")
    assert len(_manifest_hash(p.raw)) == 16

    (p.reports / "metrics.json").write_text(
        json.dumps(
            {
                "gates": {"all_passed": True},
                "test_metrics": {
                    "model_mdape": 0.06,
                    "best_baseline_mdape": 0.12,
                    "interval_coverage": 0.90,
                },
            }
        ),
        encoding="utf-8",
    )

    rc = run_verification(report_only=True, paths=p)
    assert rc == 0
    report = (p.reports / "VERIFICATION_REPORT.md").read_text(encoding="utf-8")
    assert "Overall Status" in report
    assert "**PASS**" in report
