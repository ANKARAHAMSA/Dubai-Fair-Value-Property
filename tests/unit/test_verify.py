"""Unit tests for verification report helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfv.config import PATHS, Paths
from dfv.verify import (
    _is_processed_mode,
    _manifest_hash,
    _summarize_junit_suite,
    _tool_cmd,
    check_secrets_and_gitignore,
    measure_api_predict_latency_ms,
    parse_pytest_summary,
    run_verification,
)
from dfv.verify import (
    main as verify_main,
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


def test_junit_parser_mode_flags_and_sample_non_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _tmp_paths(tmp_path)
    for d in (
        p.raw,
        p.reports,
        p.notebooks,
        p.sample_models,
        p.processed,
        p.models,
        p.root / "src" / "dfv" / "data",
        p.root / "src" / "dfv" / "api",
        p.root / "src" / "dfv" / "app",
    ):
        d.mkdir(parents=True, exist_ok=True)
    (p.root / ".gitignore").write_text(".env\n", encoding="utf-8")
    (p.root / "src" / "dfv" / "data" / "validate.py").write_text("# ok\n", encoding="utf-8")
    (p.root / "src" / "dfv" / "api" / "main.py").write_text("# ok\n", encoding="utf-8")
    (p.root / "src" / "dfv" / "app" / "Home.py").write_text("# ok\n", encoding="utf-8")
    (p.notebooks / "01_eda.ipynb").write_text("{}", encoding="utf-8")

    # Processed metrics fixture that MUST NOT be mutated when DFV_DATA_MODE=sample
    processed_metrics = {
        "gates": {"all_passed": True},
        "test_metrics": {
            "model_mdape": 0.068,
            "best_baseline_mdape": 0.095,
            "interval_coverage": 0.901,
        },
        "api_predict_latency_ms": 99.9,
    }
    proc_metrics_text = json.dumps(processed_metrics, indent=2)
    (p.reports / "metrics.json").write_text(proc_metrics_text, encoding="utf-8")
    (p.sample_models / "metadata.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "model_mdape": 0.08,
                    "best_baseline_mdape": 0.11,
                    "interval_coverage": 0.91,
                    "repro_max_log_diff": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("DFV_DATA_MODE", "sample")
    assert _is_processed_mode(p) is False
    monkeypatch.setenv("DFV_DATA_MODE", "processed")
    assert _is_processed_mode(p) is False  # processed files missing in tmp
    monkeypatch.setenv("DFV_DATA_MODE", "sample")

    invocations: list[list[str]] = []

    def _fake_run(cmd: list[str], _cwd: Path) -> tuple[int, str]:
        invocations.append(cmd)
        for arg in cmd:
            if arg.startswith("--junitxml="):
                xml_out = Path(arg.split("=", 1)[1])
                xml_out.write_text(
                    """<testsuite>
                        <testcase classname="tests.data_contracts.test_contracts" name="t1"/>
                        <testcase classname="tests.api.test_api" name="t2"/>
                        <testcase classname="tests.app.test_streamlit_app" name="t3"/>
                        <testcase classname="tests.integration.test_notebooks" name="t4"/>
                    </testsuite>""",
                    encoding="utf-8",
                )
        if any("pytest" in c for c in cmd):
            return 0, "10 passed in 0.50s\nTOTAL 100 15 85%\n"
        return 0, "ok"

    monkeypatch.setattr("dfv.verify.run_command", _fake_run)
    monkeypatch.setattr("dfv.verify.measure_api_predict_latency_ms", lambda _paths: 14.5)

    custom_report = tmp_path / "custom_report.md"
    rc = run_verification(report_only=False, paths=p, report_path=custom_report)
    assert rc == 0
    assert custom_report.exists()
    rep_txt = custom_report.read_text(encoding="utf-8")
    assert "`sample`" in rep_txt
    assert "`14.5`" in rep_txt
    # Single-pass pytest must NOT re-run sub-suites separately
    pytest_calls = [c for c in invocations if any("pytest" in part for part in c)]
    assert len(pytest_calls) == 1
    # Processed reports/metrics.json was NOT overwritten in sample mode
    assert (p.reports / "metrics.json").read_text(encoding="utf-8") == proc_metrics_text

    # Tool command & JUnit edge cases + CLI entrypoint
    assert _tool_cmd("pytest", "-q")
    assert _summarize_junit_suite(tmp_path / "missing.xml", "tests.api.") is None
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<not-xml", encoding="utf-8")
    assert _summarize_junit_suite(bad_xml, "tests.api.") is None
    out_cli = tmp_path / "cli_rep.md"
    assert verify_main(["--report-only", "--output-report", str(out_cli)]) in (0, 1)
    assert out_cli.exists()


def test_measure_api_predict_latency_ms_sample() -> None:
    if (PATHS.sample_models / "metadata.json").exists():
        lat = measure_api_predict_latency_ms(PATHS)
        assert lat is None or (0.0 < lat < 5000.0)
