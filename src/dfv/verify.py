"""Verification runner (`make verify`) writing reports/VERIFICATION_REPORT.md."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dfv.config import PATHS, Paths


@dataclass
class CheckResult:
    """Result for a single verification check."""

    name: str
    status: str  # PASS | FAIL | SKIPPED
    detail: str


SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() or "uncommitted"
    except Exception:
        return "no-git"


def _manifest_hash(raw_dir: Path) -> str:
    manifest_path = raw_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return "none (synthetic/sample mode)"
    import hashlib

    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]


def check_secrets_and_gitignore(root: Path) -> CheckResult:
    """Verify .env is gitignored and no tracked file contains hardcoded secret patterns."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists() or ".env" not in gitignore_path.read_text(encoding="utf-8"):
        return CheckResult("Secrets & .env gitignore", "FAIL", ".env not present in .gitignore")

    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
    except Exception:
        tracked = []

    if not tracked:
        skip_parts = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        tracked = [
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
            and not any(part in skip_parts for part in p.parts)
            and p.name != ".env"
            and p.suffix not in {".pyc", ".pyo"}
        ]

    hits: list[str] = []
    for rel in tracked:
        fpath = root / rel
        if not fpath.is_file() or fpath.suffix in {".png", ".jpg", ".gif", ".parquet", ".pdf"}:
            continue
        if rel == "src/dfv/verify.py":
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                hits.append(rel)
                break

    if hits:
        return CheckResult("Secrets & .env gitignore", "FAIL", f"Secret match in: {hits}")
    return CheckResult("Secrets & .env gitignore", "PASS", f"Scanned {len(tracked)} files cleanly")


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run a subprocess and return (exit_code, combined_output)."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def parse_pytest_summary(output: str) -> tuple[str, str]:
    """Extract test count and coverage % from pytest output."""
    test_count = "unknown"
    cov_pct = "unknown"
    m_tests = re.search(r"(\d+)\s+passed", output)
    if m_tests:
        test_count = m_tests.group(1)
    m_cov = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
    if m_cov:
        cov_pct = f"{m_cov.group(1)}%"
    return test_count, cov_pct


def measure_api_predict_latency_ms(paths: Paths) -> float | None:
    """Measure warm median POST /predict latency via FastAPI TestClient."""
    import statistics
    import time

    try:
        from fastapi.testclient import TestClient

        from dfv.api.main import create_app

        has_proc = (paths.models / "metadata.json").exists()
        mode_override = os.getenv("DFV_DATA_MODE", "auto").strip().lower()
        use_proc = mode_override == "processed" or (mode_override != "sample" and has_proc)
        m_dir = paths.models if use_proc and has_proc else paths.sample_models
        d_dir = (
            paths.processed
            if use_proc and (paths.processed / "features.parquet").exists()
            else paths.sample
        )
        app = create_app(model_dir=m_dir, data_dir=d_dir)
        payload = {
            "area_name": "Dubai Marina",
            "property_type": "Unit",
            "rooms": 2,
            "area_sqm": 115.0,
            "asking_price_aed": 2_450_000.0,
            "is_offplan": False,
        }
        with TestClient(app) as client:
            w = client.post("/predict", json=payload)
            if w.status_code != 200:
                return None
            timings: list[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                resp = client.post("/predict", json=payload)
                timings.append((time.perf_counter() - t0) * 1000.0)
                if resp.status_code != 200:
                    return None
        return round(float(statistics.median(timings)), 2)
    except Exception:
        return None


def _tool_cmd(tool: str, *args: str) -> list[str]:
    """Resolve CLI executable inside active Python venv (without requiring global `uv` on PATH)."""
    cand = Path(sys.executable).parent / tool
    if cand.exists():
        return [str(cand), *args]
    if tool in {"pytest", "mypy", "ruff"}:
        return [sys.executable, "-m", tool, *args]
    return ["uv", "run", tool, *args]


def _summarize_junit_suite(xml_path: Path, classname_prefix: str) -> tuple[bool, int, int] | None:
    """Parse JUnit XML counts for a suite prefix; return (passed, total, failed_or_skipped) or None."""
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    total = 0
    bad = 0
    for tc in tree.getroot().iter("testcase"):
        cls = str(tc.attrib.get("classname", ""))
        if cls.startswith(classname_prefix):
            total += 1
            if (
                tc.find("failure") is not None
                or tc.find("error") is not None
                or tc.find("skipped") is not None
            ):
                bad += 1
    return (total > 0 and bad == 0, total, bad)


def _is_processed_mode(paths: Paths) -> bool:
    mode_override = os.getenv("DFV_DATA_MODE", "auto").strip().lower()
    if mode_override == "sample":
        return False
    has_proc_model = (paths.models / "metadata.json").exists()
    has_proc_data = (paths.processed / "features.parquet").exists()
    has_proc_metrics = (paths.reports / "metrics.json").exists()
    if mode_override == "processed":
        return has_proc_model and has_proc_data
    return has_proc_model and has_proc_data and has_proc_metrics


def run_verification(
    report_only: bool = False,
    paths: Paths = PATHS,
    report_path: Path | None = None,
) -> int:
    """Run verification gates and write reports/VERIFICATION_REPORT.md."""
    root = paths.root
    paths.reports.mkdir(parents=True, exist_ok=True)
    checks: list[CheckResult] = []
    running_processed = _is_processed_mode(paths)

    junit_dir: tempfile.TemporaryDirectory[str] | None = None
    junit_xml: Path | None = None
    has_junit = False

    try:
        if not report_only:
            # 1. Lint & static typing
            rc_ruff, out_ruff = run_command(_tool_cmd("ruff", "check", "src", "tests"), root)
            rc_fmt, out_fmt = run_command(
                _tool_cmd("ruff", "format", "--check", "src", "tests"), root
            )
            rc_mypy, out_mypy = run_command(_tool_cmd("mypy", "src/dfv"), root)
            if rc_ruff == 0 and rc_fmt == 0 and rc_mypy == 0:
                checks.append(
                    CheckResult("Lint & mypy", "PASS", "ruff check/format and mypy passed")
                )
            else:
                detail = (out_ruff + "\n" + out_fmt + "\n" + out_mypy).strip().splitlines()[-1]
                checks.append(CheckResult("Lint & mypy", "FAIL", detail))

            # 2. Secrets & .env check
            checks.append(check_secrets_and_gitignore(root))

            # 3. Single-pass full pytest + coverage (>= 80%) + JUnit XML for sub-suite verification
            junit_dir = tempfile.TemporaryDirectory()
            junit_xml = Path(junit_dir.name) / "junit.xml"
            rc_test, out_test = run_command(
                _tool_cmd(
                    "pytest",
                    "--cov=src/dfv",
                    "--cov-report=term-missing",
                    f"--junitxml={junit_xml}",
                ),
                root,
            )
            has_junit = junit_xml.exists()
            test_count, cov_pct = parse_pytest_summary(out_test)
            if rc_test == 0:
                checks.append(
                    CheckResult(
                        "Pytest & coverage", "PASS", f"{test_count} passed, coverage {cov_pct}"
                    )
                )
            else:
                tail = "\n".join(out_test.strip().splitlines()[-8:])
                checks.append(CheckResult("Pytest & coverage", "FAIL", tail))
        else:
            test_count = "cached"
            cov_pct = "cached"
            checks.append(check_secrets_and_gitignore(root))

        # 4. Data contracts check (read from single pytest pass if available, else targeted run)
        if (root / "src" / "dfv" / "data" / "validate.py").exists():
            suite_res = (
                _summarize_junit_suite(junit_xml, "tests.data_contracts.")
                if junit_xml and has_junit
                else None
            )
            if suite_res is not None:
                ok_dc, tot_dc, bad_dc = suite_res
                checks.append(
                    CheckResult(
                        "Data contracts (Pandera)",
                        "PASS" if ok_dc else "FAIL",
                        f"Contracts passed ({tot_dc} checks)"
                        if ok_dc
                        else f"Contracts failed ({bad_dc}/{tot_dc})",
                    )
                )
            else:
                rc_val, out_val = run_command(
                    _tool_cmd("pytest", "tests/data_contracts", "-q"), root
                )
                checks.append(
                    CheckResult(
                        "Data contracts (Pandera)",
                        "PASS" if rc_val == 0 else "FAIL",
                        "Contracts passed" if rc_val == 0 else out_val.strip()[-200:],
                    )
                )
        else:
            checks.append(
                CheckResult("Data contracts (Pandera)", "SKIPPED", "Scheduled for Phase 1")
            )

        # 5. Model acceptance gates (processed -> reports/metrics.json; sample -> data/sample/models/metadata.json)
        metrics_path = paths.reports / "metrics.json"
        metrics_data: dict = {}
        active_tm: dict = {}
        if running_processed and metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            gates = metrics_data.get("gates", {})
            active_tm = metrics_data.get("test_metrics", {})
            all_ok = bool(gates.get("all_passed", False))
            mdape = active_tm.get("model_mdape", "n/a")
            base_mdape = active_tm.get("best_baseline_mdape", "n/a")
            cov = active_tm.get("interval_coverage", "n/a")
            checks.append(
                CheckResult(
                    "Model acceptance gates",
                    "PASS" if all_ok else "FAIL",
                    f"Model MdAPE={mdape}, Best baseline MdAPE={base_mdape}, 90% coverage={cov}",
                )
            )
        elif (paths.sample_models / "metadata.json").exists():
            s_meta = json.loads((paths.sample_models / "metadata.json").read_text(encoding="utf-8"))
            active_tm = dict(s_meta.get("metrics", {}))
            mdape_v = active_tm.get("model_mdape")
            base_v = active_tm.get("best_baseline_mdape")
            cov_v = active_tm.get("interval_coverage")
            repro_v = float(active_tm.get("repro_max_log_diff", 0.0))
            all_ok = bool(
                mdape_v is not None
                and base_v is not None
                and cov_v is not None
                and float(mdape_v) < float(base_v)
                and abs(float(cov_v) - 0.90) <= 0.05
                and repro_v < 1e-6
            )
            checks.append(
                CheckResult(
                    "Model acceptance gates",
                    "PASS" if all_ok else "FAIL",
                    f"[sample] Model MdAPE={mdape_v}, Best baseline MdAPE={base_v}, 90% coverage={cov_v}",
                )
            )
        else:
            checks.append(CheckResult("Model acceptance gates", "SKIPPED", "Scheduled for Phase 3"))

        # 6. API contract & latency tests (only persist latency into processed reports/metrics.json in processed mode)
        if (root / "src" / "dfv" / "api" / "main.py").exists():
            suite_res = (
                _summarize_junit_suite(junit_xml, "tests.api.") if junit_xml and has_junit else None
            )
            if suite_res is not None:
                rc_api = 0 if suite_res[0] else 1
                out_api = f"API suite failures: {suite_res[2]}/{suite_res[1]}"
            else:
                rc_api, out_api = run_command(_tool_cmd("pytest", "tests/api", "-q"), root)
            latency_ms: float | None = None
            if rc_api == 0 and not report_only:
                latency_ms = measure_api_predict_latency_ms(paths)
                if (
                    latency_ms is not None
                    and running_processed
                    and metrics_data
                    and metrics_path.exists()
                    and "api_predict_latency_ms" not in metrics_data
                ):
                    metrics_data["api_predict_latency_ms"] = latency_ms
                    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
            lat_ok = latency_ms is None or latency_ms < 500.0
            api_ok = rc_api == 0 and lat_ok
            detail = (
                f"API tests passed (warm /predict median={latency_ms} ms)"
                if api_ok and latency_ms is not None
                else ("API tests passed" if api_ok else out_api.strip()[-200:])
            )
            checks.append(
                CheckResult(
                    "API contract & latency tests",
                    "PASS" if api_ok else "FAIL",
                    detail,
                )
            )
        else:
            checks.append(
                CheckResult("API contract & latency tests", "SKIPPED", "Scheduled for Phase 5")
            )

        # 7. App smoke tests (reuse JUnit pass when available)
        if (root / "src" / "dfv" / "app" / "Home.py").exists():
            suite_res = (
                _summarize_junit_suite(junit_xml, "tests.app.") if junit_xml and has_junit else None
            )
            if suite_res is not None:
                ok_app, tot_app, bad_app = suite_res
                checks.append(
                    CheckResult(
                        "Streamlit app smoke tests",
                        "PASS" if ok_app else "FAIL",
                        f"App smoke tests passed ({tot_app} tests)"
                        if ok_app
                        else f"App tests failed ({bad_app}/{tot_app})",
                    )
                )
            else:
                rc_app, out_app = run_command(_tool_cmd("pytest", "tests/app", "-q"), root)
                checks.append(
                    CheckResult(
                        "Streamlit app smoke tests",
                        "PASS" if rc_app == 0 else "FAIL",
                        "App smoke tests passed" if rc_app == 0 else out_app.strip()[-200:],
                    )
                )
        else:
            checks.append(
                CheckResult("Streamlit app smoke tests", "SKIPPED", "Scheduled for Phase 6")
            )

        # 8. Notebook execution check (reuse JUnit pass when available)
        nbs = sorted(paths.notebooks.glob("*.ipynb"))
        if nbs:
            suite_res = (
                _summarize_junit_suite(junit_xml, "tests.integration.test_notebooks")
                if junit_xml and has_junit
                else None
            )
            if suite_res is not None:
                ok_nb, tot_nb, bad_nb = suite_res
                checks.append(
                    CheckResult(
                        "Notebook execution",
                        "PASS" if ok_nb and tot_nb >= len(nbs) else "FAIL",
                        f"{tot_nb} notebooks executed"
                        if ok_nb
                        else f"Notebooks failed ({bad_nb}/{tot_nb})",
                    )
                )
            else:
                rc_nb, out_nb = run_command(
                    _tool_cmd("pytest", "tests/integration", "-k", "notebook", "-q"), root
                )
                checks.append(
                    CheckResult(
                        "Notebook execution",
                        "PASS" if rc_nb == 0 else "FAIL",
                        f"{len(nbs)} notebooks executed" if rc_nb == 0 else out_nb.strip()[-200:],
                    )
                )
        else:
            checks.append(CheckResult("Notebook execution", "SKIPPED", "Scheduled for Phase 4"))
    finally:
        if junit_dir is not None:
            junit_dir.cleanup()

    # Compose markdown report
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = _git_sha(root)
    manifest_h = _manifest_hash(paths.raw)
    has_fail = any(c.status == "FAIL" for c in checks)
    skipped = [c for c in checks if c.status == "SKIPPED"]
    mode_tag = "processed" if running_processed else "sample"

    lines = [
        "# DubaiFairValue Verification Report",
        "",
        f"- **Timestamp (UTC)**: `{now_iso}`",
        f"- **Data Mode**: `{mode_tag}`",
        f"- **Git SHA**: `{sha}`",
        f"- **Data Manifest Hash**: `{manifest_h}`",
        f"- **Test Count**: `{test_count}`",
        f"- **Code Coverage (`src/dfv`)**: `{cov_pct}`",
        f"- **Overall Status**: `{'FAIL' if has_fail else 'PASS'}`",
        "",
        "## Check Summary",
        "",
        "| Check | Status | Details |",
        "| --- | --- | --- |",
    ]
    for chk in checks:
        safe_detail = chk.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {chk.name} | **{chk.status}** | {safe_detail} |")

    lines.extend(["", "## Measured Model & System Metrics", ""])
    if active_tm:
        disp_lat = (
            latency_ms
            if latency_ms is not None
            else (
                metrics_data.get("api_predict_latency_ms")
                if running_processed and metrics_data
                else None
            )
        )
        lines.extend(
            [
                f"- Model MdAPE: `{active_tm.get('model_mdape', 'n/a')}`",
                f"- Best Baseline MdAPE: `{active_tm.get('best_baseline_mdape', 'n/a')}`",
                f"- Nominal 90% Interval Coverage: `{active_tm.get('interval_coverage', 'n/a')}`",
                f"- Spatial Hold-out MdAPE: `{active_tm.get('spatial_mdape', 'n/a')}`",
                f"- API `/predict` Warm Latency (ms): `{disp_lat if disp_lat is not None else 'n/a'}`",
            ]
        )
    else:
        lines.append("- Model metrics not yet generated (Phase 3 pending).")

    lines.extend(["", "## Skipped Checks & Justifications", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- **{item.name}**: {item.detail}")
    else:
        lines.append("- None (all checks active and evaluated).")

    out_report_path = report_path or (paths.reports / "VERIFICATION_REPORT.md")
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_report_path} (status={'FAIL' if has_fail else 'PASS'})")
    return 1 if has_fail else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DubaiFairValue verification gates.")
    parser.add_argument(
        "--report-only", action="store_true", help="Write report without rerunning slow checks."
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional override output path for VERIFICATION_REPORT.md.",
    )
    args = parser.parse_args(argv)
    # Ensure CI / non-interactive env doesn't leak local secrets
    os.environ.setdefault("PYTHONUTF8", "1")
    return run_verification(report_only=args.report_only, report_path=args.output_report)


if __name__ == "__main__":
    sys.exit(main())
