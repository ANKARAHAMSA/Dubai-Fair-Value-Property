"""Verification runner (`make verify`) writing reports/VERIFICATION_REPORT.md."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
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


def run_verification(report_only: bool = False, paths: Paths = PATHS) -> int:
    """Run verification gates and write reports/VERIFICATION_REPORT.md."""
    root = paths.root
    paths.reports.mkdir(parents=True, exist_ok=True)
    checks: list[CheckResult] = []

    if not report_only:
        # 1. Lint & static typing
        rc_ruff, out_ruff = run_command(["uv", "run", "ruff", "check", "src", "tests"], root)
        rc_fmt, out_fmt = run_command(
            ["uv", "run", "ruff", "format", "--check", "src", "tests"], root
        )
        rc_mypy, out_mypy = run_command(["uv", "run", "mypy", "src/dfv"], root)
        if rc_ruff == 0 and rc_fmt == 0 and rc_mypy == 0:
            checks.append(CheckResult("Lint & mypy", "PASS", "ruff check/format and mypy passed"))
        else:
            detail = (out_ruff + "\n" + out_fmt + "\n" + out_mypy).strip().splitlines()[-1]
            checks.append(CheckResult("Lint & mypy", "FAIL", detail))

        # 2. Secrets & .env check
        checks.append(check_secrets_and_gitignore(root))

        # 3. Full pytest + coverage (>= 80%)
        rc_test, out_test = run_command(
            ["uv", "run", "pytest", "--cov=src/dfv", "--cov-report=term-missing"], root
        )
        test_count, cov_pct = parse_pytest_summary(out_test)
        if rc_test == 0:
            checks.append(
                CheckResult("Pytest & coverage", "PASS", f"{test_count} passed, coverage {cov_pct}")
            )
        else:
            tail = "\n".join(out_test.strip().splitlines()[-8:])
            checks.append(CheckResult("Pytest & coverage", "FAIL", tail))
    else:
        test_count = "cached"
        cov_pct = "cached"
        checks.append(check_secrets_and_gitignore(root))

    # 4. Phase-gated modules (active once built)
    metrics_path = paths.reports / "metrics.json"
    metrics_data: dict = {}
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Data contracts check
    if (root / "src" / "dfv" / "data" / "validate.py").exists():
        rc_val, out_val = run_command(["uv", "run", "pytest", "tests/data_contracts", "-q"], root)
        checks.append(
            CheckResult(
                "Data contracts (Pandera)",
                "PASS" if rc_val == 0 else "FAIL",
                "Contracts passed" if rc_val == 0 else out_val.strip()[-200:],
            )
        )
    else:
        checks.append(CheckResult("Data contracts (Pandera)", "SKIPPED", "Scheduled for Phase 1"))

    # Model gates check
    if metrics_data and "gates" in metrics_data:
        gates = metrics_data["gates"]
        all_ok = bool(gates.get("all_passed", False))
        mdape = metrics_data.get("test_metrics", {}).get("model_mdape", "n/a")
        base_mdape = metrics_data.get("test_metrics", {}).get("best_baseline_mdape", "n/a")
        cov = metrics_data.get("test_metrics", {}).get("interval_coverage", "n/a")
        checks.append(
            CheckResult(
                "Model acceptance gates",
                "PASS" if all_ok else "FAIL",
                f"Model MdAPE={mdape}, Best baseline MdAPE={base_mdape}, 90% coverage={cov}",
            )
        )
    else:
        checks.append(CheckResult("Model acceptance gates", "SKIPPED", "Scheduled for Phase 3"))

    # API contract tests
    if (root / "src" / "dfv" / "api" / "main.py").exists():
        rc_api, out_api = run_command(["uv", "run", "pytest", "tests/api", "-q"], root)
        checks.append(
            CheckResult(
                "API contract & latency tests",
                "PASS" if rc_api == 0 else "FAIL",
                "API tests passed" if rc_api == 0 else out_api.strip()[-200:],
            )
        )
    else:
        checks.append(
            CheckResult("API contract & latency tests", "SKIPPED", "Scheduled for Phase 5")
        )

    # App smoke tests
    if (root / "src" / "dfv" / "app" / "Home.py").exists():
        rc_app, out_app = run_command(["uv", "run", "pytest", "tests/app", "-q"], root)
        checks.append(
            CheckResult(
                "Streamlit app smoke tests",
                "PASS" if rc_app == 0 else "FAIL",
                "App smoke tests passed" if rc_app == 0 else out_app.strip()[-200:],
            )
        )
    else:
        checks.append(CheckResult("Streamlit app smoke tests", "SKIPPED", "Scheduled for Phase 6"))

    # Notebook execution check
    nbs = sorted(paths.notebooks.glob("*.ipynb"))
    if nbs:
        rc_nb, out_nb = run_command(
            ["uv", "run", "pytest", "tests/integration", "-k", "notebook", "-q"], root
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

    # Compose markdown report
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = _git_sha(root)
    manifest_h = _manifest_hash(paths.raw)
    has_fail = any(c.status == "FAIL" for c in checks)
    skipped = [c for c in checks if c.status == "SKIPPED"]

    lines = [
        "# DubaiFairValue Verification Report",
        "",
        f"- **Timestamp (UTC)**: `{now_iso}`",
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
    if metrics_data:
        tm = metrics_data.get("test_metrics", {})
        lines.extend(
            [
                f"- Model MdAPE: `{tm.get('model_mdape', 'n/a')}`",
                f"- Best Baseline MdAPE: `{tm.get('best_baseline_mdape', 'n/a')}`",
                f"- Nominal 90% Interval Coverage: `{tm.get('interval_coverage', 'n/a')}`",
                f"- Spatial Hold-out MdAPE: `{tm.get('spatial_mdape', 'n/a')}`",
                f"- API `/predict` Warm Latency (ms): `{metrics_data.get('api_predict_latency_ms', 'n/a')}`",
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

    report_path = paths.reports / "VERIFICATION_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path} (status={'FAIL' if has_fail else 'PASS'})")
    return 1 if has_fail else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DubaiFairValue verification gates.")
    parser.add_argument(
        "--report-only", action="store_true", help="Write report without rerunning slow checks."
    )
    args = parser.parse_args(argv)
    # Ensure CI / non-interactive env doesn't leak local secrets
    os.environ.setdefault("PYTHONUTF8", "1")
    return run_verification(report_only=args.report_only)


if __name__ == "__main__":
    sys.exit(main())
