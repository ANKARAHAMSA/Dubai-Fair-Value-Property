"""Integration test: execute every notebook top-to-bottom cleanly via nbconvert."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dfv.config import PATHS

NOTEBOOKS = sorted(PATHS.notebooks.glob("*.ipynb"))


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_top_to_bottom(nb_path: Path, tmp_path: Path) -> None:
    """Execute each notebook in sample mode to confirm zero runtime errors."""
    out_nb = tmp_path / nb_path.name
    env = os.environ.copy()
    env["DFV_DATA_MODE"] = "sample"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=120",
            "--output",
            str(out_nb),
            str(nb_path),
        ],
        cwd=PATHS.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Notebook {nb_path.name} failed:\n{proc.stdout}"
    assert out_nb.exists() and out_nb.stat().st_size > 500
