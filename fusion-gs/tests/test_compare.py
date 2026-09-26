"""Fast checks for the comparison report.

The full report (129-grid PINN figures, surrogate q profiles, timings) is what
``python -m eval.compare`` runs. These tests only check that the summary names
every model whose metrics are on disk, and that the Solov'ev finite-difference
error drops when the mesh is refined.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.compare import overall_table, solver_verification

ROOT = Path(__file__).resolve().parents[1]


def _metrics_ready() -> bool:
    needed = [
        ROOT / "outputs" / "pinn" / "solovev_metrics.json",
        ROOT / "outputs" / "pinn" / "nonlinear_metrics.json",
        ROOT / "outputs" / "pinn" / "parametric_metrics.json",
        ROOT / "outputs" / "surrogate" / "summary.json",
    ]
    return all(path.is_file() for path in needed)


def test_overall_table_has_a_row_per_model(tmp_path):
    if not (ROOT / "outputs").is_dir() or not _metrics_ready():
        pytest.skip("outputs/ missing")
    # Timing is covered by eval.compare.speed_table. Skipping it here keeps
    # the test on the metrics files only.
    text = overall_table(out_dir=tmp_path, with_timing=False)
    assert isinstance(text, str)
    for token in (
        "pinn-solovev",
        "pinn-nonlinear",
        "pinn-parametric",
        "mlpcnn_pw0",
        "mlpcnn_pw0p1",
        "fno_pw0",
        "fno_pw0p1",
    ):
        assert token in text, token
    inverse_ready = (ROOT / "outputs" / "inverse" / "metrics.json").is_file() and (
        ROOT / "outputs" / "inverse" / "inverse.pt"
    ).is_file()
    if inverse_ready:
        assert "inverse-fixed" in text
    assert (tmp_path / "summary.md").is_file()


def test_solver_verification_error_decreases(tmp_path):
    result = solver_verification(grids=(17, 33), out_dir=tmp_path)
    errors = result["rel_l2"]
    assert len(errors) == 2
    assert errors[1] < errors[0]
    assert result["observed_order"][1] > 1.5
    assert (tmp_path / "solver_convergence.png").is_file()
