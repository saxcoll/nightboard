"""Side-by-side comparison of the Grad-Shafranov solver, PINNs, surrogates, and inverse model.

Run from the project root::

    python -m eval.compare

Figures and tables are written to ``outputs/compare/``. Checkpoints under
``outputs/`` are loaded; nothing is retrained. The inverse section is included
only when ``models.inverse`` imports and both ``outputs/inverse/inverse.pt`` and
``outputs/inverse/metrics.json`` exist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

torch.set_num_threads(2)

# ITER-like Cerfon-Freidberg equilibrium used for the solver study and the Solov'ev PINN.
SOLOVEV_EPS = 0.32
SOLOVEV_KAPPA = 1.7
SOLOVEV_DELTA = 0.33
SOLOVEV_A = -0.155
SOLOVEV_R0 = 1.0
SOLOVEV_BOX = (0.45, 1.55, -0.85, 0.85)

SURROGATE_RUNS = ("mlpcnn_pw0", "mlpcnn_pw0p1", "fno_pw0", "fno_pw0p1")
PINN_CASES = ("solovev", "nonlinear", "parametric")


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _out(out_dir: str | Path | None = None) -> Path:
    path = Path(out_dir) if out_dir is not None else _root() / "outputs" / "compare"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _threads() -> None:
    torch.set_num_threads(2)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body]) + "\n"


def _fmt(value: Any, digits: str = ".4e") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return format(number, digits)


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _hypot(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(np.hypot(float(a), float(b)))


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _inverse_paths() -> dict[str, Path] | None:
    """Checkpoint and metrics for the fixed-boundary inverse model, if both exist."""
    try:
        import models.inverse  # noqa: F401
    except Exception:
        return None
    root = _root() / "outputs" / "inverse"
    checkpoint = root / "inverse.pt"
    metrics = root / "metrics.json"
    if not checkpoint.is_file() or not metrics.is_file():
        return None
    info = {"root": root, "checkpoint": checkpoint, "metrics": metrics}
    freegs_metrics = root / "freegs" / "metrics.json"
    freegs_checkpoint = root / "freegs" / "inverse.pt"
    if freegs_metrics.is_file():
        info["freegs_metrics"] = freegs_metrics
    if freegs_checkpoint.is_file():
        info["freegs_checkpoint"] = freegs_checkpoint
    return info


def _try_load_inverse(path: Path):
    """Load a checkpoint, or return None if it does not match the current module."""
    from models.inverse import load_inverse

    try:
        return load_inverse(path)
    except Exception as exc:
        print(f"inverse checkpoint {path} did not load ({exc.__class__.__name__})", flush=True)
        return None


def _rel_l2_masked(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    from eval.metrics import relative_l2

    return float(relative_l2(pred, ref, mask))


def _flux_levels(psi_axis: float, n: int = 4) -> np.ndarray:
    return np.sort(np.linspace(0.2, 0.8, n) * float(psi_axis))


def _zoom(mask: np.ndarray, R: np.ndarray, Z: np.ndarray, pad: int = 2) -> tuple[float, float, float, float]:
    ii, jj = np.nonzero(mask)
    i0 = max(int(ii.min()) - pad, 0)
    i1 = min(int(ii.max()) + pad, R.size - 1)
    j0 = max(int(jj.min()) - pad, 0)
    j1 = min(int(jj.max()) + pad, Z.size - 1)
    return float(R[i0]), float(R[i1]), float(Z[j0]), float(Z[j1])


def _overlay_and_error(ax_c, ax_e, R, Z, ref, pred, mask, levels, title: str) -> None:
    """Reference flux as a background, true and predicted contours, and the error map."""
    from matplotlib.lines import Line2D

    ref_m = np.where(mask, ref, np.nan)
    err = np.where(mask, pred - ref, np.nan)
    finite = err[np.isfinite(err)]
    lim = float(np.max(np.abs(finite))) if finite.size else 1e-6
    if lim == 0.0:
        lim = 1e-6
    mesh = ax_c.pcolormesh(R, Z, ref_m.T, shading="auto", cmap="viridis")
    ax_c.contour(R, Z, np.where(mask, ref, np.nan).T, levels=levels, colors="k", linewidths=1.15)
    ax_c.contour(
        R, Z, np.where(mask, pred, np.nan).T, levels=levels, colors="#d62728", linewidths=1.15, linestyles="dashed"
    )
    ax_c.set_aspect("equal")
    ax_c.set_xlim(*_zoom(mask, R, Z)[:2])
    ax_c.set_ylim(*_zoom(mask, R, Z)[2:])
    ax_c.set_xlabel("R [m]")
    ax_c.set_ylabel("Z [m]")
    ax_c.set_title(title)
    ax_c.legend(
        handles=[
            Line2D([0], [0], color="k", lw=1.2, label="true"),
            Line2D([0], [0], color="#d62728", lw=1.2, ls="--", label="predicted"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    ax_c.figure.colorbar(mesh, ax=ax_c, fraction=0.046, pad=0.04, label="psi")
    err_mesh = ax_e.pcolormesh(R, Z, err.T, shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax_e.set_aspect("equal")
    ax_e.set_xlim(*_zoom(mask, R, Z)[:2])
    ax_e.set_ylim(*_zoom(mask, R, Z)[2:])
    ax_e.set_xlabel("R [m]")
    ax_e.set_ylabel("Z [m]")
    ax_e.set_title("predicted − true")
    ax_e.figure.colorbar(err_mesh, ax=ax_e, fraction=0.046, pad=0.04, label="Δpsi")


def solver_verification(
    grids: Sequence[int] = (33, 65, 129),
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Finite-difference error against the Cerfon-Freidberg Solov'ev solution.

    ``grids`` is the list of ``n`` for uniform ``n × n`` meshes on the fixed
    evaluation box. The scheme is second order, so the log-log plot of relative
    L2 versus spacing includes a slope-2 reference through the finest point.
    """
    from gs.analytic import solovev
    from gs.solver import Grid, solve_linear

    out = _out(out_dir)
    eq = solovev(SOLOVEV_EPS, SOLOVEV_KAPPA, SOLOVEV_DELTA, SOLOVEV_A, R0=SOLOVEV_R0)
    ns = [int(n) for n in grids]
    if len(ns) < 1:
        raise ValueError("grids must contain at least one resolution")
    rels: list[float] = []
    hs: list[float] = []
    seconds: list[float] = []
    for n in ns:
        grid = Grid.uniform(*SOLOVEV_BOX, n, n)
        rhs = eq.gs_rhs(grid.RR, grid.ZZ)
        import time

        t0 = time.perf_counter()
        psi = solve_linear(grid, rhs, level_set=eq.level_set, boundary_value=0.0)
        seconds.append(time.perf_counter() - t0)
        mask = np.asarray(eq.level_set(grid.RR, grid.ZZ), dtype=float) < 0.0
        exact = np.asarray(eq.psi(grid.RR, grid.ZZ), dtype=float)
        rels.append(_rel_l2_masked(psi, exact, mask))
        hs.append(float(max(grid.dR, grid.dZ)))

    orders: list[float | None] = [None]
    for i in range(1, len(ns)):
        orders.append(float(np.log(rels[i - 1] / rels[i]) / np.log(hs[i - 1] / hs[i])))

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    ax.loglog(hs, rels, "o-", color="#1f77b4", lw=1.6, ms=7, label="FD relative L2")
    if len(hs) >= 2:
        href = np.array([hs[0], hs[-1]], dtype=float)
        eref = rels[-1] * (href / hs[-1]) ** 2
        ax.loglog(href, eref, "--", color="0.35", lw=1.2, label="slope 2")
    for n, h, err in zip(ns, hs, rels):
        ax.annotate(str(n), (h, err), textcoords="offset points", xytext=(6, 4), fontsize=8)
    last = orders[-1]
    order_txt = f"last order {last:.2f}" if last is not None else "Solov'ev FD"
    ax.set_xlabel("h = max(ΔR, ΔZ) [m]")
    ax.set_ylabel("relative L2 inside the plasma")
    ax.set_title(
        f"Solov'ev FD convergence, {order_txt}\n"
        f"eps={SOLOVEV_EPS}, kappa={SOLOVEV_KAPPA}, delta={SOLOVEV_DELTA}, A={SOLOVEV_A}"
    )
    ax.legend(frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    figure = out / "solver_convergence.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)

    payload = {
        "epsilon": SOLOVEV_EPS,
        "kappa": SOLOVEV_KAPPA,
        "delta": SOLOVEV_DELTA,
        "A": SOLOVEV_A,
        "R0": SOLOVEV_R0,
        "box": list(SOLOVEV_BOX),
        "n": ns,
        "h": hs,
        "rel_l2": rels,
        "observed_order": orders,
        "solve_seconds": seconds,
        "figure": str(figure),
    }
    _write_json(out / "solver_verification.json", payload)
    print(
        "solver verification: "
        + ", ".join(f"n={n} relL2={err:.3e}" for n, err in zip(ns, rels)),
        flush=True,
    )
    return payload


def _pinn_reference(case: str, n: int, cond: float | None = None):
    """Analytic or finite-difference reference field on an n×n grid."""
    from gs.analytic import solovev
    from gs.solver import Grid, ProfileParams, ShapeParams, solve_fixed_boundary
    from models.pinn import NONLINEAR_EVAL_BOX, NONLINEAR_PROFILE, NONLINEAR_SHAPE, SOLOVEV_SPEC

    if case == "nonlinear":
        grid = Grid.uniform(*NONLINEAR_EVAL_BOX, n, n)
        shape = ShapeParams(**NONLINEAR_SHAPE)
        profile = ProfileParams(**NONLINEAR_PROFILE)
        eq = solve_fixed_boundary(shape, profile, grid)
        return {
            "grid": grid,
            "ref": np.asarray(eq.psi, dtype=float),
            "mask": np.asarray(eq.mask, dtype=bool),
            "psi_axis": float(eq.psi_axis),
            "R_axis": float(eq.R_axis),
            "Z_axis": float(eq.Z_axis),
            "cond": None,
            "label": "nonlinear vs FD",
        }
    spec = dict(SOLOVEV_SPEC)
    if case == "parametric":
        if cond is None:
            raise ValueError("parametric reference needs the conditioning value A")
        spec["A"] = float(cond)
    eq = solovev(**spec)
    grid = Grid.uniform(*SOLOVEV_BOX, n, n)
    ref = np.asarray(eq.psi(grid.RR, grid.ZZ), dtype=float)
    mask = np.asarray(eq.level_set(grid.RR, grid.ZZ), dtype=float) < 0.0
    label = "Solov'ev vs analytic"
    if case == "parametric":
        label = f"parametric vs analytic (unseen A={float(spec['A']):.3f})"
    return {
        "grid": grid,
        "ref": ref,
        "mask": mask,
        "psi_axis": float(np.min(ref[mask])),
        "cond": None if case == "solovev" else float(spec["A"]),
        "label": label,
    }


def pinn_summary(out_dir: str | Path | None = None, n: int = 129) -> dict[str, Any]:
    """Load PINN checkpoints and compare Solov'ev / parametric to analytic and nonlinear to FD."""
    from eval.metrics import find_magnetic_axis
    from models.pinn import load_pinn, predict_grid

    _threads()
    out = _out(out_dir)
    root = _root() / "outputs" / "pinn"
    metrics = {}
    for case in PINN_CASES:
        path = root / f"{case}_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics[case] = _read_json(path)

    figure_A = float(metrics["parametric"].get("figure_A", -0.05))
    panels = []
    axis_errors = {}
    for case in PINN_CASES:
        model = load_pinn(root / f"{case}.pt")
        ref = _pinn_reference(case, n, cond=figure_A if case == "parametric" else None)
        pred = predict_grid(model, ref["grid"], cond=ref["cond"])
        rel = _rel_l2_masked(pred, ref["ref"], ref["mask"])
        r_t, z_t, _ = find_magnetic_axis(ref["ref"], ref["grid"], ref["mask"])
        r_p, z_p, _ = find_magnetic_axis(pred, ref["grid"], ref["mask"])
        axis_errors[case] = float(np.hypot(r_p - r_t, z_p - z_t))
        panels.append(
            {
                "case": case,
                "pred": pred,
                "ref": ref,
                "rel_l2_figure": rel,
                "checkpoint_rel_l2": float(metrics[case]["rel_l2"]),
            }
        )

    # Mean axis error of the parametric network over every unseen A in the metrics file.
    from models.pinn import PARAM_UNSEEN_A

    param_model = load_pinn(root / "parametric.pt")
    param_axes = []
    for A in PARAM_UNSEEN_A:
        ref = _pinn_reference("parametric", n, cond=float(A))
        pred = predict_grid(param_model, ref["grid"], cond=float(A))
        r_t, z_t, _ = find_magnetic_axis(ref["ref"], ref["grid"], ref["mask"])
        r_p, z_p, _ = find_magnetic_axis(pred, ref["grid"], ref["mask"])
        param_axes.append(float(np.hypot(r_p - r_t, z_p - z_t)))
    parametric_axis_mean = float(np.mean(param_axes))

    plt = _import_pyplot()
    fig, axes = plt.subplots(len(panels), 2, figsize=(10.4, 3.6 * len(panels)), constrained_layout=True)
    for row, panel in enumerate(panels):
        ref = panel["ref"]
        grid = ref["grid"]
        title = (
            f"{panel['case']}: {ref['label']}\n"
            f"checkpoint rel L2 {panel['checkpoint_rel_l2']:.3e} "
            f"(this {n}×{n} map {panel['rel_l2_figure']:.3e})"
        )
        _overlay_and_error(
            axes[row, 0],
            axes[row, 1],
            grid.R,
            grid.Z,
            ref["ref"],
            panel["pred"],
            ref["mask"],
            _flux_levels(ref["psi_axis"]),
            title,
        )
    fig.suptitle("PINN poloidal flux", fontsize=13)
    figure = out / "pinn_comparison.png"
    fig.savefig(figure, dpi=130)
    plt.close(fig)

    # Official nonlinear axis error is stored as separate R and Z components.
    nl = metrics["nonlinear"]
    checkpoint_axis = {
        "solovev": axis_errors["solovev"],
        "parametric": parametric_axis_mean,
        "nonlinear": _hypot(nl.get("R_axis_error"), nl.get("Z_axis_error")),
    }
    payload = {
        "metrics": {case: metrics[case] for case in PINN_CASES},
        "figure_rel_l2": {p["case"]: p["rel_l2_figure"] for p in panels},
        "figure_axis_error_m": axis_errors,
        "checkpoint_axis_error_m": checkpoint_axis,
        "grid_n": int(n),
        "figure": str(figure),
    }
    # Metrics blobs are large enough already; keep the sidecar to the comparison fields.
    _write_json(
        out / "pinn_summary.json",
        {
            "figure_rel_l2": payload["figure_rel_l2"],
            "figure_axis_error_m": axis_errors,
            "checkpoint_axis_error_m": checkpoint_axis,
            "checkpoint_rel_l2": {case: float(metrics[case]["rel_l2"]) for case in PINN_CASES},
            "checkpoint_rel_l2_mean": {
                "parametric": metrics["parametric"].get("rel_l2_mean"),
            },
            "grid_n": int(n),
            "figure": str(figure),
        },
    )
    print(
        "PINN: "
        + ", ".join(f"{case} relL2={metrics[case]['rel_l2']:.3e}" for case in PINN_CASES),
        flush=True,
    )
    return payload


def _surrogate_summary_doc() -> dict:
    path = _root() / "outputs" / "surrogate" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_json(path)


def _best_surrogate(summary: dict) -> str:
    runs = summary["runs"]
    scored = []
    for name in SURROGATE_RUNS:
        if name not in runs:
            continue
        rel = runs[name]["evaluation"]["test"]["rel_l2"]["mean"]
        scored.append((float(rel), name))
    if not scored:
        raise RuntimeError("surrogate summary has none of the expected runs")
    scored.sort()
    return scored[0][1]


def _order_params(params: np.ndarray, names) -> np.ndarray:
    from models.surrogate import PARAM_NAMES

    names = [str(x) for x in list(names)]
    index = [names.index(n) for n in PARAM_NAMES]
    return np.ascontiguousarray(np.asarray(params)[:, index])


def _load_split(path: Path, split_value: int | None):
    from data.generate import load_dataset

    data = load_dataset(str(path))
    if split_value is None:
        idx = np.arange(data["psi"].shape[0])
    else:
        idx = np.flatnonzero(np.asarray(data["split"]) == split_value)
    return data, idx


def surrogate_summary(out_dir: str | Path | None = None) -> dict[str, Any]:
    """Metrics table for the four surrogates, plus flux and q comparisons for the best one."""
    from eval.metrics import find_magnetic_axis, q_profile, relative_l2
    from gs.solver import Grid, ProfileParams, ShapeParams, solve_fixed_boundary
    from models.surrogate import load_surrogate, predict

    _threads()
    out = _out(out_dir)
    summary = _surrogate_summary_doc()
    best = _best_surrogate(summary)
    headers = [
        "run",
        "test rel L2",
        "OOD rel L2",
        "test GS residual",
        "OOD GS residual",
        "test axis error [m]",
        "OOD axis error [m]",
        "test q95 rel",
        "OOD q95 rel",
    ]
    rows = []
    extracted = {}
    for name in SURROGATE_RUNS:
        block = summary["runs"][name]["evaluation"]
        test, ood = block["test"], block["ood"]
        extracted[name] = {
            "test_rel_l2": test["rel_l2"]["mean"],
            "ood_rel_l2": ood["rel_l2"]["mean"],
            "test_gs": test["gs_residual"]["mean"],
            "ood_gs": ood["gs_residual"]["mean"],
            "test_axis_m": test["axis_error_m"]["mean"],
            "ood_axis_m": ood["axis_error_m"]["mean"],
            "test_q95_rel": test["q95_rel_error"]["mean"],
            "ood_q95_rel": ood["q95_rel_error"]["mean"],
            "test_q95_abs": test["q95_abs_error"]["mean"],
            "ood_q95_abs": ood["q95_abs_error"]["mean"],
            "test_infer_single_s": (test.get("timing") or {}).get("infer_single_s"),
            "test_infer_batch_s": (test.get("timing") or {}).get("infer_batch_s"),
        }
        rows.append(
            [
                name,
                _fmt(extracted[name]["test_rel_l2"]),
                _fmt(extracted[name]["ood_rel_l2"]),
                _fmt(extracted[name]["test_gs"]),
                _fmt(extracted[name]["ood_gs"]),
                _fmt(extracted[name]["test_axis_m"], ".3e"),
                _fmt(extracted[name]["ood_axis_m"], ".3e"),
                _fmt(extracted[name]["test_q95_rel"], ".3e"),
                _fmt(extracted[name]["ood_q95_rel"], ".3e"),
            ]
        )
    table = _markdown_table(headers, rows)
    (out / "surrogate_table.md").write_text(table)

    root = _root()
    model = load_surrogate(root / "outputs" / "surrogate" / best / "checkpoint.pt")
    test_data, test_idx = _load_split(root / "data" / "fixed_65.npz", 2)
    ood_data, ood_idx = _load_split(root / "data" / "fixed_65_ood.npz", None)
    test_params = _order_params(test_data["params"][test_idx], test_data["param_names"]).astype(np.float32)
    ood_params = _order_params(ood_data["params"][ood_idx], ood_data["param_names"]).astype(np.float32)
    test_pred = predict(model, test_params)
    ood_pred = predict(model, ood_params)

    def _scores(pred, data, idx):
        mask = np.asarray(data["mask"][idx], dtype=bool)
        truth = np.asarray(data["psi"][idx], dtype=np.float64)
        rel = np.array([relative_l2(pred[i], truth[i], mask[i]) for i in range(pred.shape[0])])
        return rel

    test_rel = _scores(test_pred, test_data, test_idx)
    ood_rel = _scores(ood_pred, ood_data, ood_idx)

    def _pick(rel: np.ndarray, quantiles: Sequence[float]) -> list[int]:
        order = np.argsort(rel)
        picks = []
        for q in quantiles:
            j = int(np.clip(round(q * (order.size - 1)), 0, order.size - 1))
            choice = int(order[j])
            if choice not in picks:
                picks.append(choice)
        return picks

    test_picks = _pick(test_rel, (0.5, 0.9))
    ood_picks = _pick(ood_rel, (0.5, 0.9))
    samples = []
    for local in test_picks:
        samples.append(("test", int(test_idx[local]), local, test_data, test_pred, float(test_rel[local])))
    for local in ood_picks:
        samples.append(("OOD", int(ood_idx[local]), local, ood_data, ood_pred, float(ood_rel[local])))

    plt = _import_pyplot()
    fig, axes = plt.subplots(len(samples), 2, figsize=(10.2, 3.35 * len(samples)), constrained_layout=True)
    for row, (split_name, global_i, local, data, pred, rel) in enumerate(samples):
        R = np.asarray(data["R"], dtype=float)
        Z = np.asarray(data["Z"], dtype=float)
        truth = np.asarray(data["psi"][global_i], dtype=float)
        mask = np.asarray(data["mask"][global_i], dtype=bool)
        psi_axis = float(data["psi_axis"][global_i])
        _overlay_and_error(
            axes[row, 0],
            axes[row, 1],
            R,
            Z,
            truth,
            np.asarray(pred[local], dtype=float),
            mask,
            _flux_levels(psi_axis),
            f"{best}  {split_name} #{global_i}   rel L2 {rel:.3e}",
        )
    fig.suptitle(f"Best surrogate ({best}): flux surfaces", fontsize=13)
    figure_samples = out / "surrogate_samples.png"
    fig.savefig(figure_samples, dpi=120)
    plt.close(fig)

    # q(psi_n) on three in-distribution test samples, using F from a fresh fixed-boundary solve.
    # Quartile samples are preferred; if a contour fails, the next test row is used.
    order = [int(i) for i in np.argsort(test_rel)]
    preferred = _pick(test_rel, (0.25, 0.5, 0.75))
    queue = preferred + [i for i in order if i not in preferred]
    psin = np.linspace(0.05, 0.95, 19)
    R = np.asarray(test_data["R"], dtype=float)
    Z = np.asarray(test_data["Z"], dtype=float)
    grid = Grid(R, Z)
    q_records = []
    fig_q, axes_q = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    ax_i = 0
    for local in queue:
        if ax_i >= 3:
            break
        global_i = int(test_idx[local])
        row = test_params[local].astype(float)
        truth = np.asarray(test_data["psi"][global_i], dtype=float)
        mask = np.asarray(test_data["mask"][global_i], dtype=bool)
        pred_i = np.asarray(test_pred[local], dtype=float)
        try:
            solved = solve_fixed_boundary(
                ShapeParams(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                ProfileParams(float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])),
                grid,
            )
            psi_axis = float(test_data["psi_axis"][global_i])
            r_axis = float(test_data["R_axis"][global_i])
            z_axis = float(test_data["Z_axis"][global_i])
            q_true = q_profile(truth, grid, psi_axis, 0.0, r_axis, z_axis, solved.F, psin)
            r_hat, z_hat, psi_hat = find_magnetic_axis(pred_i, grid, mask)
            q_pred = q_profile(pred_i, grid, float(psi_hat), 0.0, float(r_hat), float(z_hat), solved.F, psin)
        except (ValueError, RuntimeError) as exc:
            print(f"q profile skipped test #{global_i}: {exc}", flush=True)
            continue
        q95_abs = float(abs(q_pred[-1] - q_true[-1]))
        q_records.append(
            {
                "dataset_index": global_i,
                "rel_l2": float(test_rel[local]),
                "psin": psin.tolist(),
                "q_true": np.asarray(q_true, dtype=float).tolist(),
                "q_pred": np.asarray(q_pred, dtype=float).tolist(),
                "q95_abs_error": q95_abs,
                "q95_true": float(q_true[-1]),
                "q95_pred": float(q_pred[-1]),
            }
        )
        ax = axes_q[ax_i]
        ax.plot(psin, q_true, color="k", lw=1.6, label="true psi")
        ax.plot(psin, q_pred, color="#d62728", lw=1.6, ls="--", label="predicted psi")
        ax.set_xlabel("psi_n")
        ax.set_ylabel("q")
        ax.set_title(f"test #{global_i}\n|Δq95|={q95_abs:.3f}")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=8)
        ax_i += 1
    if ax_i < 3:
        plt.close(fig_q)
        raise RuntimeError(f"q profile succeeded on only {ax_i} test samples")
    fig_q.suptitle(f"{best}: safety factor with the true F(psi_n)", fontsize=12)
    figure_q = out / "surrogate_q.png"
    fig_q.savefig(figure_q, dpi=130)
    plt.close(fig_q)

    payload = {
        "best": best,
        "table": table,
        "runs": extracted,
        "figure_samples": str(figure_samples),
        "figure_q": str(figure_q),
        "q_samples": q_records,
        "figures": [str(figure_samples), str(figure_q)],
    }
    _write_json(
        out / "surrogate_summary.json",
        {
            "best": best,
            "runs": extracted,
            "q_samples": q_records,
            "figure_samples": str(figure_samples),
            "figure_q": str(figure_q),
        },
    )
    print(f"surrogate: best={best} by test relative L2", flush=True)
    print(table, flush=True)
    return payload


def speed_table(repeats: int = 5, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Median wall-clock time of one nonlinear solve and of each network's forward pass."""
    from eval.metrics import time_call
    from gs.solver import Grid, ProfileParams, ShapeParams, solve_fixed_boundary
    from models.pinn import (
        NONLINEAR_EVAL_BOX,
        NONLINEAR_PROFILE,
        NONLINEAR_SHAPE,
        load_pinn,
        predict_grid,
    )
    from models.surrogate import load_surrogate, predict

    _threads()
    out = _out(out_dir)
    root = _root()
    entries: list[dict[str, Any]] = []

    shape = ShapeParams(**NONLINEAR_SHAPE)
    profile = ProfileParams(**NONLINEAR_PROFILE)
    grid65 = Grid.uniform(*NONLINEAR_EVAL_BOX, 65, 65)

    def _solve():
        return solve_fixed_boundary(shape, profile, grid65)

    entries.append(
        {
            "id": "fd-nonlinear-65",
            "label": "FD nonlinear solve, 65×65",
            "seconds": float(time_call(_solve, repeats=repeats)),
        }
    )

    pinn_conds = {"solovev": None, "nonlinear": None, "parametric": -0.155}
    for case in PINN_CASES:
        model = load_pinn(root / "outputs" / "pinn" / f"{case}.pt")
        cond = pinn_conds[case]

        def _predict(model=model, cond=cond):
            return predict_grid(model, grid65, cond=cond)

        entries.append(
            {
                "id": f"pinn-{case}",
                "label": f"PINN {case} inference, 65×65",
                "seconds": float(time_call(_predict, repeats=repeats)),
            }
        )

    from data.generate import load_dataset

    data = load_dataset(str(root / "data" / "fixed_65.npz"))
    params = _order_params(data["params"], data["param_names"]).astype(np.float32)
    test_idx = np.flatnonzero(np.asarray(data["split"]) == 2)
    batch = params[test_idx[:500]]
    if batch.shape[0] < 500:
        raise RuntimeError(f"expected 500 test samples, found {batch.shape[0]}")
    single = batch[0]
    summary = _surrogate_summary_doc()
    best = _best_surrogate(summary)
    for name in SURROGATE_RUNS:
        model = load_surrogate(root / "outputs" / "surrogate" / name / "checkpoint.pt")

        def _one(model=model):
            return predict(model, single)

        entries.append(
            {
                "id": name,
                "label": f"surrogate {name}, 1 sample",
                "seconds": float(time_call(_one, repeats=repeats)),
            }
        )
    best_model = load_surrogate(root / "outputs" / "surrogate" / best / "checkpoint.pt")

    def _batch():
        return predict(best_model, batch)

    batch_s = float(time_call(_batch, repeats=repeats))
    entries.append(
        {
            "id": f"{best}-batch500",
            "label": f"surrogate {best}, batch of 500",
            "seconds": batch_s,
        }
    )
    entries.append(
        {
            "id": f"{best}-batch500-per-sample",
            "label": f"surrogate {best}, per sample in a batch of 500",
            "seconds": batch_s / 500.0,
        }
    )

    inv = _inverse_paths()
    if inv is not None:
        from models.inverse import reconstruct

        model = _try_load_inverse(inv["checkpoint"])
        if model is not None:
            signals = np.zeros(model.n_sensors, dtype=np.float32)
            coils = None
            if model.n_coils:
                coils = np.zeros(model.n_coils, dtype=np.float32)

            def _reconstruct(model=model, signals=signals, coils=coils):
                return reconstruct(model, signals, coils)

            entries.append(
                {
                    "id": "inverse-fixed",
                    "label": "inverse reconstruct, 1 sample",
                    "seconds": float(time_call(_reconstruct, repeats=repeats)),
                }
            )
        if "freegs_checkpoint" in inv:
            free = _try_load_inverse(inv["freegs_checkpoint"])
            if free is not None:
                free_sig = np.zeros(free.n_sensors, dtype=np.float32)
                free_coils = np.zeros(free.n_coils, dtype=np.float32) if free.n_coils else None

                def _free(model=free, signals=free_sig, coils=free_coils):
                    return reconstruct(model, signals, coils)

                entries.append(
                    {
                        "id": "inverse-freegs",
                        "label": "inverse FreeGS reconstruct, 1 sample",
                        "seconds": float(time_call(_free, repeats=repeats)),
                    }
                )

    by_id = {row["id"]: row["seconds"] for row in entries}
    headers = ["call", "median time"]
    md_rows = [[row["label"], _fmt_time(row["seconds"])] for row in entries]
    table = _markdown_table(headers, md_rows)
    (out / "speed.md").write_text(
        table
        + "\nMedian of "
        + str(int(repeats))
        + " calls after one warm-up (`eval.metrics.time_call`), torch threads = 2.\n"
        + "The nonlinear solve and the PINN forwards use the 65×65 ITER-like benchmark box. "
        + "Surrogate batches are the 500-sample fixed-boundary test split.\n"
    )

    plt = _import_pyplot()
    labels = [row["label"] for row in entries if not row["id"].endswith("-per-sample")]
    values = [row["seconds"] * 1e3 for row in entries if not row["id"].endswith("-per-sample")]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(labels) + 1.4), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c78a8")
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("median time [ms]")
    ax.set_title(f"Wall-clock comparison ({repeats} repeats)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="both", ls=":", alpha=0.5)
    figure = out / "speed.png"
    fig.savefig(figure, dpi=130)
    plt.close(fig)

    payload = {
        "repeats": int(repeats),
        "torch_threads": 2,
        "entries": entries,
        "seconds": by_id,
        "best_surrogate": best,
        "table": table,
        "figure": str(figure),
    }
    _write_json(out / "speed.json", payload)
    print(table, flush=True)
    return payload


def inverse_summary(out_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Read inverse metrics JSON. Skip, with a fixed message, when the model is absent."""
    inv = _inverse_paths()
    if inv is None:
        print("inverse model not available", flush=True)
        return None
    out = _out(out_dir)
    metrics = _read_json(inv["metrics"])
    freegs = _read_json(inv["freegs_metrics"]) if "freegs_metrics" in inv else None

    def _noise_curve(block: dict | None) -> tuple[list[float], list[float]]:
        if not block:
            return [], []
        xs, ys = [], []
        for key, value in block.items():
            xs.append(float(key))
            ys.append(float(value["relative_l2_psi"]))
        order = np.argsort(xs)
        return [xs[i] for i in order], [ys[i] for i in order]

    plt = _import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    x, y = _noise_curve(metrics.get("noise_sweep"))
    axes[0].plot(x, y, "o-", label="in distribution")
    xo, yo = _noise_curve(metrics.get("ood_noise_sweep"))
    if xo:
        axes[0].plot(xo, yo, "s--", label="OOD kappa")
    if freegs and freegs.get("noise_sweep"):
        xf, yf = _noise_curve(freegs["noise_sweep"])
        axes[0].plot(xf, yf, "^-.", label="FreeGS")
    axes[0].set_xlabel("relative sensor noise")
    axes[0].set_ylabel("test relative L2 of psi")
    axes[0].set_title("Noise sweep (from metrics JSON)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    axes[0].grid(True, ls=":", alpha=0.5)

    names = ["fixed test", "fixed OOD"]
    rels = [metrics["test"]["relative_l2_psi"], metrics.get("ood", {}).get("relative_l2_psi")]
    axes_err = [metrics["test"]["axis_error_m"], metrics.get("ood", {}).get("axis_error_m")]
    if freegs and freegs.get("test"):
        names.append("FreeGS test")
        rels.append(freegs["test"].get("relative_l2_psi"))
        axes_err.append(freegs["test"].get("axis_error_m"))
    xpos = np.arange(len(names))
    axes[1].bar(xpos - 0.16, [float(v) if v is not None else np.nan for v in rels], width=0.32, label="rel L2 psi")
    ax2 = axes[1].twinx()
    ax2.bar(
        xpos + 0.16,
        [float(v) * 1e2 if v is not None else np.nan for v in axes_err],
        width=0.32,
        color="#f58518",
        label="axis error [cm]",
    )
    axes[1].set_xticks(xpos, names, rotation=15)
    axes[1].set_ylabel("relative L2")
    ax2.set_ylabel("axis error [cm]")
    axes[1].set_title("Reconstruction error")
    axes[1].grid(True, axis="y", ls=":", alpha=0.5)
    handles_l, labels_l = axes[1].get_legend_handles_labels()
    handles_r, labels_r = ax2.get_legend_handles_labels()
    axes[1].legend(handles_l + handles_r, labels_l + labels_r, frameon=False, fontsize=8, loc="upper left")
    figure = out / "inverse_summary.png"
    fig.savefig(figure, dpi=130, bbox_inches="tight")
    plt.close(fig)

    test = metrics["test"]
    discussion = _inverse_discussion(metrics)
    (out / "inverse_discussion.md").write_text(discussion)
    print(
        "inverse: "
        f"test relL2={test['relative_l2_psi']:.4f}, "
        f"axis={test['axis_error_m']:.4e} m, "
        f"lcfs={test.get('lcfs_shape_error_m')}, "
        f"q95 abs={test.get('q95_abs_error')}",
        flush=True,
    )
    print(discussion, flush=True)
    payload = {
        "metrics": metrics,
        "freegs": freegs,
        "figure": str(figure),
        "table": discussion,
        "included": True,
    }
    _write_json(
        out / "inverse_summary.json",
        {
            "included": True,
            "test_rel_l2": test.get("relative_l2_psi"),
            "test_axis_error_m": test.get("axis_error_m"),
            "test_q95_abs_error": test.get("q95_abs_error"),
            "ood_rel_l2": (metrics.get("ood") or {}).get("relative_l2_psi"),
            "freegs_test_rel_l2": None if freegs is None else (freegs.get("test") or {}).get("relative_l2_psi"),
            "freegs_test_axis_error_m": None if freegs is None else (freegs.get("test") or {}).get("axis_error_m"),
            "freegs_test_q95_abs_error": None if freegs is None else (freegs.get("test") or {}).get("q95_abs_error"),
            "figure": str(figure),
        },
    )
    return payload


def _inverse_discussion(metrics: dict) -> str:
    """Parameter R^2, architecture variants, and the identifiability note.

    Every quoted number is taken from ``metrics`` or from the surrogate summary.
    """
    clean = metrics.get("param_r2_clean") or {}
    order = ["Ip", "R0", "a", "kappa", "delta", "beta0", "alpha", "gamma", "B0"]
    names = [name for name in order if name in clean]
    names.extend(name for name in clean if name not in names)
    r2_table = ""
    if names:
        r2_table = _markdown_table(
            ["parameter", "R^2"],
            [[name, f"{float(clean[name]):.3f}"] for name in names],
        )

    variants = metrics.get("variants") or []
    variant_table = ""
    if variants:
        variant_rows = []
        for item in variants:
            variant_rows.append(
                [
                    item.get("name", ""),
                    item.get("arch", ""),
                    item.get("epochs", ""),
                    _fmt(item.get("test_rel_l2"), ".4f"),
                    _fmt(item.get("ood_rel_l2"), ".4f"),
                    _fmt(item.get("axis_error_m"), ".4f"),
                    _fmt(item.get("q95_abs_error"), ".3f"),
                    item.get("note") or "",
                ]
            )
        variant_table = _markdown_table(
            ["variant", "arch", "epochs", "test rel L2", "OOD rel L2", "axis error [m]", "q95 abs", "note"],
            variant_rows,
        )

    def _r2(name: str) -> str:
        value = clean.get(name)
        if value is None:
            return "—"
        return f"{float(value):.2f}"

    test = metrics.get("test") or {}
    ood = metrics.get("ood") or {}
    hybrid = next((item for item in variants if item.get("name") == "hybrid_surrogate"), None)
    surrogate_bit = ""
    try:
        summary = _surrogate_summary_doc()
        best = _best_surrogate(summary)
        surrogate_rel = float(summary["runs"][best]["evaluation"]["test"]["rel_l2"]["mean"])
        surrogate_bit = (
            f" Given the true parameters, the best surrogate (`{best}`) has test relative L2 "
            f"{surrogate_rel:.4f} (about {100.0 * surrogate_rel:.1f}%)."
        )
    except (FileNotFoundError, KeyError, TypeError):
        surrogate_bit = ""

    hybrid_bit = ""
    if hybrid is not None and hybrid.get("test_rel_l2") is not None:
        hybrid_bit = (
            f" A hybrid that maps the sensors to the nine equilibrium parameters and then through the "
            f"frozen surrogate has test relative L2 {float(hybrid['test_rel_l2']):.4f}."
        )
    direct_bit = ""
    if test.get("relative_l2_psi") is not None:
        direct_bit = (
            f" The direct network, sensors to psi, reaches {float(test['relative_l2_psi']):.4f}"
        )
        if ood.get("relative_l2_psi") is not None:
            direct_bit += f" in distribution and {float(ood['relative_l2_psi']):.4f} on the high-elongation set"
        extras = []
        if test.get("lcfs_shape_error_m") is not None:
            extras.append(f"LCFS error {float(test['lcfs_shape_error_m']):.4f} m")
        if test.get("q95_abs_error") is not None:
            extras.append(f"absolute q95 error {float(test['q95_abs_error']):.3f}")
        if extras:
            direct_bit += " (" + ", ".join(extras) + ")"
        direct_bit += "."

    finding = (
        "External magnetics fix the plasma current and the major radius, and they barely constrain the "
        f"current profile. On clean signals, R^2 is {_r2('Ip')} for Ip and {_r2('R0')} for R0, against "
        f"{_r2('beta0')} for beta0, {_r2('alpha')} for alpha, and {_r2('gamma')} for gamma. "
        f"The boundary parameters sit in between (a {_r2('a')}, kappa {_r2('kappa')}, delta {_r2('delta')}). "
        f"B0 has R^2 {_r2('B0')}: it does not enter psi, and these poloidal diagnostics do not measure it."
        f"{surrogate_bit}{hybrid_bit}{direct_bit} "
        "The hybrid and the direct decoder land on the same flux error, so that floor is the part of the "
        "equilibrium the external measurements do not determine, rather than a failure to fit the network. "
        "A production EFIT reconstruction adds internal constraints (MSE, pressure, and kinetic profiles) "
        "that this sensor set does not include."
    )
    blocks = ["### Identifiability", "", finding, ""]
    if r2_table:
        blocks.extend(["### Per-parameter R^2 (`param_r2_clean`)", "", r2_table])
    if variant_table:
        blocks.extend(["### Architecture variants", "", variant_table])
    return "\n".join(blocks)


def _speed_seconds(out: Path) -> dict[str, float]:
    path = out / "speed.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    return {str(k): float(v) for k, v in payload.get("seconds", {}).items()}


def overall_table(
    out_dir: str | Path | None = None,
    with_timing: bool = True,
) -> str:
    """Markdown table of every available model. Written to ``outputs/compare/summary.md``."""
    out = _out(out_dir)
    root = _root()
    if with_timing and not (out / "speed.json").is_file():
        speed_table(out_dir=out)
    times = _speed_seconds(out)

    headers = [
        "model",
        "task",
        "input",
        "rel L2 psi",
        "axis error",
        "q95 error",
        "inference time",
    ]
    rows: list[list[str]] = []

    solver_path = out / "solver_verification.json"
    if solver_path.is_file():
        solver = _read_json(solver_path)
        rel65 = None
        for n, rel in zip(solver["n"], solver["rel_l2"]):
            if int(n) == 65:
                rel65 = rel
        if rel65 is None and solver["rel_l2"]:
            rel65 = solver["rel_l2"][-1]
        rows.append(
            [
                "fd-solovev",
                "FD verification vs analytic Solov'ev",
                "boundary + GS right-hand side",
                _fmt(rel65),
                "—",
                "—",
                "—",
            ]
        )
    rows.append(
        [
            "fd-nonlinear-65",
            "nonlinear fixed-boundary solve",
            "shape + profiles, 65×65",
            "reference",
            "—",
            "—",
            _fmt_time(times.get("fd-nonlinear-65")),
        ]
    )

    pinn_side = {}
    side_path = out / "pinn_summary.json"
    if side_path.is_file():
        pinn_side = _read_json(side_path)
    axis_from_figure = pinn_side.get("checkpoint_axis_error_m") or pinn_side.get("figure_axis_error_m") or {}
    for case, task, inputs in (
        ("solovev", "PINN, one Solov'ev equilibrium", "(R, Z)"),
        ("nonlinear", "PINN, one nonlinear equilibrium", "(R, Z)"),
        ("parametric", "PINN conditioned on A", "(R, Z, A)"),
    ):
        metrics_path = root / "outputs" / "pinn" / f"{case}_metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _read_json(metrics_path)
        rel = float(metrics["rel_l2"])
        rel_txt = _fmt(rel)
        if case == "parametric" and metrics.get("rel_l2_mean") is not None:
            rel_txt = f"{_fmt(metrics['rel_l2_mean'])} mean (max {_fmt(rel)})"
        axis = axis_from_figure.get(case)
        if case == "nonlinear":
            axis = _hypot(metrics.get("R_axis_error"), metrics.get("Z_axis_error"))
        rows.append(
            [
                f"pinn-{case}",
                task,
                inputs,
                rel_txt,
                "—" if axis is None else f"{_fmt(axis, '.3e')} m",
                "—",
                _fmt_time(times.get(f"pinn-{case}")),
            ]
        )

    summary_path = root / "outputs" / "surrogate" / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        best = _best_surrogate(summary)
        for name in SURROGATE_RUNS:
            if name not in summary["runs"]:
                continue
            ev = summary["runs"][name]["evaluation"]
            test, ood = ev["test"], ev["ood"]
            rel_txt = f"{_fmt(test['rel_l2']['mean'])} (OOD {_fmt(ood['rel_l2']['mean'])})"
            axis_txt = f"{_fmt(test['axis_error_m']['mean'], '.3e')} m (OOD {_fmt(ood['axis_error_m']['mean'], '.3e')} m)"
            q_txt = f"{_fmt(test['q95_rel_error']['mean'], '.3e')} rel (OOD {_fmt(ood['q95_rel_error']['mean'], '.3e')} rel)"
            rows.append(
                [
                    name,
                    "surrogate, parameters → psi" + (" (best)" if name == best else ""),
                    "R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0",
                    rel_txt,
                    axis_txt,
                    q_txt,
                    _fmt_time(times.get(name)),
                ]
            )

    inv = _inverse_paths()
    if inv is not None:
        metrics = _read_json(inv["metrics"])
        test = metrics["test"]
        ood = metrics.get("ood") or {}
        rel_txt = _fmt(test.get("relative_l2_psi"))
        if ood.get("relative_l2_psi") is not None:
            rel_txt = f"{rel_txt} (OOD {_fmt(ood.get('relative_l2_psi'))})"
        q_abs = test.get("q95_abs_error")
        rows.append(
            [
                "inverse-fixed",
                "inverse, magnetics → psi (1% noise)",
                f"{metrics.get('n_sensors', '—')} magnetic signals",
                rel_txt,
                f"{_fmt(test.get('axis_error_m'), '.3e')} m",
                "—" if q_abs is None else f"{_fmt(q_abs, '.3e')} abs",
                _fmt_time(times.get("inverse-fixed")),
            ]
        )
        if "freegs_metrics" in inv:
            free = _read_json(inv["freegs_metrics"])
            ft = free.get("test") or {}
            q_free = ft.get("q95_abs_error")
            rows.append(
                [
                    "inverse-freegs",
                    "inverse, FreeGS magnetics + coils (1% noise)",
                    "magnetic signals + PF coil currents",
                    _fmt(ft.get("relative_l2_psi")),
                    f"{_fmt(ft.get('axis_error_m'), '.3e')} m",
                    "—" if q_free is None else f"{_fmt(q_free, '.3e')} abs",
                    _fmt_time(times.get("inverse-freegs")),
                ]
            )

    table = _markdown_table(headers, rows)
    notes = _summary_notes(times)
    if inv is not None and "inverse-fixed" not in times:
        notes += (
            "- Inverse rows use the metrics JSON. The checkpoint on disk did not load with the "
            "current `models.inverse`, so its inference time was not measured.\n"
        )
    text = table + "\n" + notes
    (out / "summary.md").write_text(text)
    return text


def _summary_notes(times: dict[str, float]) -> str:
    lines = [
        "Notes:",
        "",
        "- Relative L2 for the PINNs is the checkpoint value on the 129×129 evaluation grid, inside the plasma. The parametric row quotes the mean over the four unseen values of A, with the worst value in parentheses. Its axis error is the mean magnetic-axis error over those same four values.",
        "- The Solov'ev finite-difference row is the relative L2 against the analytic flux on the 65×65 mesh (eps=0.32, kappa=1.7, delta=0.33, A=-0.155). It is a discretization error, not a learned-model error.",
        "- Surrogate numbers are means on the fixed-boundary test split (500 equilibria); OOD is kappa in (2.0, 2.3]. q95 error is relative, |q_pred − q_true| / |q_true| at psi_n = 0.95, using the true F.",
        "- Inverse q95 error is absolute (the metrics JSON does not store a relative q95). The fixed-boundary inverse was trained at 1% sensor noise.",
        "- Inference times are median wall-clock times from `eval.metrics.time_call` (one warm-up, then the recorded repeats) with `torch.set_num_threads(2)`. PINN times are one 65×65 grid. Surrogate times are one sample. The nonlinear solve is the 65×65 benchmark equilibrium.",
    ]
    batch_keys = [key for key in times if key.endswith("-batch500") and not key.endswith("-per-sample")]
    if batch_keys:
        key = batch_keys[0]
        lines.append(
            f"- Batched surrogate throughput ({key}): {_fmt_time(times[key])} for 500 samples "
            f"({_fmt_time(times.get(key + '-per-sample'))} per sample)."
        )
    lines.append("")
    return "\n".join(lines)


def build_notebook(path: str | Path | None = None) -> Path:
    """Write ``notebooks/comparison.ipynb`` with markdown and calls into this module."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    notebook_path = Path(path) if path is not None else _root() / "notebooks" / "comparison.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        new_markdown_cell(
            """# Grad-Shafranov equilibria: solver, PINN, surrogate, inverse

Axisymmetric tokamak equilibrium is the Grad-Shafranov problem. In cylindrical coordinates the poloidal flux $\\psi(R, Z)$ satisfies

$$
\\Delta^*\\psi \\equiv R\\partial_R\\negthinspace{}\\left(\\frac{1}{R}\\partial_R\\psi\\right) + \\partial_Z^2\\psi = -\\mu_0 R J_\\phi,
$$

with $J_\\phi$ set by the pressure and toroidal-field profiles, which themselves depend on $\\psi$. The fixed-boundary problem takes the plasma shape and those profiles and returns $\\psi$. The free-boundary problem, and equilibrium reconstruction, also have to place the boundary so that the external magnetic measurements are matched.

A nonlinear finite-difference solve is accurate and, on a $65\\times 65$ mesh, only a few tens of milliseconds. Real-time shape control and between-shot reconstruction still want a map that is cheaper than a fresh solve, and a reconstruction map that goes straight from diagnostics to $\\psi$. This notebook compares four approaches that share one set of conventions ($\\psi = 0$ on the boundary, $\\psi_\\mathrm{axis} \\gt 0$ for the solver family, arrays shaped `(nr, nz)` with `indexing="ij"`, SI units):

1. A second-order finite-difference solver, checked against a Cerfon–Freidberg Solov'ev equilibrium.
2. Physics-informed networks trained on the strong form, with no interior flux target.
3. Supervised surrogates (MLP–CNN and FNO) from the nine equilibrium parameters to $\\psi(R, Z)$.
4. An inverse network from synthetic magnetic diagnostics to $\\psi$, when that checkpoint is present.

No model is retrained here. Checkpoints under `outputs/` are loaded as they are."""
        ),
        new_code_cell(
            """import os
import sys

sys.path.insert(0, os.path.abspath(".."))
os.chdir(os.path.abspath(".."))

import torch
torch.set_num_threads(2)

from IPython.display import Image, Markdown, display
from eval import compare


def show(result):
    if result is None:
        return
    if isinstance(result, str):
        display(Markdown(result))
        return
    table = result.get("table")
    if table:
        display(Markdown(table))
    paths = []
    for key in ("figure", "figure_samples", "figure_q"):
        if result.get(key):
            paths.append(result[key])
    for extra in result.get("figures") or []:
        if extra not in paths:
            paths.append(extra)
    for path in paths:
        display(Image(filename=path))
"""
        ),
        new_markdown_cell(
            """## Finite-difference verification

The Solov'ev equilibrium used here is the up-down symmetric Cerfon–Freidberg solution with $\\epsilon = 0.32$, $\\kappa = 1.7$, $\\delta = 0.33$, $A = -0.155$, and $R_0 = 1\\thinspace{}\\mathrm{m}$. The analytic flux is negative inside the plasma. The finite-difference code solves $\\Delta^*\\psi =$ that analytic right-hand side with $\\psi = 0$ on the curved boundary (Shortley–Weller cuts). On a smooth solution the truncation error is $O(h^2)$, so the relative L2 should fall with slope 2 on a log-log plot against the mesh spacing."""
        ),
        new_code_cell(
            """solver = compare.solver_verification()
show(solver)
for n, err, order in zip(solver["n"], solver["rel_l2"], solver["observed_order"]):
    print(f"n={n:4d}  rel L2={err:.6e}  order={order}")
"""
        ),
        new_markdown_cell(
            """## Physics-informed networks

Each PINN is an MLP on affinely scaled $(R, Z)$. The Solov'ev network matches $\\Delta^*\\psi$ to the analytic right-hand side. The nonlinear network uses a lagged (Picard) current profile with the same shape and profiles as the finite-difference benchmark. The parametric network adds the Solov'ev parameter $A$ as a conditioning input and is scored on values of $A$ it did not train on.

The figure overlays interior flux surfaces (solid: reference, dashed: network) and the pointwise error. Relative L2 numbers printed from the checkpoint were computed on this same $129\\times 129$ box."""
        ),
        new_code_cell(
            """pinn = compare.pinn_summary()
show(pinn)
print("checkpoint rel L2", {k: pinn["metrics"][k]["rel_l2"] for k in pinn["metrics"]})
print("axis error [m]", pinn["checkpoint_axis_error_m"])
"""
        ),
        new_markdown_cell(
            """## Supervised surrogates

The surrogates are trained on 4000 fixed-boundary equilibria (validation 500, test 500) that share one $65\\times 65$ grid. Inputs are $(R_0, a, \\kappa, \\delta, I_p, \\beta_0, \\alpha, \\gamma, B_0)$. Two architectures, an MLP–CNN and a Fourier neural operator, are each trained with physics weight 0 and 0.1. The physics term penalizes $\\Delta^*\\psi_\\mathrm{pred} + \\mu_0 R J_\\mathrm{true}$ on interior plasma nodes. The out-of-distribution file keeps every range the same except elongation, which sits in $(2.0, 2.3]$.

The sample figure is the best run by mean test relative L2. Flux surfaces of the stored solution and of the prediction are overlaid; the second column is the error inside the known boundary. The safety-factor comparison uses three test equilibria. $F(\\psi_n) = R B_\\phi$ comes from a fresh `solve_fixed_boundary` of that equilibrium (the true $F$), and $q$ is the contour integral in `eval.metrics.q_profile` on both the stored $\\psi$ and the predicted $\\psi$, for $\\psi_n$ from 0.05 to 0.95."""
        ),
        new_code_cell(
            """surrogate = compare.surrogate_summary()
show(surrogate)
print("best:", surrogate["best"])
for sample in surrogate["q_samples"]:
    print(
        f"test #{sample['dataset_index']}: rel L2={sample['rel_l2']:.4e}  "
        f"q95 true={sample['q95_true']:.3f} pred={sample['q95_pred']:.3f}"
    )
"""
        ),
        new_markdown_cell(
            """## Speed

Times are medians from `eval.metrics.time_call` (one untimed warm-up, then five calls). The finite-difference entry is one nonlinear fixed-boundary solve on the $65\\times 65$ benchmark. Each PINN entry is a forward pass on that same grid. Surrogate entries are a single sample of each trained network, plus a batch of the 500 test equilibria for the best network. Inverse timing is one `reconstruct` call when that checkpoint exists."""
        ),
        new_code_cell(
            """speed = compare.speed_table()
show(speed)
"""
        ),
        new_markdown_cell(
            """## Inverse reconstruction

The inverse model maps synthetic pickup-probe, flux-loop, and Rogowski signals to $\\psi$. It is optional in this report: if the package or the checkpoint is missing, the cell prints `inverse model not available` and continues. When the checkpoint is present the curves below are read from the metrics JSON (no new sensor sweep and no retraining), and one `reconstruct` call is timed in the speed section. Training noise is 1% of each channel's RMS. A second head was trained on the 298-sample FreeGS TestTokamak set when that file was available.

The selected network is a direct decoder: it predicts the normalized flux shape and a scale, with sensor features normalized by $I_p$. External magnetics determine some equilibrium parameters and leave others free. The cell below prints `param_r2_clean` and the variant table from the metrics file. The pattern is that $I_p$ and $R_0$ are fixed by the measurements, while $\\beta_0$, $\\alpha$, and $\\gamma$ are only weakly constrained, and $B_0$ does not enter $\\psi$. A hybrid that predicts the nine parameters and then evaluates the frozen surrogate — accurate when those parameters are the true ones — reaches the same flux error as the direct model. That shared floor is missing information. A production EFIT fit adds internal constraints (MSE, pressure, and kinetic profiles) that this diagnostic set does not have."""
        ),
        new_code_cell(
            """inverse = compare.inverse_summary()
show(inverse)
"""
        ),
        new_markdown_cell(
            """## Summary

One row per model. Relative L2, axis error, and $q_{95}$ error are taken from the checkpoint metrics. Inference times are the measurements from the speed section. Empty cells are quantities that the corresponding metrics file does not record."""
        ),
        new_code_cell(
            """table = compare.overall_table()
show(table)
"""
        ),
    ]
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbformat.write(nb, notebook_path)
    return notebook_path


def main() -> None:
    _threads()
    solver_verification()
    pinn_summary()
    surrogate_summary()
    speed_table()
    inverse_summary()
    text = overall_table()
    print(text, flush=True)


if __name__ == "__main__":
    main()
