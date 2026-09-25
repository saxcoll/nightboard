# fusion-gs

Teaching notes on the physics and the results: [docs/REPORT.md](docs/REPORT.md).

Machine-learning models for tokamak plasma equilibria. The target is the Grad–Shafranov equation: a fixed-boundary finite-difference solver, physics-informed networks, supervised surrogates from equilibrium parameters to poloidal flux, and an inverse map from synthetic magnetic diagnostics.

Fast equilibria matter because shape control and between-shot reconstruction call the same solve many times. A 65×65 nonlinear solve here takes 44 ms on CPU. A trained network evaluates one equilibrium in about 1–3 ms, and a batch of 500 in 273 ms.

## Equation

In cylindrical \((R, Z)\),

\[
\Delta^*\psi = R\,\partial_R\!\left(\frac{1}{R}\partial_R\psi\right) + \partial_Z^2\psi = -\mu_0 R J_\phi.
\]

\(J_\phi\) depends on the normalized flux through the pressure gradient and \(FF'\). The fixed-boundary problem prescribes the plasma shape and the profile parameters and returns \(\psi\). Reconstruction goes the other way: magnetic measurements to \(\psi\).

The shaped boundary used by the solver is the \(\bar\psi = 0\) surface of an up-down symmetric Cerfon–Freidberg Solov'ev equilibrium.

## Layout

```
gs/            analytic Solov'ev equilibria and the finite-difference solver
data/          fixed-boundary and FreeGS dataset generators
models/        PINN, surrogate, inverse network, synthetic diagnostics
eval/          metrics and the comparison report (eval.compare)
notebooks/     comparison.ipynb
tests/
outputs/       checkpoints, metrics, and comparison figures (gitignored)
```

## Setup

From this directory:

```bash
pip install -r requirements.txt
```

FreeGS 0.8.2 on PyPI fails under NumPy 2.x. `requirements.txt` installs FreeGS from the GitHub `main` branch, which works with NumPy 2. Re-executing the notebook also needs `nbconvert` and `ipykernel` (`nbformat` is already listed).

Commands below are run from the `fusion-gs` directory. PyTorch is limited to two threads (`torch.set_num_threads(2)`); the machine these timings used has four CPUs shared with other work.

## Reproduce

### Data

In-distribution fixed boundary (5000 equilibria, split 4000/500/500), out-of-distribution elongation, and FreeGS TestTokamak:

```bash
python -m data.generate fixed --n 5000 --nr 65 --nz 65 --seed 0 --out data/fixed_65.npz
python -m data.generate fixed --ood --n 500 --nr 65 --nz 65 --seed 0 --out data/fixed_65_ood.npz
python -m data.generate freegs --n 600 --seed 0 --nx 65 --ny 65 --out data/freegs_65.npz
```

Solves that fail the keep criteria are dropped. The FreeGS file used for the numbers below contains 298 equilibria (238/30/30), not 600.

Fixed-boundary inputs are \(R_0, a, \kappa, \delta, I_p, \beta_0, \alpha, \gamma, B_0\). The OOD file uses the same ranges except \(\kappa \in (2.0, 2.3]\).

### PINN

```bash
python -m models.pinn --case solovev --steps 5000 --seed 0 --out outputs/pinn
python -m models.pinn --case nonlinear --steps 5000 --seed 0 --out outputs/pinn
python -m models.pinn --case parametric --steps 5000 --seed 0 --out outputs/pinn
```

Checkpoints: `outputs/pinn/{solovev,nonlinear,parametric}.pt` with matching `*_metrics.json`. The Solov'ev and parametric cases are matched to the analytic right-hand side. The nonlinear case is a Picard iteration for one fixed equilibrium. No interior flux target is used.

### Surrogate

The four runs behind `outputs/surrogate/summary.json`:

```bash
python -m models.surrogate --data data/fixed_65.npz --model mlpcnn --physics-weight 0 \
    --epochs 36 --patience 12 --physics-warmup 6 --seed 0 --out outputs/surrogate/mlpcnn_pw0
python -m models.surrogate --data data/fixed_65.npz --model mlpcnn --physics-weight 0.1 \
    --epochs 36 --patience 12 --physics-warmup 6 --seed 0 --out outputs/surrogate/mlpcnn_pw0p1
python -m models.surrogate --data data/fixed_65.npz --model fno --physics-weight 0 \
    --epochs 24 --patience 8 --physics-warmup 5 --seed 0 --out outputs/surrogate/fno_pw0
python -m models.surrogate --data data/fixed_65.npz --model fno --physics-weight 0.1 \
    --epochs 24 --patience 8 --physics-warmup 5 --seed 0 --out outputs/surrogate/fno_pw0p1
```

`--ood` defaults to `data/fixed_65_ood.npz` when that file is present. `--q95-samples` defaults to 50.

### Inverse

```bash
python -m models.inverse --data data/fixed_65.npz --noise 0.01 --epochs 48 --patience 16 --threads 2 --arch direct --out outputs/inverse
```

The argparse default for `--epochs` is 150 and for `--arch` is `auto`. The metrics file used below records `epochs_requested: 48`, `arch: direct`, noise 0.01, patience 16, and 2 threads. That direct model predicts the flux shape and a scale (`psi / psi_axis`), with sensor features normalized by \(I_p\). If `data/freegs_65.npz` is present the same CLI also trains `outputs/inverse/freegs` unless `--skip-freegs` is set. `--arch hybrid` needs `--surrogate` pointing at a frozen params-to-psi checkpoint.

### Evaluation

```bash
python -m eval.compare
```

This loads the checkpoints already on disk, writes figures and `outputs/compare/summary.md`, and does not retrain. The notebook `notebooks/comparison.ipynb` calls the same functions. Its first cell puts the project root on `sys.path` and changes to that directory, so it runs from `notebooks/`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/comparison.ipynb --ExecutePreprocessor.timeout=900
```

### Visualization

```bash
python -m eval.visualize --out outputs/visualize
```

Writes figures for one fixed-boundary equilibrium and for the saved models. Optional flags `--R0 --a --kappa --delta --Ip --beta0 --alpha --gamma --B0 --grid-n` override the default case (R0 = 1.7 m, a = 0.6 m, κ = 1.7, δ = 0.3, Ip = 1 MA, β0 = 0.5, α = 1, γ = 2, B0 = 2 T, 129×129).

| file | contents |
| --- | --- |
| `equilibrium.png` | ψ_n with labelled flux surfaces and the LCFS, J_φ, pressure / F / q(ψ_n), and the scalar summary (q95, Shafranov shift, β_p, iterations) |
| `surrogate_vs_solver.png` | `mlpcnn_pw0p1` against the solver on the training grid R ∈ [0.80, 2.76], Z ∈ [−1.90, 1.90], 65×65. Solid contours are the solver, dashed are the network; the second panel is \|Δψ\| / ψ_axis |
| `pinn_vs_solver.png` | nonlinear PINN against the solver for shape (1.7, 0.6, 1.7, 0.3) and profiles (Ip = 1 MA, β0 = 0.5, α = 1, γ = 2, B0 = 2 T) |
| `kappa_sweep.gif` | flux surfaces as κ runs from 1.2 to 2.0, solver solid and the surrogate dashed |
| `delta_sweep.gif` | the same scan in triangularity, δ from 0 to 0.5 |

A missing checkpoint is skipped. The sweeps use the command-line shape and profiles as the fixed parameters.

### Tests

```bash
python3 -m pytest -q -m 'not slow'
```

Training tests are marked `slow`.

## Results

Numbers are from `outputs/compare/summary.md` after `python -m eval.compare` on the saved checkpoints. Surrogate and inverse errors are means over the test split. OOD is the high-elongation fixed-boundary file.

On the Solov'ev benchmark (\(\epsilon=0.32\), \(\kappa=1.7\), \(\delta=0.33\), \(A=-0.155\)) the finite-difference error against the analytic flux is \(1.362\times 10^{-4}\) at \(33^2\), \(3.059\times 10^{-5}\) at \(65^2\), and \(6.822\times 10^{-6}\) at \(129^2\). The last refinement has observed order 2.16.

The Solov'ev PINN matches that analytic flux to a relative L2 of \(3.15\times 10^{-4}\) on the \(129^2\) grid. The nonlinear PINN, compared with the finite-difference solution of the same equilibrium, has relative L2 \(8.48\times 10^{-3}\) and a magnetic-axis error of \(5.03\times 10^{-3}\) m. The parametric PINN, scored on four unseen values of \(A\), has mean relative L2 \(1.55\times 10^{-3}\) (worst \(1.94\times 10^{-3}\)).

Among the surrogates, `mlpcnn_pw0p1` is the best on the test split: relative L2 \(1.09\times 10^{-2}\) (OOD \(1.86\times 10^{-2}\)), axis error \(1.40\times 10^{-3}\) m, and relative \(q_{95}\) error \(2.43\times 10^{-2}\). The physics weight is visible in the Grad–Shafranov residual. For the MLP–CNN the mean test residual drops from \(0.977\) at weight 0 to \(0.0327\) at weight 0.1; for the FNO it drops from \(0.564\) to \(0.051\).

The inverse model is the direct architecture in `outputs/inverse/metrics.json` (48 epochs). On the fixed-boundary test split its relative L2 is \(0.1707\) (OOD \(0.1894\)), the magnetic-axis error is \(6.063\times 10^{-3}\) m, the LCFS error is \(0.0656\) m, and the absolute \(q_{95}\) error is \(1.403\). Train and validation relative L2 are \(0.1688\) and \(0.1751\). The FreeGS inverse, on 30 test equilibria, has relative L2 \(0.1154\). One `reconstruct` call takes 1.20 ms for either checkpoint.

External magnetics fix some parameters and leave the current profile free. Clean-signal \(R^2\) (`param_r2_clean`) is 1.00 for \(I_p\) and 0.96 for \(R_0\), against 0.00 for \(\beta_0\), 0.22 for \(\alpha\), and 0.41 for \(\gamma\). The boundary is only partly determined (\(a\) 0.43, \(\kappa\) 0.51, \(\delta\) 0.58). \(B_0\) has \(R^2\) \(-0.03\): it does not enter \(\psi\), and these poloidal diagnostics do not measure it. With the true parameters, `mlpcnn_pw0p1` has test relative L2 0.0109 (about 1.1%). The hybrid variant, sensors to the nine parameters and then that frozen surrogate, has test relative L2 0.1696, the same floor as the direct decoder at 0.1707. The error is the part of the equilibrium the external measurements do not determine. A production EFIT fit adds internal constraints (MSE, pressure, and kinetic profiles) that this sensor set does not include.

| model | task | input | rel L2 psi | axis error | q95 error | inference time |
| --- | --- | --- | --- | --- | --- | --- |
| fd-solovev | FD verification vs analytic Solov'ev | boundary + GS right-hand side | 3.0586e-05 | — | — | — |
| fd-nonlinear-65 | nonlinear fixed-boundary solve | shape + profiles, 65×65 | reference | — | — | 44.32 ms |
| pinn-solovev | PINN, one Solov'ev equilibrium | (R, Z) | 3.1462e-04 | 4.684e-05 m | — | 2.06 ms |
| pinn-nonlinear | PINN, one nonlinear equilibrium | (R, Z) | 8.4817e-03 | 5.031e-03 m | — | 2.05 ms |
| pinn-parametric | PINN conditioned on A | (R, Z, A) | 1.5538e-03 mean (max 1.9414e-03) | 3.376e-04 m | — | 2.07 ms |
| mlpcnn_pw0 | surrogate, parameters → psi | R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0 | 1.4165e-02 (OOD 2.1474e-02) | 8.356e-03 m (OOD 2.235e-02 m) | 1.356e-01 rel (OOD 8.442e-02 rel) | 1.09 ms |
| mlpcnn_pw0p1 | surrogate, parameters → psi (best) | R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0 | 1.0868e-02 (OOD 1.8626e-02) | 1.397e-03 m (OOD 2.300e-03 m) | 2.427e-02 rel (OOD 4.639e-02 rel) | 1.05 ms |
| fno_pw0 | surrogate, parameters → psi | R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0 | 2.4592e-02 (OOD 3.8916e-02) | 5.765e-03 m (OOD 7.389e-03 m) | 2.596e-01 rel (OOD 2.394e-01 rel) | 2.89 ms |
| fno_pw0p1 | surrogate, parameters → psi | R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0 | 2.1135e-02 (OOD 3.6404e-02) | 3.388e-03 m (OOD 6.074e-03 m) | 2.736e-01 rel (OOD 2.856e-01 rel) | 2.84 ms |
| inverse-fixed | inverse, magnetics → psi (1% noise) | 101 magnetic signals | 1.7067e-01 (OOD 1.8937e-01) | 6.063e-03 m | 1.403e+00 abs | 1.20 ms |
| inverse-freegs | inverse, FreeGS magnetics + coils (1% noise) | magnetic signals + PF coil currents | 1.1541e-01 | 1.207e-02 m | — | 1.20 ms |

Inference times are medians from `eval.metrics.time_call` (one warm-up, then five calls). The nonlinear solve and the PINN forwards use the 65×65 benchmark box. Surrogate times are one sample; a batch of 500 with `mlpcnn_pw0p1` takes 273 ms (0.55 ms per sample). Inverse times are one `reconstruct` call. The parametric axis error is the mean over the four unseen values of \(A\). Surrogate \(q_{95}\) errors are relative. The inverse \(q_{95}\) error is absolute, which is all the metrics file stores. The Solov'ev finite-difference relative L2 is the discretization error on the 65×65 mesh.

## Conventions

- SI units: metres, amperes, tesla, \(\psi\) in Wb/rad, \(J\) in A/m².
- Grid arrays have shape `(nr, nz)` with `indexing="ij"`: `psi[i, j]` is at `(R[i], Z[j])`.
- \(\Delta^*\psi = -\mu_0 R J_\phi\).
- For the finite-difference solver, the datasets, the surrogates, and the inverse model, \(\psi = 0\) on the plasma boundary and \(\psi_\mathrm{axis} > 0\) when \(I_p > 0\). \(\psi_n = 0\) on axis and \(1\) on the boundary.
- The analytic Cerfon–Freidberg flux \(\bar\psi\) is negative inside the plasma and zero on the boundary. The Solov'ev PINN learns that sign. The solver verification compares the finite-difference solution of the analytic right-hand side with that analytic flux.

## Limitations

The supervised models see fixed-boundary synthetic equilibria with up-down symmetric Cerfon–Freidberg shapes and a two-parameter current profile. They are not a free-boundary equilibrium code, and they have not been trained on experimental magnetics. The FreeGS set is the TestTokamak coil set only, 298 samples, so the free-boundary inverse is a small-data result (30 test cases) and its \(q_{95}\) comparison did not complete. All networks here are CPU-scale (PINN width 64; surrogate MLP–CNN 629002 parameters; FNO 202786). A relative L2 near \(10^{-2}\) on flux moves \(q_{95}\) by a few percent on the best surrogate. The inverse model's absolute \(q_{95}\) error is 1.40 on the fixed-boundary test set, consistent with a current profile that the external diagnostics barely determine. The FreeGS \(q_{95}\) comparison did not return a value (30 failures out of 30).

## References

- A. J. Cerfon and J. P. Freidberg, "One size fits all" analytic solutions to the Grad–Shafranov equation, *Phys. Plasmas* **17**, 032502 (2010).
- FreeGS, https://github.com/freegs-plasma/freegs
- M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-informed neural networks, *J. Comput. Phys.* **378**, 686 (2019).
- Z. Li et al., Fourier neural operator for parametric PDEs, ICLR (2021).
- Neural surrogates of the QLKNN kind (van de Plassche, Citrin, et al.) are the fusion precedent for replacing a repeated physics solve with a network; the surrogates here do that for the Grad–Shafranov solve rather than for turbulent transport.
