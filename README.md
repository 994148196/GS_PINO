# GS_PINO

Masked U-FNO/PINO surrogate for fixed-boundary Grad-Shafranov equilibria.

The code trains a neural operator that maps
`[R0, a, kappa, delta, Ip, betap, alpha_m, alpha_n]` plus grid coordinates and LCFS geometry channels to the normalized in-LCFS poloidal flux `psi_bar`.

## Quick start

```bash
python -m gs_pino.generate_dataset --out data/gs_fixed_boundary.npz --n-samples 64 --nr 64 --nz 64
python -m gs_pino.train --data data/gs_fixed_boundary.npz --epochs 20 --output-dir outputs/run
python -m gs_pino.evaluate --data data/gs_fixed_boundary.npz --checkpoint outputs/run/best.pt --output-dir outputs/run_eval
```

The included generator is a solver-adapter scaffold with an analytic fixed-boundary fallback. Replace `AnalyticFixedBoundarySolver` in `src/gs_pino/solvers.py` with the external `GS_solver` call when that package is installed.
