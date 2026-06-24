# GS_PINO

Masked U-FNO/PINO surrogate for fixed-boundary Grad-Shafranov equilibria.

The model maps
`[R0, a, kappa, delta, Ip, betap, alpha_m, alpha_n]` plus grid coordinates and
LCFS geometry channels to the normalized in-LCFS poloidal flux `psi_bar`.

## Quick start

```bash
python -m gs_pino.generate_dataset --out data/gs_fixed_boundary.npz --n-samples 64 --nr 64 --nz 64
python -m gs_pino.train --data data/gs_fixed_boundary.npz --epochs 20 --output-dir outputs/run
python -m gs_pino.evaluate --data data/gs_fixed_boundary.npz --checkpoint outputs/run/best.pt --output-dir outputs/run_eval
```

For a tiny end-to-end smoke test:

```bash
./scripts/run_smoke.sh
```

For a larger practical workflow:

```bash
./scripts/run_practical.sh
```

## Documentation

See [`PROJECT.md`](PROJECT.md) for the detailed project guide, including:

- generated `.npz` data structure;
- U-FNO input channels;
- model architecture;
- loss definitions;
- evaluation figures and metrics;
- practical training workflow;
- notes for replacing the analytic fallback with the real `GS_solver`.

The included generator is a solver-adapter scaffold with an analytic fixed-boundary
fallback. Replace `AnalyticFixedBoundarySolver` in `src/gs_pino/solvers.py` with
the external `GS_solver` call when that package is installed.
