"""One-time verification of the physics residual implementation on TRUTH fields.

Run before training: python -m gs_pino_fno_phys.verify_pde

Checks (all must pass before exp101/102 training is meaningful):
  1. torch lap_star(psi_plasma_true) + mu0*R*J_data == 0 inside the core mask
     -> mean|res| ~ 8e-4 Wb/m^2, residual ratio ~ 0.0033
  2. torch lap_star agrees with evaluate_dn_fno.lap_star (numpy) elementwise,
     and the torch residual ratio matches evaluate's gs_residual_ratio
  3. greens identity: sum_k I_k * G_k == psi_coils (max diff < 1e-7 Wb)
  4. Ip reconstruction: sum(mask*J)*dA == params[:,0] (mean rel err < 1e-3)
  5. PDE-loss floor: masked mean((res/pde_scale)^2) ~ 5e-6 (FD truncation)
"""
from __future__ import annotations

import numpy as np
import torch

from gs_pino_dn_fno_2608 import evaluate_dn_fno
from gs_pino_fno_phys.data_pino import DNPinoDataset, MU0
from gs_pino_fno_phys.losses_pino import lap_star_torch, pde_residual_rhs

N = 200
NPZ = "dn_fno_2608/data_v5/dn/train.npz"


def main() -> None:
    ds = DNPinoDataset(NPZ, indices=np.arange(N))
    R_c = torch.from_numpy(ds.R_phys[1:-1, 1:-1])
    dR, dZ = ds.dR, ds.dZ

    psi_true = torch.from_numpy(ds.psi_plasma[:N])[:, None]        # (N,1,65,65) Wb
    j_true = torch.from_numpy(ds.j_phys[:N])[:, None]              # (N,1,65,65) A/m^2
    mask_i = torch.from_numpy(ds.mask[:N])[:, None, 1:-1, 1:-1] > 0.5

    lap = lap_star_torch(psi_true, R_c, dR, dZ)
    rhs = MU0 * R_c[None, None] * j_true[..., 1:-1, 1:-1]
    res = lap + rhs

    m = mask_i.bool()
    mean_abs = float((res.abs()[m]).mean())
    ratio = float(torch.linalg.vector_norm(res[m]) / torch.linalg.vector_norm(rhs[m]))
    l_pde = float(pde_residual_rhs(psi_true, j_true, R_c, dR, dZ,
                                   mask_i.float(), ds.pde_scale))

    print(f"[1] truth residual  mean|res| = {mean_abs:.4e} Wb/m^2  (expect ~8e-4)")
    print(f"[1] residual ratio          = {ratio:.4f}             (expect ~0.0033)")
    print(f"[5] L_pde floor (scaled)^2  = {l_pde:.4e}             (expect ~5e-6)")

    # 2) numpy cross-check on sample 0
    lap_np = evaluate_dn_fno.lap_star(ds.psi_plasma[0], ds.R_phys, ds.Z_phys)
    lap_t = lap[0, 0].numpy()
    diff = np.abs(lap_np - lap_t).max()
    with np.load(NPZ) as d:
        dp, ff, msk = d["dpdpsi"][0], d["FdFdpsi"][0], d["mask"][0]
    ratio_np = evaluate_dn_fno.gs_residual_ratio(
        ds.psi_plasma[0], ds.R_phys, ds.Z_phys, dp, ff, msk)
    ratio_t = float(torch.linalg.vector_norm(res[0][m[0]])
                    / torch.linalg.vector_norm(rhs[0][m[0]]))
    print(f"[2] torch vs numpy lap_star  max diff = {diff:.2e}        (expect <1e-8)")
    print(f"[2] torch ratio {ratio_t:.4f} vs gs_residual_ratio {ratio_np:.4f}")

    # 3) greens identity
    with np.load(NPZ) as d:
        psi_coils = d["psi_coils"][:N]
        greens = d["greens"][:N]
        coils = d["coil_currents"][:N]
    psi_coils_est = np.einsum("nijk,ni->njk", greens, coils)
    gdiff = float(np.abs(psi_coils_est - psi_coils).max())
    print(f"[3] greens identity  max|sum(I*G) - psi_coils| = {gdiff:.2e}  (expect <1e-7)")

    # 4) Ip reconstruction (dataset-level)
    ip_est = (ds.j_phys * (ds.mask > 0.5)).sum(axis=(1, 2)) * ds.dR * ds.dZ
    rel = np.abs(ip_est - ds.params[:, 0]) / ds.params[:, 0]
    print(f"[4] Ip reconstruction  mean rel err = {rel.mean():.2e}  (expect <1e-3)")

    ok = (mean_abs < 5e-3 and ratio < 2e-2 and diff < 1e-8
          and abs(ratio_t - ratio_np) < 5e-3 and gdiff < 1e-7
          and rel.mean() < 1e-3 and l_pde < 5e-5)
    print(f"\n{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")


if __name__ == "__main__":
    main()
