"""Physics-informed FNO for the GS equation (exp101/102 experiments).

Adds Grad-Shafranov PDE residuals to the paper FNO training:
  - mode `rhs`     : single-stage, residual from dataset RHS (J_data from
                     dpdpsi/FdFdpsi), loss = MSE(psi_plasma) + w_pde * PDE
  - mode `twostage`: stage-1 supervised psi_plasma + J; stage-2 adds the
                     self-consistent PDE residual (network J) + Ip constraint.

Coil separation: the network predicts only psi_plasma; the total flux is
psi_total = psi_plasma + sum_k I_k * G_k (exact Green-function addition at
evaluation time, greens are stored in the npz).
"""
