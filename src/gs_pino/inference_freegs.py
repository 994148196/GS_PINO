"""Inference module for the free-boundary GS PINO surrogate.

Provides functions to predict psi_total from coil currents and plasma parameters,
using freegs to compute the vacuum contribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data_freegs import Normalization, build_ufno_input
from .models import PlaNetCore, UFNO2d_v2


def compute_greens(tokamak, R, Z):
    """Compute Green functions for all control coils.

    Important: the training data was generated on freegs' native grid
    (2^n + 1 = 65 points) and then quintic-interpolated down to 64x64
    (see data_freegs._preload_data). Green's function has steep gradients
    near the coil centers, so evaluating it directly on the 64x64 grid would
    NOT reproduce the training psi_coils (measured max diff ~0.1 vs the
    stored fields). We reproduce the exact training preprocessing here:
    greens are computed at 65x65 and interpolated to the target grid.

    Args:
        tokamak: freegs tokamak object
        R: 2D array of major radius coordinates [nx, ny]
        Z: 2D array of height coordinates [nx, ny]

    Returns:
        greens: [n_coils, nx, ny] array of Green functions
        coil_names: list of coil names
    """
    from .data_freegs import interp_fun

    nr, nz = R.shape
    if nr == nz == 64:  # training grid: greens must come from the 65x65 freegs grid
        r1 = np.linspace(R[0, 0], R[-1, -1], nr + 1)
        z1 = np.linspace(Z[0, 0], Z[-1, -1], nz + 1)
        R65, Z65 = np.meshgrid(r1, z1, indexing="ij")
        interp = True
    else:  # any other grid (e.g. freegs-native 65x65): use it directly
        R65, Z65 = R, Z
        interp = False

    greens = []
    coil_names = []
    for label, coil in tokamak.coils:
        if coil.control:
            g = coil.createPsiGreens(R65, Z65).astype(np.float32)
            if interp:
                g = interp_fun(g, R65, Z65, R, Z).astype(np.float32)
            greens.append(g)
            coil_names.append(label)
    return np.stack(greens), coil_names


def compute_psi_coils(coil_currents: np.ndarray, greens: np.ndarray) -> np.ndarray:
    """Compute vacuum flux from coil currents and Green functions.
    
    psi_coils = sum(I_k * G_k(R,Z))
    
    Args:
        coil_currents: [n_coils] array of coil currents
        greens: [n_coils, nx, ny] array of Green functions
        
    Returns:
        psi_coils: [nx, ny] array of vacuum poloidal flux
    """
    psi_coils = np.zeros_like(greens[0])
    for i, I in enumerate(coil_currents):
        psi_coils += I * greens[i]
    return psi_coils


def predict_psi(
    model: torch.nn.Module,
    model_type: str,
    R: np.ndarray,
    Z: np.ndarray,
    coil_currents: np.ndarray,
    params: np.ndarray,
    param_norm: Normalization,
    greens: np.ndarray,
    predict_plasma: bool,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """Predict psi_total from coil currents and plasma parameters.

    Args:
        model: Trained model (PlaNetCore or UFNO2d_v2)
        model_type: "planet" or "ufno"
        R: 2D array of major radius coordinates [nx, ny]
        Z: 2D array of height coordinates [nx, ny]
        coil_currents: [4] array of coil currents
        params: [5] array of plasma parameters [Ip, paxis, alpha_m, alpha_n, fvac]
        param_norm: Normalization object for parameters
        greens: [4, nx, ny] array of Green functions
        predict_plasma: whether the model predicts psi_plasma directly
        device: torch device

    Returns:
        Dictionary with:
            - psi_plasma: Predicted plasma contribution
            - psi_coils: Vacuum contribution
            - psi_total: Total flux (psi_plasma + psi_coils)
    """
    # measures 顺序与 FreeBndDataset 一致：[coil0..3, Ip, paxis, alpha_m, alpha_n, fvac]
    measures = np.concatenate([coil_currents, params], axis=0)
    measures = param_norm.apply(measures).astype(np.float32)

    # 训练数据为 float32（FreeBndDataset），模型权重也是 float32；
    # 输入必须显式转 float32，否则 double 输入会触发 conv 类型不匹配
    R_t = torch.from_numpy(np.asarray(R, dtype=np.float32)).unsqueeze(0).to(device)  # [1, nx, ny]
    Z_t = torch.from_numpy(np.asarray(Z, dtype=np.float32)).unsqueeze(0).to(device)
    m_t = torch.from_numpy(measures).unsqueeze(0).to(device)  # [1, 9]
    psi_coils_np = compute_psi_coils(coil_currents, greens).astype(np.float32)
    coils_t = torch.from_numpy(psi_coils_np).unsqueeze(0).to(device)  # [1, nx, ny]

    model.eval()
    with torch.no_grad():
        if model_type == "planet":
            pred = model((m_t, R_t, Z_t, coils_t))
        else:
            ufno_input = build_ufno_input(m_t, R_t, Z_t, coils_t)
            pred = model(ufno_input)

    pred_np = pred.squeeze().cpu().numpy()

    if predict_plasma:
        psi_plasma = pred_np
        psi_total = psi_plasma + psi_coils_np
    else:
        psi_total = pred_np
        psi_plasma = psi_total - psi_coils_np

    return {
        "psi_plasma": psi_plasma,
        "psi_coils": psi_coils_np,
        "psi_total": psi_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--R", type=float, nargs="+", default=[0.1, 2.0], help="R range [min, max].")
    parser.add_argument("--Z", type=float, nargs="+", default=[-1.0, 1.0], help="Z range [min, max].")
    parser.add_argument("--nx", type=int, default=65, help="Grid resolution in R.")
    parser.add_argument("--ny", type=int, default=65, help="Grid resolution in Z.")
    parser.add_argument("--coil-currents", type=float, nargs=4, required=True, help="4 coil currents.")
    parser.add_argument("--params", type=float, nargs=5, required=True, help="5 plasma params: Ip paxis alpha_m alpha_n fvac.")
    parser.add_argument("--output", help="Output .npz path for predictions.")
    parser.add_argument("--screening", action="store_true",
                        help="Run physics-based quality checks on the prediction "
                             "(uses screening_thresholds.json next to the checkpoint if present).")
    parser.add_argument("--strict", action="store_true",
                        help="With --screening, exit code 1 if any check FAILs.")
    args = parser.parse_args()

    try:
        import freegs
        from freegs import boundary
    except ImportError:
        print("Error: freegs not found. Please install freegs first.")
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params_config = checkpoint["args"]
    param_norm = Normalization(checkpoint["param_mean"], checkpoint["param_std"])

    # The model is trained on the checkpoint's grid resolution (64x64);
    # --nx/--ny must match it or the prediction is meaningless.
    nr = checkpoint["nr"]
    nz = checkpoint["nz"]
    if args.nx != nr or args.ny != nz:
        print(f"[warning] --nx/--ny ({args.nx}x{args.ny}) != checkpoint grid "
              f"({nr}x{nz}); using checkpoint grid.")
    R = np.linspace(args.R[0], args.R[1], nr)
    Z = np.linspace(args.Z[0], args.Z[1], nz)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")

    tokamak = freegs.machine.TestTokamak()
    greens, coil_names = compute_greens(tokamak, R_grid, Z_grid)
    print(f"Computed Green functions for coils: {coil_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = params_config.get("model", "planet")
    n_measures = checkpoint.get("n_measures", 9)
    predict_plasma = checkpoint.get("predict_plasma", False)

    if model_type == "planet":
        model = PlaNetCore(
            n_measures=n_measures, hidden_dim=params_config.get("hidden_dim", 256),
            nr=nr, nz=nz,
            dropout=params_config.get("dropout", 0.0),
            fourier_freqs=params_config.get("fourier_freqs", 0),
            use_coil_input=params_config.get("use_coil_input", False),
        ).to(device)
    else:
        model = UFNO2d_v2(
            in_channels=12,
            modes1=params_config.get("modes1", 32),
            modes2=params_config.get("modes2", 32),
            width=params_config.get("width", 128),
            layers=params_config.get("layers", 6),
        ).to(device)
    model.load_state_dict(checkpoint["model"])

    coil_currents = np.array(args.coil_currents, dtype=np.float32)
    plasma_params = np.array(args.params, dtype=np.float32)

    print(f"\nModel: {model_type} (predict_plasma={predict_plasma})")
    print("Input parameters:")
    print(f"  Coil currents: {coil_currents}")
    print(f"  Ip: {plasma_params[0]} A")
    print(f"  paxis: {plasma_params[1]} Pa")
    print(f"  alpha_m: {plasma_params[2]}")
    print(f"  alpha_n: {plasma_params[3]}")
    print(f"  fvac: {plasma_params[4]}")

    result = predict_psi(model, model_type, R_grid, Z_grid, coil_currents, plasma_params, param_norm, greens, predict_plasma, device)

    print("\nPrediction stats:")
    print(f"  psi_plasma: min={result['psi_plasma'].min():.4f}, max={result['psi_plasma'].max():.4f}, mean={result['psi_plasma'].mean():.4f}")
    print(f"  psi_coils: min={result['psi_coils'].min():.4f}, max={result['psi_coils'].max():.4f}, mean={result['psi_coils'].mean():.4f}")
    print(f"  psi_total: min={result['psi_total'].min():.4f}, max={result['psi_total'].max():.4f}, mean={result['psi_total'].mean():.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            R=R_grid.astype(np.float32),
            Z=Z_grid.astype(np.float32),
            psi_plasma=result["psi_plasma"].astype(np.float32),
            psi_coils=result["psi_coils"].astype(np.float32),
            psi_total=result["psi_total"].astype(np.float32),
            coil_currents=coil_currents,
            params=plasma_params,
            coil_names=np.array(coil_names),
        )
        print(f"\nPredictions saved to: {args.output}")

    if args.screening:
        from .screening import CheckThresholds, format_report, screen_prediction

        thresholds = CheckThresholds()
        th_path = Path(args.checkpoint).parent / "screening_thresholds.json"
        if th_path.exists():
            import json

            th = json.loads(th_path.read_text())["thresholds"]
            thresholds = CheckThresholds(
                gs_residual=[th["gs_residual"]["warn"], th["gs_residual"]["fail"]],
                boundary_ratio=[th["boundary_ratio"]["warn"], th["boundary_ratio"]["fail"]],
                max_z=[th["max_z"]["warn"], th["max_z"]["fail"]],
            )
        else:
            print("\n[warning] screening_thresholds.json not found next to the "
                  "checkpoint; using no-threshold defaults (run "
                  "`python -m gs_pino.screening --checkpoint ... --calibrate` first).")

        measures_raw = np.concatenate([coil_currents, plasma_params]).astype(np.float64)
        screen = screen_prediction(
            psi_plasma=result["psi_plasma"],
            psi_coils=result["psi_coils"],
            R=R_grid, Z=Z_grid,
            measures_raw=measures_raw,
            param_mean=checkpoint["param_mean"],
            param_std=checkpoint["param_std"],
            params={
                "Ip": plasma_params[0], "paxis": plasma_params[1],
                "alpha_m": plasma_params[2], "alpha_n": plasma_params[3],
                "fvac": plasma_params[4],
            },
            thresholds=thresholds,
        )
        print("\n" + format_report(screen))
        if args.strict and screen["overall"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
