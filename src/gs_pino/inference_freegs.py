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
    
    Args:
        tokamak: freegs tokamak object
        R: 2D array of major radius coordinates
        Z: 2D array of height coordinates
        
    Returns:
        greens: [n_coils, nx, ny] array of Green functions
        coil_names: list of coil names
    """
    greens = []
    coil_names = []
    for label, coil in tokamak.coils:
        if coil.control:
            greens.append(coil.createPsiGreens(R, Z).astype(np.float32))
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

    R = np.linspace(args.R[0], args.R[1], args.nx)
    Z = np.linspace(args.Z[0], args.Z[1], args.ny)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")

    tokamak = freegs.machine.TestTokamak()
    greens, coil_names = compute_greens(tokamak, R_grid, Z_grid)
    print(f"Computed Green functions for coils: {coil_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = params_config.get("model", "planet")
    n_measures = checkpoint.get("n_measures", 9)
    nr = checkpoint.get("nr", args.nx)
    nz = checkpoint.get("nz", args.ny)
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


if __name__ == "__main__":
    main()
