"""Inference module for the free-boundary GS PINO surrogate.

Provides functions to predict psi_total from coil currents and plasma parameters,
using freegs to compute the vacuum contribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data_freegs import build_input, Normalization
from .models import UFNO2d


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
    model: UFNO2d,
    R: np.ndarray,
    Z: np.ndarray,
    coil_currents: np.ndarray,
    params: np.ndarray,
    param_norm: Normalization,
    greens: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """Predict psi_total from coil currents and plasma parameters.
    
    Args:
        model: Trained UFNO2d model
        R: 2D array of major radius coordinates [nx, ny]
        Z: 2D array of height coordinates [nx, ny]
        coil_currents: [4] array of coil currents
        params: [5] array of plasma parameters [Ip, paxis, alpha_m, alpha_n, fvac]
        param_norm: Normalization object for parameters
        greens: [4, nx, ny] array of Green functions
        device: torch device
        
    Returns:
        Dictionary with:
            - psi_plasma: Predicted plasma contribution
            - psi_coils: Vacuum contribution
            - psi_total: Total flux (psi_plasma + psi_coils)
    """
    sample = {
        "R": R,
        "Z": Z,
        "coil_currents": coil_currents,
        "params": params,
    }
    
    x = build_input(sample, param_norm.mean, param_norm.std)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_plasma = model(x_tensor)
    
    psi_plasma = pred_plasma.squeeze().cpu().numpy()
    psi_coils = compute_psi_coils(coil_currents, greens)
    psi_total = psi_plasma + psi_coils
    
    return {
        "psi_plasma": psi_plasma,
        "psi_coils": psi_coils,
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
    model = UFNO2d(11, params_config["modes1"], params_config["modes2"], params_config["width"], params_config["layers"]).to(device)
    model.load_state_dict(checkpoint["model"])

    coil_currents = np.array(args.coil_currents, dtype=np.float32)
    plasma_params = np.array(args.params, dtype=np.float32)

    print("\nInput parameters:")
    print(f"  Coil currents: {coil_currents}")
    print(f"  Ip: {plasma_params[0]} A")
    print(f"  paxis: {plasma_params[1]} Pa")
    print(f"  alpha_m: {plasma_params[2]}")
    print(f"  alpha_n: {plasma_params[3]}")
    print(f"  fvac: {plasma_params[4]}")

    result = predict_psi(model, R_grid, Z_grid, coil_currents, plasma_params, param_norm, greens, device)

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
