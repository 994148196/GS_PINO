"""Evaluate a trained masked U-FNO model and create diagnostic figures.

This module is intentionally verbose because it is meant to be the main
post-training inspection tool.  It reports scalar metrics on the held-out test
split, writes per-case comparison plots, and creates aggregate plots that show
whether errors correlate with the eight GS input parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GSDataset, Normalization
from .geometry import PARAM_NAMES
from .losses import masked_mse
from .models import UFNO2d


def rel_l2(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return per-sample relative L2 error restricted to the LCFS mask."""
    # Sum over channel and spatial dimensions; batch dimension is preserved.
    num = (((pred - y) ** 2) * mask).sum(dim=(1, 2, 3)).sqrt()
    # Clamp the denominator so empty/near-zero fields cannot produce NaNs.
    den = ((y**2 * mask).sum(dim=(1, 2, 3))).sqrt().clamp_min(1e-8)
    return num / den


def _format_params(params: np.ndarray) -> str:
    """Compactly format all eight input parameters for plot titles."""
    parts: list[str] = []
    for name, value in zip(PARAM_NAMES, params):
        # Ip is much larger than the dimensionless shape/profile parameters.
        if name == "Ip":
            parts.append(f"{name}={value / 1e6:.2f}MA")
        else:
            parts.append(f"{name}={value:.3g}")
    return ", ".join(parts)


def plot_case(path: Path, R: np.ndarray, Z: np.ndarray, true: np.ndarray, pred: np.ndarray, mask: np.ndarray, params: np.ndarray, rel_error: float) -> None:
    """Save true/prediction/error contours for one test equilibrium.

    The LCFS is drawn as a black contour, invalid outside-LCFS pixels are hidden,
    and the title includes all eight parameters so visual failures can be traced
    back to the sampled equilibrium settings.
    """
    # Hide non-physical exterior values so colorbars are controlled by plasma data.
    err = np.where(mask > 0, pred - true, np.nan)
    true = np.where(mask > 0, true, np.nan)
    pred = np.where(mask > 0, pred, np.nan)

    # Three panels make it easy to compare solver truth, surrogate, and residual.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    fig.suptitle(f"Relative L2={rel_error:.3e}\n{_format_params(params)}", fontsize=10)
    for axis, data, title in zip(ax, [true, pred, err], ["true psi_bar", "pred psi_bar", "pred - true"]):
        im = axis.contourf(R, Z, data, levels=48)
        axis.contour(R, Z, mask, levels=[0.5], colors="k", linewidths=1.0)
        axis.set_xlabel("R")
        axis.set_ylabel("Z")
        axis.set_aspect("equal")
        axis.set_title(title)
        fig.colorbar(im, ax=axis)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(output_dir: Path, rel_errors: np.ndarray, params: np.ndarray) -> None:
    """Create dataset-level plots for held-out performance inspection."""
    # Histogram shows the overall error distribution on the test set.
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.hist(rel_errors, bins=min(30, max(5, len(rel_errors) // 2)), color="#4c72b0", alpha=0.85)
    ax.axvline(np.mean(rel_errors), color="k", linestyle="--", label=f"mean={np.mean(rel_errors):.3e}")
    ax.axvline(np.percentile(rel_errors, 95), color="r", linestyle=":", label=f"p95={np.percentile(rel_errors, 95):.3e}")
    ax.set_xlabel("masked relative L2 error")
    ax.set_ylabel("number of cases")
    ax.set_title("Held-out test-set error distribution")
    ax.legend()
    fig.savefig(output_dir / "summary_error_histogram.png", dpi=180)
    plt.close(fig)

    # Parameter scatter plots reveal whether the surrogate struggles in corners
    # of the eight-dimensional input space, e.g. high elongation or high beta_p.
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, name, values in zip(axes.ravel(), PARAM_NAMES, params.T):
        x = values / 1e6 if name == "Ip" else values
        xlabel = "Ip [MA]" if name == "Ip" else name
        axis.scatter(x, rel_errors, s=18, alpha=0.75)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("relative L2")
        axis.set_title(f"error vs {name}")
        axis.grid(alpha=0.25)
    fig.suptitle("Held-out test-set error versus the eight input parameters")
    fig.savefig(output_dir / "summary_error_vs_parameters.png", dpi=180)
    plt.close(fig)


def load_model(checkpoint: dict, in_channels: int, device: torch.device) -> UFNO2d:
    """Rebuild the U-FNO architecture from training-time checkpoint args."""
    model_args = checkpoint["args"]
    model = UFNO2d(
        in_channels,
        model_args["modes1"],
        model_args["modes2"],
        model_args["width"],
        model_args["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    """CLI entry point for model evaluation and visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gs_fixed_boundary.npz", help="Dataset produced by gs_pino.generate_dataset.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint, typically outputs/<run>/best.pt.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory for metrics JSON and PNG plots.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--max-plots", type=int, default=12, help="Maximum number of per-case comparison plots to save.")
    args = parser.parse_args()

    # Create output directories before any expensive work so failures are obvious.
    output_dir = Path(args.output_dir)
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Restore checkpoint and the parameter normalization learned from training.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    test_indices = checkpoint.get("test_indices")
    dataset = GSDataset(args.data, test_indices, Normalization(checkpoint["param_mean"], checkpoint["param_std"]))
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # Use CUDA automatically when available; CPU keeps the workflow portable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, dataset[0][0].shape[0], device)

    # Accumulate predictions only for scalar metrics; per-case plots are generated
    # in a second small loop so the code remains simple and memory efficient.
    mse_values: list[float] = []
    rel_values: list[float] = []
    with torch.no_grad():
        for x, y, mask, _sdf, _params in loader:
            pred = model(x.to(device)).cpu()
            mse_values.append(float(masked_mse(pred, y, mask)))
            rel_values.extend(rel_l2(pred, y, mask).numpy().tolist())

    rel_array = np.asarray(rel_values, dtype=np.float64)
    raw = np.load(args.data)
    test_params = raw["params"][dataset.indices]
    metrics = {
        "masked_mse": float(np.mean(mse_values)),
        "relative_l2_mean": float(np.mean(rel_array)),
        "relative_l2_median": float(np.median(rel_array)),
        "relative_l2_p95": float(np.percentile(rel_array, 95)),
        "relative_l2_max": float(np.max(rel_array)),
        "n_cases": int(len(dataset)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_summary(output_dir, rel_array, test_params)

    # Save detailed comparison figures for the first few test cases.
    for local_idx in range(min(args.max_plots, len(dataset))):
        raw_idx = int(dataset.indices[local_idx])
        x, y, mask, _sdf, params_tensor = dataset[local_idx]
        with torch.no_grad():
            pred = model(x[None].to(device)).cpu().numpy()[0, 0]
        plot_case(
            case_dir / f"case_{local_idx:03d}.png",
            raw["R"][raw_idx],
            raw["Z"][raw_idx],
            y.numpy()[0],
            pred,
            mask.numpy()[0],
            params_tensor.numpy(),
            float(rel_array[local_idx]),
        )


if __name__ == "__main__":
    main()
