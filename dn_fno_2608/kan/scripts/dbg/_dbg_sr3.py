"""Quantify h0/h1 error distribution (corner-localized vs diffuse)."""
import importlib.util
import numpy as np
import torch

from gs_pino_kan_2608.model_kan import build_model
from gs_pino_kan_2608.data_kan import PointKANDataset

CKPT = "dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/best.pt"
SR = "dn_fno_2608/kan/experiments/exp001_kan_v5/sr/symbolic_gs_kan.py"
TEST = "dn_fno_2608/data_v5/sn/test.npz"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
         for k, v in ckpt["stats"].items()}
model = build_model(**ckpt.get("arch", {})).to("cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()
ds = PointKANDataset(TEST, stats=stats)

spec = importlib.util.spec_from_file_location("symbolic_gs_kan", SR)
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

tot_h0, tot_h1 = 0.0, 0.0
for i in range(8):
    b = ds.full_grid(i)
    xn = b["x"].numpy()
    with torch.no_grad():
        h0_t = model.layer0(b["x"]).numpy()
        out_t = model(b["x"]).numpy()
    h0_s = np.zeros_like(h0_t)
    for o, j, e in sr._LAYER0:
        h0_s[:, o] += e(np.clip(xn[:, j], sr._LO0[j], sr._HI0[j]))
    h1_s = np.zeros_like(out_t)
    for o, j, e in sr._LAYER1:
        h1_s[:, o] += e(np.clip(h0_s[:, j], sr._LO1[j], sr._HI1[j]))
    d0 = np.abs(h0_s - h0_t); d1 = np.abs(h1_s - out_t)
    # corner points = x==1.0 or -1.0 on R/Z channels
    corner = (xn[:, 0] >= 1.0 - 1e-9) | (xn[:, 0] <= -1.0 + 1e-9) | \
             (xn[:, 1] >= 1.0 - 1e-9) | (xn[:, 1] <= -1.0 + 1e-9)
    tot_h0 += d0.mean(); tot_h1 += d1.mean()
    print(f"sample {i}: h0 mean|d| {d0.mean():.4f} (corner {d0[corner].mean():.4f} / "
          f"inner {d0[~corner].mean():.4f}) | h1 mean|d| {d1.mean():.4f} | "
          f"corner pts {corner.sum()}/{len(xn)}")
