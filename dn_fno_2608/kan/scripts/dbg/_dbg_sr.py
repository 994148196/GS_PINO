"""Debug KAN-SR explosion: compare torch model vs generated module layer by layer."""
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
b = ds.full_grid(0)
x = b["x"]  # (4225, 19)

with torch.no_grad():
    h0_t = model.layer0(x).numpy()
    out_t = model(x).numpy()

spec = importlib.util.spec_from_file_location("symbolic_gs_kan", SR)
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
xn = x.numpy()
h0_s = np.zeros((xn.shape[0], h0_t.shape[1]))
for o, i, e in sr._LAYER0:
    h0_s[:, o] += e(xn[:, i])
h1_s = np.zeros((xn.shape[0], 2))
for o, i, e in sr._LAYER1:
    h1_s[:, o] += e(h0_s[:, i])

print("h0 torch range:", h0_t.min(), h0_t.max())
print("h0 sym   range:", np.nanmin(h0_s), np.nanmax(h0_s))
print("h0 max abs diff:", np.abs(h0_s - h0_t).max())
print("h1 torch range:", out_t.min(), out_t.max())
print("h1 sym   range:", np.nanmin(h1_s), np.nanmax(h1_s))
print("h1 max abs diff:", np.abs(h1_s - out_t).max())

# which hidden nodes feed the worst layer1 edges
d = np.abs(h0_s - h0_t)
for j in range(h0_t.shape[1]):
    dj = d[:, j].max()
    if dj > 0.1:
        print(f"  hidden node {j}: max diff {dj:.3g}  torch range "
              f"[{h0_t[:, j].min():.3g},{h0_t[:, j].max():.3g}]")
