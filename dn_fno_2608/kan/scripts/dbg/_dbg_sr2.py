"""Find the worst h0 points: which input channel/edge causes the blow-up."""
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
x = b["x"]

with torch.no_grad():
    h0_t = model.layer0(x).numpy()

spec = importlib.util.spec_from_file_location("symbolic_gs_kan", SR)
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
xn = x.numpy()
h0_s = np.zeros_like(h0_t)
for o, i, e in sr._LAYER0:
    h0_s[:, o] += e(np.clip(xn[:, i], sr._LO0[i], sr._HI0[i]))

d = np.abs(h0_s - h0_t)
j = d.argmax() // d.shape[1], d.argmax() % d.shape[1]
node = j[1]
print(f"worst: point {j[0]} node {node} diff {d[j]:.4g}  torch {h0_t[j]:.4g} sym {h0_s[j]:.4g}")
print("input channels at that point (min/max over data in parens):")
# per-edge torch activation for this node at that point
with torch.no_grad():
    _, acts, _, _ = model.layer0.forward_activations(x[j[0]:j[0]+1])
acts = acts[0, :, node].numpy()
# symbolic per-edge
sym_edges = np.zeros(19)
for o, i, e in sr._LAYER0:
    if o == node:
        sym_edges[i] = e(np.clip(xn[j[0], i], sr._LO0[i], sr._HI0[i]))
for i in range(19):
    if abs(acts[i] - sym_edges[i]) > 0.05:
        lo, hi = sr._LO0[i], sr._HI0[i]
        print(f"  ch{i:2d} x={xn[j[0], i]:+.4f} (obs [{lo:+.3f},{hi:+.3f}]) "
              f"torch_edge={acts[i]:+.4f} sym_edge={sym_edges[i]:+.4f} diff={acts[i]-sym_edges[i]:+.4f}")
