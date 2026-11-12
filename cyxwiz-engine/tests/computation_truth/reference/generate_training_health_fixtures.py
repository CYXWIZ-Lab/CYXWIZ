"""Truth for training health (TOFIX129 D11), for test_training_health.

Random tensors (seed 17) in the Engine's layouts and what the Health tab
reports for them, by numpy / torch:

  weights      a conv weight [3, 3, 4, 8] and a dense weight [10, 32]:
               ||W|| (Frobenius) and numpy.histogram(W, 24, (min, max)).
  gradients    three steps of gradients for both: the mean over the steps
               of each step's ||g||.
  relu_image   two validation batches of an image ReLU output [5, 5, 8, N]
               ([H, W, C, N]) with channels 2 and 6 zero everywhere:
               channels whose maximum over H, W and N is 0.
  relu_dense   two batches of a dense ReLU output [N, 12] with units 0, 5
               and 11 zero everywhere: units whose maximum over N is 0.

Run with numpy 2.3, torch 2.10:
  python generate_training_health_fixtures.py
Writes ../fixtures/training_health_pytorch.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(17)


def plain(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.reshape(-1)]}


def weight_truth(w, grads):
    hist, edges = np.histogram(w.numpy(), bins=24, range=(float(w.min()), float(w.max())))
    return {"weight": plain(w), "gradients": [plain(g) for g in grads], "norm": float(w.norm()),
            "mean_gradient_norm": float(sum(float(g.norm()) for g in grads) / len(grads)),
            "histogram": hist.tolist(), "low": float(edges[0]), "high": float(edges[-1])}


conv = torch.randn(3, 3, 4, 8) * 0.2
dense = torch.randn(10, 32) * 0.1
conv_grads = [torch.randn(3, 3, 4, 8) * 0.01 * (k + 1) for k in range(3)]
dense_grads = [torch.randn(10, 32) * 0.02 for _ in range(3)]

image_batches = []
for n in (4, 3):
    a = torch.relu(torch.randn(5, 5, 8, n))
    a[:, :, 2, :] = 0
    a[:, :, 6, :] = 0
    image_batches.append(a)
image_max = torch.stack([b.amax(dim=(0, 1, 3)) for b in image_batches]).amax(dim=0)

dense_batches = []
for n in (6, 2):
    a = torch.relu(torch.randn(n, 12))
    a[:, [0, 5, 11]] = 0
    dense_batches.append(a)
dense_max = torch.cat(dense_batches).amax(dim=0)

fixture = {
    "conv": weight_truth(conv, conv_grads),
    "dense": weight_truth(dense, dense_grads),
    "relu_image": {"batches": [plain(b) for b in image_batches], "dead": int((image_max <= 0).sum()), "units": 8},
    "relu_dense": {"batches": [plain(b) for b in dense_batches], "dead": int((dense_max <= 0).sum()), "units": 12},
}
out = Path(__file__).resolve().parent.parent / "fixtures" / "training_health_pytorch.json"
out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
print("dead", fixture["relu_image"]["dead"], fixture["relu_dense"]["dead"], "norms", fixture["conv"]["norm"], fixture["dense"]["norm"])
