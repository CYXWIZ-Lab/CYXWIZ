"""Truth for the Model window's Summary (TOFIX129 D11), for
test_model_summary_presentation.

The five graphs the test builds, as PyTorch modules. For every layer the
test checks:

output         the sample shape in the Engine's order ([H, W, C] for images,
               [L, E] for sequences; torch's [C, H, W] is permuted).
parameters     sum of the module's parameters' numel (learnable only; buffers
               such as BatchNorm's running statistics are not parameters).
multiply_adds  torch.utils.flop_counter.FlopCounterMode's count for one sample
               divided by 2 (it counts 2 FLOPs per multiply-add: matrix
               products, convolutions, attention through the math kernel). Recurrent layers run as one
               aten op that the counter does not split, so for them it is the
               numel of every weight matrix (weight_ih / weight_hh, both
               directions, every stacked layer) times the sequence length:
               each weight entry is used once per step.

Dense layers and Linear use the Engine's own default bias; convolutions are
[H, W, C] in the Engine and torch's Conv2d on [C, H, W] here.

Run with torch 2.10:
  python generate_model_summary_fixtures.py
Writes ../fixtures/model_summary_pytorch.json.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

torch.manual_seed(0)


def flops(module, *inputs):
    module.train()  # the eval fast paths of attention are single ops the counter skips
    # The math attention kernel: the fused CPU kernels are single ops the
    # counter does not count.
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        module(*inputs)
    return counter.get_total_flops()


def params(module):
    return sum(p.numel() for p in module.parameters())


def image(t):
    # [1, C, H, W] -> [H, W, C]
    return [t.shape[2], t.shape[3], t.shape[1]]


def flat(t):
    return list(t.shape[1:])


class Step:
    def __init__(self, name, module, shape=flat, recurrent=False):
        self.name, self.module, self.shape, self.recurrent = name, module, shape, recurrent


def run_chain(steps, x):
    layers = []
    for step in steps:
        out = step.module(x)
        if isinstance(out, tuple):  # LSTM / GRU / RNN
            out = out[0]
        if step.recurrent:
            length = x.shape[1]
            weights = sum(p.numel() for n, p in step.module.named_parameters() if n.startswith("weight"))
            macs = weights * length
        else:
            macs = flops(step.module, x) // 2
        if step.name is not None:
            layers.append({"name": step.name, "output": [int(d) for d in step.shape(out)],
                           "parameters": params(step.module), "multiply_adds": int(macs)})
        x = out
    return layers, x


def mnist_cnn():
    steps = [
        Step("Conv2D 32", nn.Conv2d(1, 32, 3), image),
        Step("ReLU 1", nn.ReLU(), image),
        Step("MaxPool 1", nn.MaxPool2d(2, 2), image),
        Step("Conv2D 64", nn.Conv2d(32, 64, 3), image),
        Step("ReLU 2", nn.ReLU(), image),
        Step("MaxPool 2", nn.MaxPool2d(2, 2), image),
        Step("Flatten", nn.Flatten()),
        Step("Dense 128", nn.Linear(1600, 128)),
        Step("ReLU 3", nn.ReLU()),
        Step("Dropout", nn.Dropout(0.5)),
        Step("Dense 10", nn.Linear(128, 10)),
    ]
    return run_chain(steps, torch.randn(1, 1, 28, 28))[0]


def image_mix():
    steps = [
        Step("Conv same", nn.Conv2d(3, 16, 3, padding=1), image),
        Step("BatchNorm", nn.BatchNorm2d(16), image),
        Step("ReLU", nn.ReLU(), image),
        Step("Depthwise s2", nn.Conv2d(16, 32, 3, stride=2, padding=1, groups=16), image),
        Step("Pointwise", nn.Conv2d(32, 8, 1), image),
        Step("Up", nn.ConvTranspose2d(8, 4, 2, stride=2), image),
        Step("Global pool", nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())),
        Step("Head", nn.Linear(4, 3)),
    ]
    return run_chain(steps, torch.randn(1, 3, 32, 32))[0]


def residual():
    x = torch.randn(1, 8)
    a, b, head = nn.Linear(8, 6), nn.Linear(8, 6), nn.Linear(6, 3)
    layers = []
    for name, module in (("Dense a", a), ("Dense b", b)):
        layers.append({"name": name, "output": flat(module(x)), "parameters": params(module),
                       "multiply_adds": flops(module, x) // 2})
    added = a(x) + b(x)
    layers.append({"name": "Add", "output": flat(added), "parameters": 0, "multiply_adds": 0})
    layers.append({"name": "Head", "output": flat(head(added)), "parameters": params(head),
                   "multiply_adds": flops(head, added) // 2})
    return layers


def recurrent():
    ids = torch.randint(0, 1000, (1, 20))
    embedding = nn.Embedding(1000, 32)
    steps = [
        Step("LSTM", nn.LSTM(32, 24, num_layers=2, bidirectional=True, batch_first=True), recurrent=True),
        Step("GRU", nn.GRU(48, 16, batch_first=True), recurrent=True),
    ]
    x = embedding(ids)
    layers = [{"name": "Embedding", "output": flat(x), "parameters": params(embedding), "multiply_adds": 0}]
    more, x = run_chain(steps, x)
    layers += more
    rnn = nn.RNN(16, 12, batch_first=True)
    weights = sum(p.numel() for n, p in rnn.named_parameters() if n.startswith("weight"))
    last = rnn(x)[0][:, -1, :]
    layers.append({"name": "RNN", "output": flat(last), "parameters": params(rnn),
                   "multiply_adds": weights * x.shape[1]})
    head = nn.Linear(12, 5)
    layers.append({"name": "Head", "output": flat(head(last)), "parameters": params(head),
                   "multiply_adds": flops(head, last) // 2})
    return layers


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)

    def forward(self, x):
        return self.attention(x, x, x, need_weights=False)[0]


def attention():
    ids = torch.randint(0, 500, (1, 16))
    embedding = nn.Embedding(500, 32)
    x = embedding(ids)
    layers = [{"name": "Embedding", "output": flat(x), "parameters": params(embedding), "multiply_adds": 0}]
    steps = [
        Step("Attention", SelfAttention(32, 4)),
        Step("Encoder", nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True)),
        Step("Flatten", nn.Flatten()),
        Step("Head", nn.Linear(512, 2)),
    ]
    layers += run_chain(steps, x)[0]
    return layers


def case(layers):
    return {"layers": layers, "parameters": sum(l["parameters"] for l in layers),
            "multiply_adds": sum(l["multiply_adds"] for l in layers)}


fixture = {
    "torch": torch.__version__,
    "mnist_cnn": case(mnist_cnn()),
    "image_mix": case(image_mix()),
    "residual": case(residual()),
    "recurrent": case(recurrent()),
    "attention": case(attention()),
}
out = Path(__file__).resolve().parent.parent / "fixtures" / "model_summary_pytorch.json"
out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
for name in ("mnist_cnn", "image_mix", "residual", "recurrent", "attention"):
    print(name, fixture[name]["parameters"], fixture[name]["multiply_adds"])
